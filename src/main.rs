mod ablate;
mod abstraction;
mod aivat;
mod benchmark;
mod bot;
mod br;
mod cards;
mod cfr;
mod clone;
mod compact;
mod bucket_check;
mod depth_match;
mod offmenu_match;
mod distill;
mod engine;
mod eval;
mod flop;
mod lbr;
mod mirror;
mod net;
mod play;
mod preflop;
// Exercised by its own tests; no command drives the bandit yet.
#[cfg(test)]
mod portfolio;
mod river;
mod search;
mod serve;
mod slumbot;
mod table;
mod turn;
mod valuenet;

// Solver walks allocate range-vector scratch at every node visit; mimalloc
// is measurably faster than the system allocator on that churn.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use abstraction::{AbsConfig, Abstraction, Centroids};
use bot::{Policy, SearchParams};
use cfr::{Blueprint, TrainConfig, Trainer};
use clap::{Parser, Subcommand};
use abstraction::MenuShape;
use engine::{HandConfig, Players, Stack};
use indicatif::{ProgressBar, ProgressStyle};
use std::num::{NonZeroU16, NonZeroU32, NonZeroUsize};
use std::sync::Arc;
use table::Baseline;

/// `serve --also PATH:STACK`: another blueprint and the stack it was
/// trained at.
#[derive(Debug, Clone)]
struct DepthBlueprint {
    path: String,
    stack: Stack,
}

impl std::str::FromStr for DepthBlueprint {
    type Err = String;

    fn from_str(s: &str) -> Result<DepthBlueprint, String> {
        let (path, stack) = s.rsplit_once(':').ok_or_else(|| format!("expected PATH:STACK, got '{s}'"))?;
        Ok(DepthBlueprint { path: path.to_string(), stack: stack.parse()? })
    }
}

#[derive(Parser)]
#[command(
    name = "pluribus",
    about = "Pluribus-style no-limit hold'em bot: Linear MCCFR blueprint + online subgame resolving"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Train a blueprint strategy with parallel Linear MCCFR.
    Train {
        /// MCCFR iterations (traversals).
        #[arg(long, default_value_t = 1_000_000)]
        iters: u64,
        /// Output blueprint file.
        #[arg(long, default_value = "blueprint.bin")]
        out: String,
        #[arg(long, default_value_t = Players::SIX)]
        players: Players,
        /// Starting stack in chips (blinds are 50/100, so 10,000 = 100bb
        /// — the Pluribus setup — and 20,000 = 200bb — the Slumbot / GTO
        /// Wizard benchmark setup).
        #[arg(long, default_value_t = Stack::DEFAULT)]
        stack: Stack,
        /// Resume from a training checkpoint.
        #[arg(long)]
        resume: Option<String>,
        /// Also write a training checkpoint here (enables later --resume).
        #[arg(long)]
        checkpoint: Option<String>,
        /// Postflop equity buckets per street.
        #[arg(long, default_value_t = NonZeroU16::new(12).unwrap())]
        buckets: NonZeroU16,
        /// Bet-size menu shape (stored in the blueprint): pluribus (coarse
        /// postflop, Pluribus's shape; the default, measured least
        /// exploitable), wide (2026-07 menu), pluribus-fine-pre (also a
        /// fine preflop menu; worse at every budget tested), or uniform
        /// (33/50/100/150/200% + all-in at every decision, any raise count).
        #[arg(long, value_enum, default_value_t = MenuShape::Pluribus)]
        menu: MenuShape,
        /// Monte Carlo rollouts per river equity estimate.
        #[arg(long, default_value_t = NonZeroU32::new(200).unwrap())]
        rollouts: NonZeroU32,
        /// Sampled future boards per flop/turn distribution estimate.
        #[arg(long, default_value_t = NonZeroU32::new(24).unwrap())]
        runouts: NonZeroU32,
        /// Preloaded flop/turn bucket table from `bucket-table` (same
        /// buckets/rollouts/runouts): skips centroid training and the
        /// Monte Carlo bucketing that otherwise takes ~70% of training.
        #[arg(long)]
        bucket_table: Option<String>,
        /// Situations sampled per street when training k-means centroids.
        #[arg(long, default_value_t = NonZeroUsize::new(30_000).unwrap())]
        kmeans_samples: NonZeroUsize,
        /// Use raw-equity bucketing instead of EMD k-means clustering.
        #[arg(long)]
        raw_buckets: bool,
        /// Cluster hands by the combined potential-aware feature (equity
        /// quantiles + opponent-cluster hand strength) instead of equity
        /// quantiles alone — a richer, opponent-relative card abstraction.
        /// Mutually exclusive with the other abstraction modes.
        #[arg(long, conflicts_with_all = ["raw_buckets", "strategic_from"])]
        ochs: bool,
        /// Strategy-aware co-training: cluster hands by how THIS previous
        /// blueprint plays and realizes them, instead of by equity
        /// distributions. The resulting blueprint must be loaded with
        /// --strat-prev pointing at the same file.
        #[arg(long)]
        strategic_from: Option<String>,
        /// Traversal RNG seed: distinct seeds give independent self-play
        /// runs (equilibrium-selection studies).
        #[arg(long, default_value_t = 0)]
        train_seed: u64,
        /// Restricted Nash response: opponent model (random | caller).
        #[arg(long)]
        rnr_model: Option<Baseline>,
        /// Restricted Nash response against a cloned opponent given as a
        /// blueprint file (e.g. slumbot_clone.bin). Overrides --rnr-model;
        /// the trainer adopts the clone's abstraction so keys align.
        #[arg(long, conflicts_with = "rnr_model")]
        rnr_opponent: Option<String>,
        /// RNR mixture weight p (0 = plain equilibrium, 1 = pure best
        /// response to the model). Requires --rnr-model or --rnr-opponent.
        #[arg(long, default_value_t = 0.5)]
        rnr_p: f64,
        /// Disable negative-regret pruning.
        #[arg(long)]
        no_prune: bool,
        /// Regret below which an action may be pruned (chips, linearly
        /// weighted). Absolute, so a long run reaches it for almost every
        /// negative-regret action; scale it with the run length.
        #[arg(long, default_value_t = -3.0e8, allow_negative_numbers = true)]
        prune_threshold: f64,
        /// VR-MCCFR: learned control-variate baselines at sampled opponent
        /// nodes (Schmid et al. 2019). Unbiased; cuts the variance of every
        /// regret update, so each visit is worth more.
        #[arg(long)]
        vr_baseline: bool,
        /// Legacy pruning: decide per action and prune everywhere, including
        /// the river and terminal-leading actions. The default is Pluribus's
        /// rules (per traversal, river and terminal actions exempt), which
        /// halved the BR exploitability bound at equal iterations.
        #[arg(long, conflicts_with = "no_prune")]
        per_action_prune: bool,
        /// Pluribus's postflop blueprint: mean of periodic snapshots of the
        /// current strategy (after a 10% warm-up, every 5% of iterations)
        /// instead of the linear running average; preflop keeps the average.
        #[arg(long)]
        snapshot_avg: bool,
        /// Multiway-focused sampling weight f in [0,1): opponents' preflop
        /// folds are suppressed (actions drawn from (1-f)*strategy +
        /// f*strategy-renormalised-over-non-fold), importance-weighted so
        /// the equilibrium is unchanged; multiway lines get trained more
        /// per iteration. 0 = off.
        #[arg(long, default_value_t = 0.0)]
        multiway_focus: f64,
        /// Cap on the Linear CFR iteration weight (weight = min(t, cap));
        /// Pluribus stopped linear weighting after 400 minutes. Default:
        /// uncapped.
        #[arg(long, default_value_t = u64::MAX)]
        linear_cap: u64,
        /// Regret floor in chips per unit of the current linear weight
        /// (floor = -C * t) instead of the absolute 2 * --prune-threshold,
        /// which a long run makes almost zero relative to its updates.
        #[arg(long)]
        floor_per_weight: Option<f64>,
        /// Apply --floor-per-weight to preflop infosets only.
        #[arg(long, requires = "floor_per_weight")]
        floor_preflop_only: bool,
        /// Pruning threshold in chips per unit of the current linear weight
        /// (threshold = -C * t) instead of the absolute --prune-threshold.
        #[arg(long)]
        prune_per_weight: Option<f64>,
        /// Preflop exploration weight e in [0,1): other players' preflop
        /// actions are sampled from (1-e)*strategy + e*uniform,
        /// importance-weighted so the fixed point is unchanged; rare
        /// preflop branches are trained more often. 0 = off.
        #[arg(long, default_value_t = 0.0)]
        preflop_explore: f64,
        /// Preflop average from own-reach sweeps every this many iterations
        /// (Pluribus's UPDATE-STRATEGY, enumerated) instead of accumulating
        /// at sampled visits, which weights it by the other players' reach.
        /// 0 = off.
        #[arg(long, default_value_t = 0)]
        preflop_sweep_every: u64,
        /// Also save the blueprint at these iteration counts, as
        /// `<out>.<iterations>` (comma-separated, e.g. 100000000,200000000).
        #[arg(long, value_delimiter = ',')]
        save_at: Vec<u64>,
        /// Worker threads (default: all cores).
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Precompute every canonical flop/turn card bucket for the default
    /// abstraction into a table that `train --bucket-table` preloads.
    BucketTable {
        #[arg(long, default_value = "buckets.bin")]
        out: String,
        /// Postflop equity buckets per street.
        #[arg(long, default_value_t = NonZeroU16::new(12).unwrap())]
        buckets: NonZeroU16,
        #[arg(long, default_value_t = NonZeroU32::new(200).unwrap())]
        rollouts: NonZeroU32,
        #[arg(long, default_value_t = NonZeroU32::new(24).unwrap())]
        runouts: NonZeroU32,
        #[arg(long, default_value_t = NonZeroUsize::new(30_000).unwrap())]
        kmeans_samples: NonZeroUsize,
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Play interactively against the bot (you are seat 0).
    Play {
        #[arg(long, default_value = "blueprint.bin")]
        blueprint: String,
        #[arg(long, default_value_t = Players::SIX)]
        players: Players,
        /// Online subgame resolving for the bots' postflop decisions is
        /// on by default (A3 gate, Aug 2026); this flag is kept for
        /// scripts and is a no-op.
        #[arg(long, hide = true)]
        search: bool,
        /// Bots play the raw blueprint (no online resolving).
        #[arg(long)]
        no_search: bool,
        /// Time budget per searched decision, in milliseconds.
        #[arg(long, default_value_t = 2_000)]
        search_ms: u64,
        /// Model opponents as lambda-rational (logit QRE) during search:
        /// 0 = uniform random, higher = more rational; omit for equilibrium.
        #[arg(long)]
        qre_lambda: Option<f64>,
        /// Belief-state value net enabling ReBeL-style flop solving.
        #[arg(long)]
        value_net: Option<String>,
        /// Safe (gadget) river resolving with rollout-estimated safety
        /// values — bounds exploitation when tracked ranges are wrong.
        #[arg(long)]
        safe_resolve: bool,
        /// Metareasoning: probe-solve first and exit early on decisions
        /// that are already near-pure.
        #[arg(long)]
        adaptive_search: bool,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Answer bot decisions for another program: JSON lines on stdin and
    /// stdout (see src/serve.rs). Blueprint only, no online search.
    Serve {
        #[arg(long, default_value = "blueprint.bin")]
        blueprint: String,
        /// The stack the blueprint was trained at, in chips; deeper stacks
        /// are played as if this deep.
        #[arg(long, default_value_t = Stack::DEFAULT)]
        stack: Stack,
        /// Another blueprint and its training stack, as PATH:STACK
        /// (repeatable). Each request uses the blueprint nearest the acting
        /// seat's effective stack.
        #[arg(long)]
        also: Vec<DepthBlueprint>,
        /// Re-solve postflop decisions online (every served blueprint, any
        /// format; compact ones stay memory-mapped).
        #[arg(long)]
        search: bool,
        /// Per-decision search budget in milliseconds (with --search).
        #[arg(long, default_value_t = 75)]
        search_ms: u64,
        /// Per-decision cap on multiway resolve iterations (with --search).
        #[arg(long, default_value_t = 20_000)]
        search_iters: u64,
        /// Precomputed flop/turn bucket table to prefill the card
        /// abstraction (with --search; avoids cold-cache bucketing). Not
        /// needed for a compact blueprint that includes its table.
        #[arg(long)]
        bucket_table: Option<String>,
        /// Play two-player turns from the blueprint, with its full bet menu,
        /// instead of the exact turn+river solve and its slim menu (A/B arm;
        /// measured more exploitable).
        #[arg(long)]
        turn_blueprint: bool,
    },
    /// Export a blueprint in the compact, memory-mapped serving format.
    Compact {
        #[arg(long, default_value = "blueprint.bin")]
        blueprint: String,
        #[arg(long)]
        out: String,
        /// Round each action's share to the nearest 1/STEPS. 20 plays mixes
        /// in 5% steps, dropping the small residue training leaves on every
        /// action; 255 keeps the strategy as trained.
        #[arg(long, default_value_t = 255, value_parser = clap::value_parser!(u8).range(1..))]
        steps: u8,
        /// Store this flop/turn bucket table (from `bucket-table`, built for
        /// the blueprint's abstraction) in the file, so serving reads it in
        /// place instead of computing buckets or loading a separate table.
        #[arg(long)]
        bucket_table: Option<String>,
    },
    /// Measure what playing blueprint A at stacks it was not trained for
    /// costs, against a field of blueprint B trained at those stacks.
    DepthMatch {
        #[arg(long)]
        a: String,
        /// A's training stack in chips.
        #[arg(long, default_value_t = Stack::DEFAULT)]
        a_stack: Stack,
        #[arg(long)]
        b: String,
        /// B's training stack in chips.
        #[arg(long)]
        b_stack: Stack,
        /// Real stacks at the table, in chips.
        #[arg(long)]
        stack: Stack,
        #[arg(long, default_value_t = 100_000)]
        hands: u64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Re-solve preflop with the blueprint's postflop play frozen for every
    /// player: MCCFR on the preflop tree only, each traversal finishing the
    /// hand by sampling the base blueprint postflop. Writes a preflop-only
    /// blueprint; serve it as `--blueprint BASE+PREFLOP` or merge it with
    /// `merge-preflop`.
    RefinePreflop {
        /// Base blueprint (training or compact) whose postflop is frozen.
        #[arg(long)]
        blueprint: String,
        #[arg(long)]
        bucket_table: Option<String>,
        #[arg(long, default_value_t = 100_000_000)]
        iters: u64,
        #[arg(long)]
        out: String,
        #[arg(long, default_value_t = Stack::DEFAULT)]
        stack: Stack,
        /// Regret floor per unit of linear weight (see train).
        #[arg(long, default_value_t = 3.0e8)]
        floor_per_weight: f64,
        /// Disable negative-regret pruning.
        #[arg(long)]
        no_prune: bool,
        #[arg(long, default_value_t = 0.0)]
        preflop_explore: f64,
        /// Sweep the preflop average (own-reach weighting) every this many
        /// iterations; 0 = accumulate at sampled visits instead.
        #[arg(long, default_value_t = 0)]
        sweep_every: u64,
        #[arg(long, default_value_t = 0)]
        train_seed: u64,
        #[arg(long, value_delimiter = ',')]
        save_at: Vec<u64>,
        /// At every save, print current vs average strategies at this
        /// preflop line (preflop-probe syntax) for --report-hands.
        #[arg(long)]
        report_line: Option<String>,
        #[arg(long, default_value = "AA,KK,QQ,TT,AKo,KQo,Q8o,86o")]
        report_hands: String,
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Replace a training blueprint's preflop strategies with a
    /// `refine-preflop` result (infosets the refinement never reached keep
    /// the base strategy).
    MergePreflop {
        #[arg(long)]
        base: String,
        #[arg(long)]
        preflop: String,
        #[arg(long)]
        out: String,
    },
    /// The blueprint's strategy for every starting hand at one preflop
    /// node, with the all-in share overall and for KK, QQ, JJ, TT, AKs, AKo.
    PreflopProbe {
        #[arg(long)]
        blueprint: String,
        /// The blueprint's training stack in chips.
        #[arg(long, default_value_t = Stack::DEFAULT)]
        stack: Stack,
        /// Preflop actions from the start of the hand, button in seat 0
        /// (UTG acts first): f, c, rN (raise BET_SIZES[N] of the pot), a.
        /// "f f f r3 f" is the big blind facing a 2.9bb button open.
        #[arg(long)]
        line: String,
        /// Print every hand class, not just the summary and premiums.
        #[arg(long)]
        all: bool,
    },
    /// How often training reaches every preflop decision of the menu's
    /// tree under the blueprint's own strategies: estimated regret updates
    /// per combo, grouped into tiers, for the decisions `serve` answers
    /// from the blueprint and those it gives to the preflop all-in resolve
    PreflopReach {
        #[arg(long)]
        blueprint: String,
        /// The blueprint's training stack in chips.
        #[arg(long, default_value_t = Stack::DEFAULT)]
        stack: Stack,
        /// Iterations that trained the preflop strategies (800M for
        /// it-800m, 1B for its refine-preflop overlay).
        #[arg(long, default_value_t = 800_000_000)]
        iters: u64,
        /// Also write every decision as CSV here.
        #[arg(long)]
        csv: Option<String>,
        /// A second blueprint of the same abstraction, e.g. another seed:
        /// how far its strategies are from the first's at each decision,
        /// per tier (the noise left at that many updates).
        #[arg(long)]
        compare: Option<String>,
    },
    /// What each action at a preflop node is worth to a hand class when
    /// everyone, the hero included, plays the blueprint afterwards, against
    /// the blueprint's own ranges for the line.
    ActionEv {
        #[arg(long)]
        blueprint: String,
        #[arg(long, default_value_t = Stack::DEFAULT)]
        stack: Stack,
        /// Preflop line, as for preflop-probe.
        #[arg(long)]
        line: String,
        /// Comma-separated hand classes, or "all" for a node summary: the
        /// best response's gain over the blueprint's mix, averaged over
        /// the hands that reach the node.
        #[arg(long, default_value = "KK,QQ,JJ,TT,AKs,AKo")]
        hands: String,
        /// Deals per hand class.
        #[arg(long, default_value_t = 200_000)]
        deals: u64,
        /// Preloaded flop/turn bucket table (speed only).
        #[arg(long)]
        bucket_table: Option<String>,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Compare lazy flop/turn bucketing with a bucket table.
    BucketCheck {
        #[arg(long)]
        table: String,
        #[arg(long, default_value_t = 100_000)]
        hands: u64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Measure what resizing bets off the menu wins against bots that
    /// translate sizes to the log-nearest menu size.
    OffmenuMatch {
        #[arg(long)]
        blueprint: String,
        /// The blueprint's training stack in chips; the table plays at it.
        #[arg(long, default_value_t = Stack::DEFAULT)]
        stack: Stack,
        #[arg(long, value_enum)]
        resize: offmenu_match::Resize,
        #[arg(long, default_value_t = 100_000)]
        hands: u64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Measure blueprint winrate against baseline opponents (mbb/hand).
    Eval {
        #[arg(long, default_value = "blueprint.bin")]
        blueprint: String,
        #[arg(long, default_value_t = 100_000)]
        hands: u64,
        /// Opponent type in the other seats: random | caller | maniac |
        /// shover | push-fold | nit.
        #[arg(long, default_value = "random")]
        baseline: Baseline,
        #[arg(long, default_value_t = Players::SIX)]
        players: Players,
        /// Duplicate mode: play each deal once per seat (hero rotating) and
        /// average within the deal — large variance reduction, same estimand.
        #[arg(long)]
        duplicate: bool,
        /// AIVAT mode: unbiased estimator with chance/decision correction
        /// terms (Burch et al. 2018) — strongest variance reduction.
        #[arg(long)]
        aivat: bool,
        /// Hero uses range-tracked online search per decision (slow;
        /// combine with --search-ms).
        #[arg(long)]
        search: bool,
        /// Paired search-gain mode: each deal played twice (hero searching
        /// vs hero on blueprint, everyone else blueprint); reports the mean
        /// per-deal difference — the value added by search itself.
        #[arg(long)]
        search_gain: bool,
        /// Paired net-gain mode: search WITH the value net vs search
        /// without it — isolates the value network's contribution.
        #[arg(long)]
        net_gain: bool,
        /// Per-decision search budget in milliseconds (with --search).
        #[arg(long, default_value_t = 100)]
        search_ms: u64,
        /// Per-decision search iteration cap (with --search).
        #[arg(long, default_value_t = 2_000_000)]
        search_iters: u64,
        /// Belief-state value net for ReBeL flop solving (with --search).
        #[arg(long)]
        value_net: Option<String>,
        /// Previous blueprint for strategic-abstraction lookups.
        #[arg(long)]
        strat_prev: Option<String>,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Tighter exploitability lower bound than `lbr`: same harness, but
    /// turn and river decisions play an exact best response of the whole
    /// remaining game (single expectimax pass, deterministic per seed).
    Br {
        #[arg(long, default_value = "blueprint.bin")]
        blueprint: String,
        #[arg(long, default_value_t = 2_000)]
        hands: u64,
        /// Board completions per equity estimate on the greedy (preflop/
        /// flop) streets.
        #[arg(long, default_value_t = NonZeroU32::new(100).unwrap())]
        runouts: NonZeroU32,
        /// Previous blueprint for strategic-abstraction lookups.
        #[arg(long)]
        strat_prev: Option<String>,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// The bot resolves postflop decisions online (the probe then
        /// faces the bot as it actually plays, not the raw blueprint).
        #[arg(long)]
        search: bool,
        /// Per-decision search budget in milliseconds (with --search).
        #[arg(long, default_value_t = 2_000)]
        search_ms: u64,
        /// Per-decision search iteration cap (with --search); a cap makes
        /// the probe independent of machine load.
        #[arg(long, default_value_t = 2_000_000)]
        search_iters: u64,
        /// Root multiway resolves at the current decision instead of the
        /// start of the betting round (the pre-A3 behaviour; A/B arm).
        #[arg(long)]
        decision_root: bool,
        /// Play two-player turns from the blueprint, with its full bet menu,
        /// instead of the exact turn+river solve and its slim menu (A/B arm;
        /// measured more exploitable).
        #[arg(long)]
        turn_blueprint: bool,
        /// Postflop card buckets inside multiway resolves (equity
        /// quantiles; exact per board on the river). Default: the
        /// blueprint's.
        #[arg(long)]
        search_buckets: Option<NonZeroU16>,
        /// Starting stack in chips: the depth the blueprint was trained at.
        #[arg(long, default_value_t = Stack::DEFAULT)]
        stack: Stack,
    },
    /// Lower-bound the blueprint's exploitability with a Local Best Response
    /// agent (Lisý & Bowling 2017): heads-up blind vs blind, other seats fold.
    Lbr {
        #[arg(long, default_value = "blueprint.bin")]
        blueprint: String,
        #[arg(long, default_value_t = 20_000)]
        hands: u64,
        /// Multiway probe: LBR in one rotating seat against the bot in
        /// every other seat (pots go multiway); the default probe is
        /// heads-up blind-vs-blind with the other seats folding.
        #[arg(long)]
        multiway: bool,
        /// Board completions sampled per equity estimate.
        #[arg(long, default_value_t = NonZeroU32::new(100).unwrap())]
        runouts: NonZeroU32,
        /// Previous blueprint for strategic-abstraction lookups.
        #[arg(long)]
        strat_prev: Option<String>,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// The bot resolves postflop decisions online (the probe then
        /// faces the bot as it actually plays, not the raw blueprint).
        #[arg(long)]
        search: bool,
        /// Per-decision search budget in milliseconds (with --search).
        #[arg(long, default_value_t = 2_000)]
        search_ms: u64,
        /// Per-decision search iteration cap (with --search); a cap makes
        /// the probe independent of machine load.
        #[arg(long, default_value_t = 2_000_000)]
        search_iters: u64,
        /// Root multiway resolves at the current decision instead of the
        /// start of the betting round (the pre-A3 behaviour; A/B arm).
        #[arg(long)]
        decision_root: bool,
        /// Play two-player turns from the blueprint, with its full bet menu,
        /// instead of the exact turn+river solve and its slim menu (A/B arm;
        /// measured more exploitable).
        #[arg(long)]
        turn_blueprint: bool,
        /// Postflop card buckets inside multiway resolves (equity
        /// quantiles; exact per board on the river). Default: the
        /// blueprint's.
        #[arg(long)]
        search_buckets: Option<NonZeroU16>,
        /// Starting stack in chips: the depth the blueprint was trained at.
        #[arg(long, default_value_t = Stack::DEFAULT)]
        stack: Stack,
    },
    /// Safety ablation: unsafe vs gadget river resolving under corrupted
    /// range beliefs — reports best-response margins beyond safety values.
    AblateSafety {
        #[arg(long, default_value_t = 40)]
        spots: usize,
        #[arg(long, default_value_t = 400)]
        iters: u64,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Generate exactly-solved turn spots (blueprint self-play + turn
    /// solver) as training data for the belief-state value network.
    GenTurnData {
        #[arg(long, default_value = "blueprint.bin")]
        blueprint: String,
        #[arg(long, default_value = "turn_data.bin")]
        out: String,
        #[arg(long, default_value_t = 10_000)]
        samples: usize,
        /// Vector-CFR iterations per exact turn solve.
        #[arg(long, default_value_t = 200)]
        solve_iters: u64,
        /// Per-solve wall-clock cap in milliseconds.
        #[arg(long, default_value_t = 30_000)]
        solve_ms: u64,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Train the belief-state value network on solved turn spots.
    TrainValueNet {
        /// Training data file(s); comma-separated files are concatenated.
        #[arg(long, default_value = "turn_data.bin")]
        data: String,
        #[arg(long, default_value = "value_net.bin")]
        out: String,
        /// Hidden layer sizes, comma-separated.
        #[arg(long, default_value = "512,512")]
        hidden: String,
        #[arg(long, default_value_t = 50)]
        epochs: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f32,
        #[arg(long, default_value_t = 128)]
        batch: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Expert-iteration distillation: self-play with online search, then
    /// blend the resolved distributions back into the blueprint (the
    /// flywheel). Evaluate the output with br/crossplay/eval before use.
    Distill {
        #[arg(long, default_value = "blueprint.bin")]
        blueprint: String,
        #[arg(long, default_value = "distilled.bin")]
        out: String,
        /// Self-play hands (every seat searches).
        #[arg(long, default_value_t = 20_000)]
        hands: u64,
        /// Per-decision search budget in milliseconds.
        #[arg(long, default_value_t = 200)]
        search_ms: u64,
        /// Blend weight toward the search distribution (0 = no change,
        /// 1 = replace outright at recorded infosets).
        #[arg(long, default_value_t = 0.5)]
        alpha: f64,
        /// Belief-state value net for ReBeL flop solving during self-play.
        #[arg(long)]
        value_net: Option<String>,
        /// Gadget-safe resolves during self-play (slower).
        #[arg(long)]
        safe_resolve: bool,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Cross-play two blueprints: --focal in one rotating seat against a
    /// full table of --field. If self-play equilibria were interchangeable,
    /// the result would be ~0 — the multiplayer equilibrium-selection probe.
    /// Blueprints with different bet menus each see the other's bets mapped
    /// to their own nearest size, as `serve` maps a human's.
    Crossplay {
        #[arg(long)]
        focal: String,
        #[arg(long)]
        field: String,
        /// Previous blueprint for strategic-abstraction lookups (applied to
        /// whichever side is strategic).
        #[arg(long)]
        strat_prev: Option<String>,
        #[arg(long, default_value_t = 200_000)]
        hands: u64,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Play against Slumbot over its public API (heads-up NLHE, 200bb,
    /// 50/100) and report our winrate. Train the blueprint with
    /// `train --players 2 --stack 20000`.
    Slumbot {
        #[arg(long, default_value = "bp_hu200.bin")]
        blueprint: String,
        #[arg(long, default_value_t = 1_000)]
        hands: u64,
        /// Enable online subgame resolving (recommended).
        #[arg(long)]
        search: bool,
        #[arg(long, default_value_t = 800)]
        search_ms: u64,
        /// Safe (gadget) resolving.
        #[arg(long)]
        safe_resolve: bool,
        /// Belief-state value net for flop solving (must be trained on
        /// 200bb HU spots to help).
        #[arg(long)]
        value_net: Option<String>,
        /// Registered slumbot.com account (optional; anonymous otherwise).
        #[arg(long)]
        username: Option<String>,
        #[arg(long)]
        password: Option<String>,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Log per-hand protocol problems.
        #[arg(long)]
        verbose: bool,
        /// Append each hand's final API response (JSON lines) — data for
        /// opponent modeling.
        #[arg(long)]
        log: Option<String>,
    },
    /// Build a behavioral clone of Slumbot from a `slumbot --log` JSONL
    /// file: replay each hand, count Slumbot's abstract actions per
    /// infoset, and save the normalized strategy in Blueprint format.
    Clone {
        /// Hand log written by `slumbot --log`.
        #[arg(long, default_value = "slumbot_hands.jsonl")]
        log: String,
        /// Blueprint supplying the abstraction (buckets + centroids) the
        /// exploiter will use — must match at train and play time.
        #[arg(long, default_value = "bp_hu200_300m.bin")]
        blueprint: String,
        #[arg(long, default_value = "slumbot_clone.bin")]
        out: String,
        /// Fraction of hands held out for top-1 agreement measurement.
        #[arg(long, default_value_t = 0.1)]
        holdout: f64,
    },
    /// Print blueprint statistics.
    Inspect {
        #[arg(long, default_value = "blueprint.bin")]
        blueprint: String,
    },
    /// Replay the 10,000 real Pluribus hands (PHH dataset) and measure how
    /// often the blueprint agrees with Pluribus's actual decisions.
    Benchmark {
        #[arg(long, default_value = "blueprint.bin")]
        blueprint: String,
        /// Directory of .phh files (searched recursively).
        #[arg(long, default_value = "data/pluribus")]
        dir: String,
    },
}

fn main() {
    match Cli::parse().cmd {
        Cmd::Train {
            iters,
            out,
            bucket_table,
            players,
            stack,
            resume,
            checkpoint,
            buckets,
            menu,
            rollouts,
            runouts,
            kmeans_samples,
            raw_buckets,
            ochs,
            strategic_from,
            train_seed,
            rnr_model,
            rnr_opponent,
            rnr_p,
            no_prune,
            prune_threshold,
            vr_baseline,
            per_action_prune,
            snapshot_avg,
            multiway_focus,
            linear_cap,
            floor_per_weight,
            floor_preflop_only,
            prune_per_weight,
            preflop_explore,
            preflop_sweep_every,
            save_at,
            threads,
        } => {
            if let Some(t) = threads {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build_global()
                    .expect("failed to build thread pool");
            }
            let abs_cfg = AbsConfig {
                postflop_buckets: buckets,
                menu,
                equity_rollouts: rollouts,
                dist_runouts: runouts,
                ..AbsConfig::default()
            };
            let train_cfg = TrainConfig {
                hand: HandConfig {
                    num_players: players,
                    stack,
                    ..HandConfig::default()
                },
                prune_after: if no_prune { u64::MAX } else { 200_000 },
                prune_threshold,
                seed: train_seed,
                pluribus_prune: !per_action_prune,
                snapshot_avg,
                multiway_focus,
                linear_cap,
                floor_per_weight,
                floor_preflop_only,
                prune_per_weight,
                preflop_explore,
                preflop_sweep: preflop_sweep_every > 0,
                ..TrainConfig::default()
            };
            let trainer = match &resume {
                Some(path) => {
                    let t = Trainer::load_checkpoint(path, train_cfg)
                        .unwrap_or_else(|e| die(&format!("cannot load checkpoint {path}: {e}")));
                    println!(
                        "resumed from {path}: {} iterations, {} infosets",
                        t.iterations(),
                        t.node_count()
                    );
                    t
                }
                None if rnr_opponent.is_some() => {
                    let path = rnr_opponent.as_ref().unwrap();
                    let clone = Blueprint::load(path)
                        .unwrap_or_else(|e| die(&format!("cannot load clone {path}: {e}")));
                    println!(
                        "RNR vs clone {path}: {} strategies, p = {rnr_p} \
                         (adopting the clone's abstraction)",
                        clone.strategies.len()
                    );
                    // Adopt the clone's abstraction so bucket/history keys align.
                    let abs =
                        Abstraction::with_centroids(clone.abs_cfg.clone(), clone.centroids.clone());
                    let rnr = Some(cfr::RnrCfg {
                        model: Baseline::Caller, // ignored when rnr_opp is set
                        p: rnr_p,
                    });
                    Trainer::new(Arc::new(abs), train_cfg)
                        .with_rnr(rnr)
                        .with_rnr_opponent(Some(Arc::new(clone)))
                }
                None => {
                    let strat_ctx = strategic_from.as_ref().map(|p| load_strat_ctx(p));
                    let table = bucket_table.as_ref().map(|path| {
                        let t = abstraction::BucketTable::load(path)
                            .unwrap_or_else(|e| die(&format!("cannot load bucket table '{path}': {e}")));
                        if !t.matches(&abs_cfg) {
                            die(&format!(
                                "bucket table '{path}' was built for {} buckets / {} rollouts / {} runouts x {}; \
                                 this run asks for {} / {} / {} x {}",
                                t.cfg.postflop_buckets, t.cfg.equity_rollouts, t.cfg.dist_runouts,
                                t.cfg.runout_rollouts, abs_cfg.postflop_buckets, abs_cfg.equity_rollouts,
                                abs_cfg.dist_runouts, abs_cfg.runout_rollouts
                            ));
                        }
                        if strat_ctx.is_some() || raw_buckets || ochs {
                            die("--bucket-table only supports the default EMD k-means abstraction");
                        }
                        println!("loaded bucket table {path}: {} canonical flop/turn buckets", t.entries.len());
                        t
                    });
                    let centroids = if let Some(t) = &table {
                        t.centroids.clone()
                    } else if let Some(sc) = &strat_ctx {
                        println!(
                            "training STRATEGIC centroids from previous blueprint \
                             ({buckets} buckets, {kmeans_samples} samples/street)..."
                        );
                        let t0 = std::time::Instant::now();
                        let c = Centroids::train_strategic(&abs_cfg, kmeans_samples, 0xC1A5, sc);
                        println!("centroids trained in {:.1}s", t0.elapsed().as_secs_f64());
                        Some(c)
                    } else if raw_buckets {
                        None
                    } else if ochs {
                        println!(
                            "training OCHS k-means centroids ({buckets} buckets, \
                             {kmeans_samples} samples/street)..."
                        );
                        let t0 = std::time::Instant::now();
                        let c = Centroids::train_combined(&abs_cfg, kmeans_samples, 0xC1A5);
                        println!("centroids trained in {:.1}s", t0.elapsed().as_secs_f64());
                        Some(c)
                    } else {
                        println!(
                            "training EMD k-means centroids ({buckets} buckets, \
                             {kmeans_samples} samples/street)..."
                        );
                        let t0 = std::time::Instant::now();
                        let c = Centroids::train(&abs_cfg, kmeans_samples, 0xC1A5);
                        println!("centroids trained in {:.1}s", t0.elapsed().as_secs_f64());
                        Some(c)
                    };
                    let mut abs = Abstraction::with_centroids(abs_cfg, centroids);
                    if let Some(t) = &table {
                        abs.prefill_canon(&t.entries);
                    }
                    if let Some(sc) = strat_ctx {
                        abs = abs.with_strat(sc);
                    }
                    let rnr = rnr_model.map(|model| {
                        println!("RNR training: model {model:?}, p = {rnr_p}");
                        cfr::RnrCfg { model, p: rnr_p }
                    });
                    Trainer::new(Arc::new(abs), train_cfg).with_rnr(rnr)
                }
            }
            .with_vr(vr_baseline);

            println!(
                "training {players}-max, {iters} iterations, menu {menu:?}, pruning {}, \
                 vr baselines {}, snapshot avg {}, multiway focus {multiway_focus}, \
                 floor per weight {floor_per_weight:?}{}, prune per weight {prune_per_weight:?}, \
                 preflop explore {preflop_explore}",
                if no_prune { "off" } else if per_action_prune { "per-action" } else { "pluribus" },
                if vr_baseline { "on" } else { "off" },
                if snapshot_avg { "on" } else { "off" },
                if floor_preflop_only { " (preflop only)" } else { "" }
            );
            let pb = ProgressBar::new(iters);
            pb.set_style(
                ProgressStyle::with_template(
                    "{bar:40.cyan/blue} {pos}/{len} ({per_sec}, ETA {eta})",
                )
                .unwrap(),
            );
            let started = std::time::Instant::now();

            // Train in chunks so long runs checkpoint periodically (and so
            // snapshot mode can sample the current strategy every 5%).
            let chunk = if checkpoint.is_some() || snapshot_avg {
                iters.div_ceil(20).max(100_000).min(iters)
            } else {
                iters
            };
            let mut done_before = 0u64;
            let mut snapshots = 0u32;
            while done_before < iters {
                let next_save = save_at.iter().copied().filter(|&n| n > done_before).min();
                let next_sweep = done_before
                    .checked_div(preflop_sweep_every)
                    .map_or(u64::MAX, |k| (k + 1) * preflop_sweep_every);
                let this = chunk
                    .min(iters - done_before)
                    .min(next_save.map_or(u64::MAX, |n| n - done_before))
                    .min(next_sweep - done_before);
                let base = done_before;
                trainer.run(this, &|done| pb.set_position(base + done));
                done_before += this;
                if preflop_sweep_every > 0 && (done_before.is_multiple_of(preflop_sweep_every) || done_before == iters) {
                    let t = trainer.iterations().min(linear_cap) as f64;
                    trainer.sweep_preflop_average(t);
                }
                if snapshot_avg && done_before * 10 > iters {
                    trainer.snapshot_strategy();
                    snapshots += 1;
                }
                if save_at.contains(&done_before) && done_before < iters {
                    let path = format!("{out}.{done_before}");
                    trainer
                        .to_blueprint()
                        .save(&path)
                        .unwrap_or_else(|e| die(&format!("blueprint save failed: {e}")));
                    println!("saved {path} at {:.0}s", started.elapsed().as_secs_f64());
                }
                if let Some(path) = &checkpoint {
                    trainer
                        .save_checkpoint(path)
                        .unwrap_or_else(|e| die(&format!("checkpoint save failed: {e}")));
                }
            }
            pb.finish();
            if snapshot_avg {
                println!("postflop blueprint = mean of {snapshots} strategy snapshots");
            }

            let secs = started.elapsed().as_secs_f64();
            println!(
                "trained {iters} iterations in {:.1}s ({:.0} iters/s), {} infosets",
                secs,
                iters as f64 / secs,
                trainer.node_count()
            );
            let bp = trainer.to_blueprint();
            bp.save(&out)
                .unwrap_or_else(|e| die(&format!("blueprint save failed: {e}")));
            println!(
                "blueprint saved to {out} ({} strategies)",
                bp.strategies.len()
            );
        }

        Cmd::BucketTable {
            out,
            buckets,
            rollouts,
            runouts,
            kmeans_samples,
            threads,
        } => {
            if let Some(t) = threads {
                rayon::ThreadPoolBuilder::new().num_threads(t).build_global().ok();
            }
            let abs_cfg = AbsConfig {
                postflop_buckets: buckets,
                equity_rollouts: rollouts,
                dist_runouts: runouts,
                cache_cap: 40_000_000, // both caches must hold every canonical pair
                ..AbsConfig::default()
            };
            println!("training EMD k-means centroids ({buckets} buckets, {kmeans_samples} samples/street)...");
            let t0 = std::time::Instant::now();
            let centroids = Centroids::train(&abs_cfg, kmeans_samples, 0xC1A5);
            println!("centroids trained in {:.1}s", t0.elapsed().as_secs_f64());
            let abs = Abstraction::with_centroids(abs_cfg.clone(), Some(centroids.clone()));
            let t0 = std::time::Instant::now();
            let n = abs.build_bucket_table(&|street, done| {
                eprintln!(
                    "  {} buckets: {done} done, {:.0}s",
                    if street == 3 { "flop" } else { "turn" },
                    t0.elapsed().as_secs_f64()
                );
            });
            let table = abstraction::BucketTable {
                cfg: abs_cfg,
                centroids: Some(centroids),
                entries: abs.canon_entries(),
            };
            table.save(&out).unwrap_or_else(|e| die(&format!("cannot write {out}: {e}")));
            println!(
                "bucket table: {n} canonical pairs, {} entries, built in {:.0}s -> {out}",
                table.entries.len(),
                t0.elapsed().as_secs_f64()
            );
        }

        Cmd::Play {
            blueprint,
            players,
            search,
            no_search,
            search_ms,
            qre_lambda,
            value_net,
            safe_resolve,
            adaptive_search,
            seed,
        } => {
            let net = value_net.map(|p| {
                let n = valuenet::ValueNet::load(&p)
                    .unwrap_or_else(|e| die(&format!("cannot load value net '{p}': {e}")));
                println!("loaded value net from {p}");
                Arc::new(n)
            });
            let policy = load_policy(&blueprint).with_value_net(net);
            let opts = play::PlayOpts {
                cfg: HandConfig {
                    num_players: players,
                    ..HandConfig::default()
                },
                search: (search || !no_search).then_some(SearchParams {
                    time_ms: search_ms,
                    qre_lambda,
                    safe_resolve,
                    adaptive: adaptive_search,
                    ..SearchParams::default()
                }),
                seed,
            };
            play::play(&policy, &opts);
        }

        Cmd::Serve { blueprint, stack, also, search, search_ms, search_iters, bucket_table, turn_blueprint } => {
            let table = bucket_table.as_ref().map(|path| {
                abstraction::BucketTable::load(path)
                    .unwrap_or_else(|e| die(&format!("cannot load bucket table '{path}': {e}")))
            });
            let params = SearchParams {
                time_ms: search_ms,
                max_iters: search_iters,
                turn_solve: !turn_blueprint,
                ..SearchParams::default()
            };
            let load = |path: &str, depth: Stack| -> serve::Served {
                let (bot, cfg) = load_bot(path, depth);
                if !search {
                    return serve::Served { bot, cfg, search: None };
                }
                if let Some(t) = &table {
                    if !t.matches(&bot.abs.cfg) {
                        die(&format!("the bucket table does not match '{path}'s abstraction"));
                    }
                    bot.abs.prefill_canon(&t.entries);
                }
                let strategies: Arc<dyn mirror::Strategies> = Arc::from(bot.strategies);
                let policy = Policy::from_strategies(strategies.clone(), cfg.num_players, bot.abs.clone());
                let bot = mirror::Bot { abs: bot.abs, strategies: Box::new(strategies) };
                let train_cfg = TrainConfig {
                    hand: cfg.clone(),
                    prune_after: u64::MAX,
                    ..TrainConfig::default()
                };
                serve::Served { bot, cfg, search: Some(serve::Searcher { policy, params, train_cfg }) }
            };
            let mut served = vec![load(&blueprint, stack)];
            for b in &also {
                served.push(load(&b.path, b.stack));
            }
            if served.iter().any(|s| s.cfg.num_players != served[0].cfg.num_players) {
                die("every served blueprint must have the same number of players");
            }
            serve::serve(&served);
        }

        Cmd::Compact { blueprint, out, steps, bucket_table } => {
            let t0 = std::time::Instant::now();
            let bp = Blueprint::load(&blueprint)
                .unwrap_or_else(|e| die(&format!("cannot load blueprint '{blueprint}': {e}")));
            let table = bucket_table.as_ref().map(|path| {
                abstraction::BucketTable::load(path)
                    .unwrap_or_else(|e| die(&format!("cannot load bucket table '{path}': {e}")))
            });
            let records = compact::write(&bp, &out, steps, table.as_ref())
                .unwrap_or_else(|e| die(&format!("cannot write '{out}': {e}")));
            let bytes = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
            println!(
                "{records} of {} infosets -> {out} ({:.0} MB, {:.1} bytes/infoset) in {:.0}s",
                bp.strategies.len(),
                bytes as f64 / 1e6,
                bytes as f64 / records.max(1) as f64,
                t0.elapsed().as_secs_f64()
            );
        }

        Cmd::RefinePreflop {
            blueprint,
            bucket_table,
            iters,
            out,
            stack,
            floor_per_weight,
            no_prune,
            preflop_explore,
            sweep_every,
            train_seed,
            save_at,
            report_line,
            report_hands,
            threads,
        } => {
            if let Some(t) = threads {
                rayon::ThreadPoolBuilder::new().num_threads(t).build_global().expect("thread pool");
            }
            let (bot, cfg) = load_bot(&blueprint, stack);
            if let Some(path) = &bucket_table {
                let t = abstraction::BucketTable::load(path)
                    .unwrap_or_else(|e| die(&format!("cannot load bucket table '{path}': {e}")));
                if !t.matches(&bot.abs.cfg) {
                    die(&format!("bucket table '{path}' does not match the blueprint's abstraction"));
                }
                bot.abs.prefill_canon(&t.entries);
            }
            let train_cfg = TrainConfig {
                hand: cfg.clone(),
                prune_after: if no_prune { u64::MAX } else { 200_000 },
                seed: train_seed,
                floor_per_weight: Some(floor_per_weight),
                preflop_explore,
                preflop_sweep: sweep_every > 0,
                ..TrainConfig::default()
            };
            let frozen: Arc<dyn mirror::Strategies> = Arc::from(bot.strategies);
            let trainer = Trainer::new(bot.abs.clone(), train_cfg).with_frozen_postflop(Some(frozen));
            println!(
                "refining preflop over {blueprint}: {iters} iterations, floor {floor_per_weight:e} per weight, \
                 pruning {}, explore {preflop_explore}",
                if no_prune { "off" } else { "pluribus" }
            );
            let started = std::time::Instant::now();
            let mut done = 0u64;
            let mut marks: Vec<u64> = save_at.iter().copied().filter(|&n| n < iters).collect();
            marks.sort_unstable();
            marks.push(iters);
            let mut sweep_secs = 0.0;
            for m in marks {
                if sweep_every > 0 {
                    while done < m {
                        let step = sweep_every.min(m - done);
                        trainer.run(step, &|_| {});
                        done += step;
                        let t0 = std::time::Instant::now();
                        trainer.sweep_preflop_average(done as f64);
                        sweep_secs += t0.elapsed().as_secs_f64();
                    }
                } else {
                    trainer.run(m - done, &|_| {});
                }
                done = m;
                let path = if m == iters { out.clone() } else { format!("{out}.{m}") };
                let bp = trainer.to_blueprint();
                bp.save(&path).unwrap_or_else(|e| die(&format!("save failed: {e}")));
                if let Some(line) = &report_line {
                    let n = preflop::node(&cfg, &trainer.abs, line, cards::fresh_deck()).unwrap_or_else(|e| die(&e));
                    for class in report_hands.split(',') {
                        let combo = preflop::class_combos(class).unwrap_or_else(|e| die(&e))[0];
                        let b = abstraction::preflop_bucket(*combo);
                        let f = |v: Option<Vec<f64>>| {
                            v.map_or("unvisited".to_string(), |v| {
                                v.iter().map(|x| format!("{:>4.0}", 100.0 * x)).collect::<Vec<_>>().join("")
                            })
                        };
                        println!(
                            "  {class:<4} current {}  average {}",
                            f(trainer.current_strategy(b, &n.hist)),
                            f(trainer.avg_strategy(b, &n.hist))
                        );
                    }
                }
                println!(
                    "{m} iterations at {:.0}s ({:.0} iters/s, {sweep_secs:.0}s sweeping): {} preflop strategies -> {path}",
                    started.elapsed().as_secs_f64(),
                    m as f64 / started.elapsed().as_secs_f64(),
                    bp.strategies.len()
                );
            }
        }
        Cmd::MergePreflop { base, preflop, out } => {
            let mut bp = Blueprint::load(&base).unwrap_or_else(|e| die(&format!("cannot load '{base}': {e}")));
            let pre = Blueprint::load(&preflop).unwrap_or_else(|e| die(&format!("cannot load '{preflop}': {e}")));
            let before = bp.strategies.len();
            // Preflop infosets the refinement never reached keep the base
            // strategy rather than falling back to check/call.
            bp.strategies.retain(|k, _| !pre.strategies.contains_key(k));
            let replaced = before - bp.strategies.len();
            let added = pre.strategies.len();
            bp.strategies.extend(pre.strategies);
            bp.save(&out).unwrap_or_else(|e| die(&format!("save failed: {e}")));
            println!(
                "{out}: {replaced} preflop strategies replaced, {} new, {} in all",
                added - replaced,
                bp.strategies.len()
            );
        }
        Cmd::PreflopProbe { blueprint, stack, line, all } => {
            let (bot, cfg) = load_bot(&blueprint, stack);
            let p = preflop::probe(&bot, &cfg, &line).unwrap_or_else(|e| die(&e));
            println!(
                "line '{line}': all-in {:.1}% of combos ({:.0}% of them residue under 5%), premiums {:.1}%",
                100.0 * p.all_in,
                100.0 * p.residue,
                100.0 * p.premium_all_in
            );
            print!("{:<5}", "hand");
            for a in &p.actions {
                print!(" {a:>7}");
            }
            println!();
            for (name, row) in &p.rows {
                if all || preflop::PREMIUMS.contains(&name.as_str()) {
                    print!("{name:<5}");
                    for x in row {
                        print!(" {:>6.1}%", 100.0 * x);
                    }
                    println!();
                }
            }
        }
        Cmd::PreflopReach { blueprint, stack, iters, csv, compare } => {
            let (bot, cfg) = load_bot(&blueprint, stack);
            let other = compare.map(|path| load_bot(&path, stack).0);
            let rows = preflop::reach(&bot, other.as_ref(), &cfg);
            let per_combo = |r: &preflop::ReachRow| iters as f64 / cfg.num_players.get() as f64 * r.opp_reach / 1326.0;
            if let Some(path) = &csv {
                let mut out = String::from("tokens,line,seat,opp_reach,own_reach,updates_per_combo,resolved,unseen_combos,diff\n");
                for r in &rows {
                    out += &format!(
                        "{},\"{}\",{},{:e},{:e},{:.3},{},{},{}\n",
                        r.tokens,
                        r.line,
                        preflop::SEATS[r.seat],
                        r.opp_reach,
                        r.own_reach,
                        per_combo(r),
                        r.resolved,
                        r.unseen,
                        r.diff.map_or(String::new(), |d| format!("{d:.4}"))
                    );
                }
                std::fs::write(path, out).unwrap_or_else(|e| die(&format!("cannot write '{path}': {e}")));
            }
            let total: f64 = rows.iter().map(|r| r.opp_reach * r.own_reach).sum();
            let tiers = [1e5, 1e4, 1e3, 1e2, 1e1, 1.0, 0.0];
            println!("{} preflop decisions, {} answered by the all-in resolve under serve --search", rows.len(), rows.iter().filter(|r| r.resolved).count());
            println!("est. regret updates per combo | blueprint decisions | resolved decisions | share of self-play preflop decisions (blueprint / resolved){}", if other.is_some() { " | mean strategy distance to --compare" } else { "" });
            let mut hi = f64::INFINITY;
            for lo in tiers {
                let tier: Vec<&preflop::ReachRow> = rows.iter().filter(|r| (lo..hi).contains(&per_combo(r))).collect();
                let share = |resolved: bool| {
                    100.0 * tier.iter().filter(|r| r.resolved == resolved).map(|r| r.opp_reach * r.own_reach).sum::<f64>() / total + 0.0
                };
                let diffs: Vec<f64> = tier.iter().filter_map(|r| r.diff).collect();
                let diff = if other.is_none() {
                    String::new()
                } else if diffs.is_empty() {
                    " | -".to_string()
                } else {
                    format!(" | {:.3}", diffs.iter().sum::<f64>() / diffs.len() as f64)
                };
                let label = if lo == 0.0 { format!("< {hi}") } else if hi.is_infinite() { format!(">= {lo}") } else { format!("{lo} - {hi}") };
                println!(
                    "{label:>29} | {:>19} | {:>18} | {:.4}% / {:.4}%{diff}",
                    tier.iter().filter(|r| !r.resolved).count(),
                    tier.iter().filter(|r| r.resolved).count(),
                    share(false),
                    share(true)
                );
                hi = lo;
            }
        }
        Cmd::ActionEv { blueprint, stack, line, hands, deals, bucket_table, seed } => {
            let (bot, cfg) = load_bot(&blueprint, stack);
            if let Some(path) = &bucket_table {
                let t = abstraction::BucketTable::load(path)
                    .unwrap_or_else(|e| die(&format!("cannot load bucket table '{path}': {e}")));
                if !t.matches(&bot.abs.cfg) {
                    die(&format!("bucket table '{path}' does not match the blueprint's abstraction"));
                }
                bot.abs.prefill_canon(&t.entries);
            }
            if hands == "all" {
                // Node summary: the best response's gain over the mix,
                // weighted by combos and the hero's own reach.
                let mut rows = Vec::new();
                for class in preflop::all_classes() {
                    let r = preflop::action_ev(&bot, &cfg, &line, &class, deals, seed)
                        .unwrap_or_else(|e| die(&e));
                    let w = preflop::class_combos(&class).unwrap().len() as f64 * r.hero_reach;
                    rows.push((w, r));
                }
                let tw: f64 = rows.iter().map(|r| r.0).sum();
                let gain: f64 = rows.iter().map(|r| r.0 * r.1.gain()).sum::<f64>() / tw;
                println!(
                    "'{line}': best-response gain over the mix {gain:.3} bb per reaching hand \
                     ({deals} deals per class, {tw:.0} reach-weighted combos)"
                );
                rows.sort_by(|a, b| (b.0 * b.1.gain()).total_cmp(&(a.0 * a.1.gain())));
                for (w, r) in rows.iter().take(12) {
                    let best = (0..r.actions.len()).max_by(|&i, &j| r.vs_mix[i].0.total_cmp(&r.vs_mix[j].0)).unwrap();
                    let mix: Vec<String> = r.actions.iter().zip(&r.mix).map(|(a, p)| format!("{a} {:.0}%", 100.0 * p)).collect();
                    println!(
                        "  {:<5} weight {:>5.1}  gain {:+.2} bb by {}  (mix: {})",
                        r.class, w, r.gain(), r.actions[best], mix.join(", ")
                    );
                }
                return;
            }
            for class in hands.split(',') {
                let t0 = std::time::Instant::now();
                let r = preflop::action_ev(&bot, &cfg, &line, class.trim(), deals, seed)
                    .unwrap_or_else(|e| die(&e));
                println!(
                    "{} at '{line}': {} deals, ESS {:.0}, {:.0}s",
                    r.class,
                    r.deals,
                    r.ess,
                    t0.elapsed().as_secs_f64()
                );
                println!("  {:<8} {:>6} {:>16} {:>16}", "action", "mix", "EV bb", "EV - mix bb");
                for j in 0..r.actions.len() {
                    println!(
                        "  {:<8} {:>5.1}% {:>+9.2} ±{:<5.2} {:>+9.2} ±{:<5.2}",
                        r.actions[j],
                        100.0 * r.mix[j],
                        r.ev[j].0,
                        r.ev[j].1,
                        r.vs_mix[j].0,
                        r.vs_mix[j].1
                    );
                }
            }
        }
        Cmd::BucketCheck { table, hands, seed } => {
            let t = abstraction::BucketTable::load(&table)
                .unwrap_or_else(|e| die(&format!("cannot load bucket table '{table}': {e}")));
            let t0 = std::time::Instant::now();
            let [flop, turn] = bucket_check::run(&t, hands, seed);
            let pct = |(m, n): (u64, u64)| 100.0 * m as f64 / n.max(1) as f64;
            println!(
                "{hands} hands in {:.0}s: lazy buckets differ from the table on {:.2}% of flops, {:.2}% of turns",
                t0.elapsed().as_secs_f64(),
                pct(flop),
                pct(turn)
            );
        }

        Cmd::OffmenuMatch {
            blueprint,
            stack,
            resize,
            hands,
            seed,
        } => {
            let (bot, cfg) = load_bot(&blueprint, stack);
            let t0 = std::time::Instant::now();
            let r = offmenu_match::run(&bot, &cfg, resize, hands, seed);
            println!(
                "{} paired hands, resize {resize:?}, in {:.0}s ({:.0}% of hero bets resized)\n  \
                 resized hero: {:+.0} ±{:.0} mbb/hand\n  \
                 on-menu hero: {:+.0} ±{:.0} mbb/hand\n  \
                 gain (paired): {:+.0} ±{:.0} mbb/hand",
                r.hands,
                t0.elapsed().as_secs_f64(),
                100.0 * r.resized_share,
                r.resized.0,
                r.resized.1,
                r.on_menu.0,
                r.on_menu.1,
                r.gain.0,
                r.gain.1,
            );
        }

        Cmd::DepthMatch {
            a,
            a_stack,
            b,
            b_stack,
            stack,
            hands,
            seed,
        } => {
            let (pa, cfg_a) = load_bot(&a, a_stack);
            let (pb, cfg_b) = load_bot(&b, b_stack);
            let t0 = std::time::Instant::now();
            let r = depth_match::run(
                &depth_match::Side { bot: &pa, cfg: cfg_a },
                &depth_match::Side { bot: &pb, cfg: cfg_b },
                stack,
                hands,
                seed,
            );
            let bb = stack.get() / 100;
            println!(
                "{} paired hands at {bb}bb in {:.0}s\n  \
                 A ({a_stack} chips) as hero: {:+.0} ±{:.0} mbb/hand ({:.1}% unseen, {:.1}% capped)\n  \
                 B ({b_stack} chips) as hero: {:+.0} ±{:.0} mbb/hand\n  \
                 A minus B (paired): {:+.0} ±{:.0} mbb/hand",
                r.hands,
                t0.elapsed().as_secs_f64(),
                r.a.0,
                r.a.1,
                100.0 * r.a_unseen,
                100.0 * r.a_capped,
                r.b.0,
                r.b.1,
                r.cost.0,
                r.cost.1,
            );
        }

        Cmd::Eval {
            blueprint,
            hands,
            baseline,
            players,
            duplicate,
            aivat,
            search,
            search_gain,
            net_gain,
            search_ms,
            search_iters,
            value_net,
            strat_prev,
            seed,
        } => {
            let net = value_net.map(|p| {
                let n = valuenet::ValueNet::load(&p)
                    .unwrap_or_else(|e| die(&format!("cannot load value net '{p}': {e}")));
                println!("loaded value net from {p}");
                Arc::new(n)
            });
            let policy =
                load_policy_strat(&blueprint, strat_prev.as_deref()).with_value_net(net);
            let cfg = HandConfig {
                num_players: players,
                ..HandConfig::default()
            };
            println!(
                "evaluating {} hands vs {:?} baselines ({}-max{})...",
                hands,
                baseline,
                players,
                if net_gain {
                    ", paired net-gain"
                } else if search_gain {
                    ", paired search-gain"
                } else if search {
                    ", with search"
                } else if aivat {
                    ", AIVAT"
                } else if duplicate {
                    ", duplicate deals"
                } else {
                    ""
                }
            );
            let started = std::time::Instant::now();
            let params = SearchParams {
                time_ms: search_ms,
                max_iters: search_iters,
                adaptive: true,
                ..SearchParams::default()
            };
            let r = if net_gain {
                if policy.value_net.is_none() {
                    die("--net-gain needs --value-net");
                }
                let plain = policy.clone_without_net();
                table::run_eval_paired_policies(
                    &policy,
                    Some(params),
                    &plain,
                    Some(params),
                    &cfg,
                    hands,
                    seed,
                )
            } else if search_gain {
                table::run_eval_paired(&policy, &cfg, hands, params, seed)
            } else if search {
                table::run_eval_search(&policy, &cfg, baseline, hands, params, seed)
            } else if aivat {
                aivat::run_eval_aivat(&policy, &cfg, baseline, hands, seed)
            } else if duplicate {
                table::run_eval_duplicate(&policy, &cfg, baseline, hands / players.get() as u64, seed)
            } else {
                table::run_eval(&policy, &cfg, baseline, hands, seed)
            };
            println!(
                "winrate: {:+.1} mbb/hand (95% CI ±{:.1}) over {} hands in {:.1}s",
                r.mbb_per_hand,
                r.ci95,
                r.hands,
                started.elapsed().as_secs_f64()
            );
        }

        Cmd::Lbr {
            blueprint,
            hands,
            multiway,
            runouts,
            strat_prev,
            seed,
            search,
            search_ms,
            search_iters,
            decision_root,
            turn_blueprint,
            search_buckets,
            stack,
        } => {
            let params = search.then_some(SearchParams {
                time_ms: search_ms,
                max_iters: search_iters,
                round_root: !decision_root,
                turn_solve: !turn_blueprint,
                ..SearchParams::default()
            });
            let policy = load_policy_strat(&blueprint, strat_prev.as_deref())
                .with_search_buckets(search_buckets);
            let cfg = HandConfig {
                num_players: policy.blueprint.num_players,
                stack,
                ..HandConfig::default()
            };
            println!(
                "LBR probe: {hands} hands {} ({}-max game, {runouts} runouts{})...",
                if multiway { "multiway, LBR seat rotating" } else { "blind-vs-blind" },
                cfg.num_players,
                if search {
                    format!(", bot searching {search_ms}ms/{search_iters} iters per decision, {} root, {} buckets, two-player turns {}", if decision_root { "decision" } else { "round" }, search_buckets.map_or("blueprint".to_string(), |b| b.to_string()), if turn_blueprint { "from the blueprint" } else { "solved" })
                } else {
                    String::new()
                }
            );
            if search && !multiway {
                die("--search is only wired into the multiway LBR probe (use --multiway)");
            }
            let started = std::time::Instant::now();
            let r = if multiway {
                lbr::run_lbr_multiway(&policy, &cfg, hands, runouts, seed, params)
            } else {
                lbr::run_lbr(&policy, &cfg, hands, runouts, seed)
            };
            println!(
                "LBR wins {:+.1} mbb/hand (95% CI ±{:.1}) over {} hands in {:.1}s",
                r.mbb_per_hand,
                r.ci95,
                r.hands,
                started.elapsed().as_secs_f64()
            );
            println!("(lower bound on the blueprint's exploitability; 0 = unexploited)");
            if let Some(r) = bot::fallback_report() {
                println!("{r}");
            }
            if let Some(r) = bot::search_timing_report() {
                println!("{r}");
            }
        }

        Cmd::Br {
            blueprint,
            hands,
            runouts,
            strat_prev,
            seed,
            search,
            search_ms,
            search_iters,
            decision_root,
            turn_blueprint,
            search_buckets,
            stack,
        } => {
            let params = search.then_some(SearchParams {
                time_ms: search_ms,
                max_iters: search_iters,
                round_root: !decision_root,
                turn_solve: !turn_blueprint,
                ..SearchParams::default()
            });
            let policy = load_policy_strat(&blueprint, strat_prev.as_deref())
                .with_search_buckets(search_buckets);
            let cfg = HandConfig {
                num_players: policy.blueprint.num_players,
                stack,
                ..HandConfig::default()
            };
            println!(
                "BR probe: {hands} hands blind-vs-blind ({}-max game), exact \
                 turn/river subgame best response{}...",
                cfg.num_players,
                if search {
                    format!(", bot searching {search_ms}ms/{search_iters} iters per decision, {} root, {} buckets, two-player turns {}", if decision_root { "decision" } else { "round" }, search_buckets.map_or("blueprint".to_string(), |b| b.to_string()), if turn_blueprint { "from the blueprint" } else { "solved" })
                } else {
                    String::new()
                }
            );
            let started = std::time::Instant::now();
            let r = br::run_br(&policy, &cfg, hands, runouts, seed, params);
            println!(
                "BR probe wins {:+.1} mbb/hand (95% CI ±{:.1}) over {} hands in {:.1}s",
                r.mbb_per_hand,
                r.ci95,
                r.hands,
                started.elapsed().as_secs_f64()
            );
            println!("(tighter lower bound on exploitability than lbr; 0 = unexploited)");
            if let Some(r) = bot::search_timing_report() {
                println!("{r}");
            }
        }

        Cmd::AblateSafety { spots, iters, seed } => {
            println!(
                "safety ablation: {spots} random river spots, {iters} CFR iters/solve..."
            );
            let started = std::time::Instant::now();
            let rows = ablate::run(spots, iters, seed);
            println!(
                "\n{:<8} {:>13} {:>12} {:>11} {:>10} {:>9}",
                "epsilon", "unsafe mean", "unsafe max", "safe mean", "safe max", "pot"
            );
            for r in rows {
                println!(
                    "{:<8} {:>13.1} {:>12.1} {:>11.1} {:>10.1} {:>9.0}",
                    r.epsilon, r.unsafe_mean, r.unsafe_max, r.safe_mean, r.safe_max, r.pot
                );
            }
            println!(
                "\n(margins in chips: opponent best-response value beyond its safety \
                 value,\n true-range weighted; {:.1}s)",
                started.elapsed().as_secs_f64()
            );
        }

        Cmd::GenTurnData {
            blueprint,
            out,
            samples,
            solve_iters,
            solve_ms,
            seed,
        } => {
            let policy = load_policy(&blueprint);
            println!(
                "generating {samples} exactly-solved turn spots \
                 ({solve_iters} CFR iters or {solve_ms}ms per solve)..."
            );
            let started = std::time::Instant::now();
            let data = valuenet::generate(&policy, solve_iters, solve_ms, samples, seed, &|done| {
                println!(
                    "  {done}/{samples} samples ({:.1}s elapsed)",
                    started.elapsed().as_secs_f64()
                );
            });
            valuenet::save_samples(&out, &data)
                .unwrap_or_else(|e| die(&format!("cannot write {out}: {e}")));
            println!(
                "wrote {} samples to {out} in {:.1}s",
                data.len(),
                started.elapsed().as_secs_f64()
            );
        }

        Cmd::TrainValueNet {
            data,
            out,
            hidden,
            epochs,
            lr,
            batch,
            seed,
        } => {
            // Comma-separated data files are concatenated (e.g. combining
            // generation batches).
            let mut samples = Vec::new();
            for path in data.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                let mut s = valuenet::load_samples(path)
                    .unwrap_or_else(|e| die(&format!("cannot load {path}: {e}")));
                println!("loaded {} samples from {path}", s.len());
                samples.append(&mut s);
            }
            let hidden: Vec<usize> = hidden
                .split(',')
                .map(|s| s.trim().parse().unwrap_or_else(|_| die("bad --hidden")))
                .collect();
            println!(
                "training value net on {} samples (hidden {hidden:?}, {epochs} epochs)...",
                samples.len()
            );
            let started = std::time::Instant::now();
            let (net, val_loss) = valuenet::train(
                &samples,
                &hidden,
                epochs,
                lr,
                batch,
                seed,
                &mut |e, tr, va| {
                    println!(
                        "  epoch {:>3}: train {:.5}  val {:.5}  ({:.0}s)",
                        e + 1,
                        tr,
                        va,
                        started.elapsed().as_secs_f64()
                    );
                },
            );
            net.save(&out)
                .unwrap_or_else(|e| die(&format!("cannot write {out}: {e}")));
            println!(
                "value net saved to {out} (final val loss {val_loss:.5}, {:.0}s)",
                started.elapsed().as_secs_f64()
            );
        }

        Cmd::Distill {
            blueprint,
            out,
            hands,
            search_ms,
            alpha,
            value_net,
            safe_resolve,
            seed,
        } => {
            let net = value_net.map(|p| {
                let n = valuenet::ValueNet::load(&p)
                    .unwrap_or_else(|e| die(&format!("cannot load value net '{p}': {e}")));
                println!("loaded value net from {p}");
                Arc::new(n)
            });
            let policy = load_policy(&blueprint).with_value_net(net);
            let cfg = HandConfig {
                num_players: policy.blueprint.num_players,
                ..HandConfig::default()
            };
            let dcfg = distill::DistillCfg {
                hands,
                params: SearchParams {
                    time_ms: search_ms,
                    safe_resolve,
                    ..SearchParams::default()
                },
                seed,
            };
            println!(
                "distilling: {hands} self-play hands ({}-max, {search_ms}ms/decision, alpha {alpha})...",
                cfg.num_players
            );
            let started = std::time::Instant::now();
            let (records, samples) = distill::collect(&policy, &cfg, &dcfg);
            println!(
                "collected {samples} searched decisions at {} infosets in {:.1}s",
                records.len(),
                started.elapsed().as_secs_f64()
            );
            // Reclaim sole ownership of the blueprint so the merge mutates
            // in place instead of cloning a multi-GB map.
            let bot::Policy { blueprint: bp_arc, .. } = policy;
            let owned = Arc::try_unwrap(bp_arc)
                .unwrap_or_else(|arc| (*arc).clone());
            let (bp2, updated, skipped) = distill::merge(owned, &records, alpha);
            bp2.save(&out)
                .unwrap_or_else(|e| die(&format!("save failed: {e}")));
            println!(
                "updated {updated} infosets ({skipped} skipped on menu mismatch); \
                 distilled blueprint saved to {out} ({} strategies)",
                bp2.strategies.len()
            );
            println!("gate it: br/lbr, crossplay --focal {out} --field {blueprint}, eval");
        }

        Cmd::Crossplay {
            focal,
            field,
            strat_prev,
            hands,
            seed,
        } => {
            let focal_p = load_policy_strat(&focal, strat_prev.as_deref());
            let field_p = load_policy_strat(&field, strat_prev.as_deref());
            if focal_p.blueprint.num_players != field_p.blueprint.num_players {
                die("blueprints trained for different table sizes");
            }
            let (focal_menu, field_menu) = (focal_p.blueprint.abs_cfg.menu, field_p.blueprint.abs_cfg.menu);
            if focal_menu != field_menu {
                println!(
                    "menus differ ({focal_menu:?} focal, {field_menu:?} field): each plays from its own \
                     view, the other's bets mapped to its nearest size"
                );
            }
            let cfg = HandConfig {
                num_players: focal_p.blueprint.num_players,
                ..HandConfig::default()
            };
            println!("cross-play: {focal} (1 seat) vs {field} (rest), {hands} hands...");
            let r = table::run_crossplay(&focal_p, &field_p, &cfg, hands, seed);
            println!(
                "focal winrate: {:+.1} mbb/hand (95% CI ±{:.1}) over {} hands",
                r.mbb_per_hand, r.ci95, r.hands
            );
            if let Some(r) = bot::fallback_report() {
                println!("{r}");
            }
        }

        Cmd::Slumbot {
            blueprint,
            hands,
            search,
            search_ms,
            safe_resolve,
            value_net,
            username,
            password,
            seed,
            verbose,
            log,
        } => {
            let net = value_net.map(|p| {
                let n = valuenet::ValueNet::load(&p)
                    .unwrap_or_else(|e| die(&format!("cannot load value net '{p}': {e}")));
                println!("loaded value net from {p}");
                Arc::new(n)
            });
            let policy = load_policy(&blueprint).with_value_net(net);
            if policy.blueprint.num_players != Players::TWO {
                die("Slumbot is heads-up: train with --players 2 --stack 20000");
            }
            let mut transport = slumbot::HttpTransport;
            let token = match (username, password) {
                (Some(u), Some(p)) => Some(
                    transport
                        .login(&u, &p)
                        .unwrap_or_else(|e| die(&format!("login failed: {e}"))),
                ),
                _ => None,
            };
            let cfg = slumbot::SlumbotCfg {
                hands,
                search: search.then_some(SearchParams {
                    time_ms: search_ms,
                    safe_resolve,
                    ..SearchParams::default()
                }),
                seed,
                token,
                verbose,
                log,
            };
            println!("playing {hands} hands vs Slumbot (search: {search})...");
            let started = std::time::Instant::now();
            let r = slumbot::run(&policy, &mut transport, &cfg, &mut |h, mean| {
                println!("  {h} hands, running mean {mean:+.0} mbb/hand");
            })
            .unwrap_or_else(|e| die(&format!("slumbot session failed: {e}")));
            println!(
                "vs Slumbot: {:+.1} mbb/hand (95% CI ±{:.1}) over {} hands \
                 ({} desyncs) in {:.0}s",
                r.mbb_per_hand,
                r.ci95,
                r.hands,
                r.desyncs,
                started.elapsed().as_secs_f64()
            );
            print!("{}", r.autopsy.report());
        }

        Cmd::Clone {
            log,
            blueprint,
            out,
            holdout,
        } => {
            let policy = load_policy(&blueprint);
            let (bp, stats) = clone::build(&log, &policy.abs, holdout)
                .unwrap_or_else(|e| die(&format!("clone build failed: {e}")));
            println!(
                "clone: {} hands ({} skipped), {} decisions, {} infosets",
                stats.hands, stats.skipped, stats.decisions, stats.infosets
            );
            let streets = ["preflop", "flop", "turn", "river"];
            for (s, (hit, n)) in streets.iter().zip(stats.agree) {
                if n > 0 {
                    println!(
                        "  held-out top-1 agreement {s}: {:.1}% ({n} decisions)",
                        100.0 * hit as f64 / n as f64
                    );
                }
            }
            bp.save(&out)
                .unwrap_or_else(|e| die(&format!("cannot save {out}: {e}")));
            println!("saved clone to {out}");
        }

        Cmd::Inspect { blueprint } => {
            let bp = Blueprint::load(&blueprint)
                .unwrap_or_else(|e| die(&format!("cannot load {blueprint}: {e}")));
            println!("blueprint: {blueprint}");
            println!("  trained iterations: {}", bp.iterations);
            println!("  players: {}", bp.num_players);
            println!("  infosets: {}", bp.strategies.len());
            // Street distribution: count separator tokens after the 2 bucket bytes.
            let mut by_street = [0u64; 4];
            for k in bp.strategies.keys() {
                let seps = k[2..]
                    .iter()
                    .filter(|&&t| t == abstraction::Token::StreetSep as u8)
                    .count()
                    .min(3);
                by_street[seps] += 1;
            }
            for (i, name) in ["preflop", "flop", "turn", "river"].iter().enumerate() {
                println!("  {name}: {}", by_street[i]);
            }
        }

        Cmd::Benchmark { blueprint, dir } => {
            let policy = load_policy(&blueprint);
            println!("replaying Pluribus hands from {dir}...");
            let started = std::time::Instant::now();
            let r = benchmark::run(&dir, &policy)
                .unwrap_or_else(|e| die(&format!("benchmark failed: {e}")));
            println!(
                "replayed {} hands in {:.1}s ({} skipped, {} desynced)",
                r.hands,
                started.elapsed().as_secs_f64(),
                r.skipped,
                r.desynced
            );
            if r.chip_checked > 0 {
                println!(
                    "chip accounting: {}/{} hands match the logged finishing stacks",
                    r.chip_checked - r.chip_mismatch,
                    r.chip_checked
                );
            }
            println!(
                "\n{:<8} {:>10} {:>10} {:>12} {:>12}",
                "street", "decisions", "covered", "top-1 agree", "mean prob"
            );
            let mut tot = benchmark::StreetStats::default();
            for (i, name) in ["preflop", "flop", "turn", "river"].iter().enumerate() {
                let s = &r.by_street[i];
                print_bench_row(name, s);
                tot.decisions += s.decisions;
                tot.covered += s.covered;
                tot.top1 += s.top1;
                tot.prob_sum += s.prob_sum;
            }
            print_bench_row("TOTAL", &tot);
        }
    }
}

fn print_bench_row(name: &str, s: &benchmark::StreetStats) {
    let pct = |num: u64, den: u64| {
        if den == 0 {
            "-".to_string()
        } else {
            format!("{:.1}%", 100.0 * num as f64 / den as f64)
        }
    };
    let mean_prob = if s.covered == 0 {
        "-".to_string()
    } else {
        format!("{:.3}", s.prob_sum / s.covered as f64)
    };
    println!(
        "{:<8} {:>10} {:>10} {:>12} {:>12}",
        name,
        s.decisions,
        pct(s.covered, s.decisions),
        pct(s.top1, s.covered),
        mean_prob
    );
}

/// Previous-round context for strategic abstractions. The referenced
/// blueprint must itself use an equity abstraction (one co-training level).
fn load_strat_ctx(path: &str) -> abstraction::StratCtx {
    let bp = Blueprint::load(path)
        .unwrap_or_else(|e| die(&format!("cannot load previous blueprint '{path}': {e}")));
    if bp.centroids.as_ref().is_some_and(|c| c.is_strategic()) {
        die("chained strategic blueprints are not supported: the previous blueprint must use an equity abstraction");
    }
    println!(
        "previous blueprint for strategic fingerprints: {} infosets ({} buckets)",
        bp.strategies.len(),
        bp.abs_cfg.postflop_buckets
    );
    let abs = Abstraction::with_centroids(bp.abs_cfg.clone(), bp.centroids.clone());
    abstraction::StratCtx {
        bp: Arc::new(bp),
        abs: Arc::new(abs),
        rollouts: 16,
    }
}

fn load_policy(path: &str) -> Policy {
    load_policy_strat(path, None)
}

fn load_policy_strat(path: &str, strat_prev: Option<&str>) -> Policy {
    // Compact blueprints and preflop overlays serve lookups only.
    if compact::is_compact(path) || path.contains('+') {
        let (bot, cfg) = load_bot(path, HandConfig::default().stack);
        return Policy::from_strategies(Arc::from(bot.strategies), cfg.num_players, bot.abs);
    }
    let bp = Blueprint::load(path).unwrap_or_else(|e| {
        die(&format!(
            "cannot load blueprint '{path}': {e}\nrun `pluribus train --out {path}` first"
        ))
    });
    let strategic = bp.centroids.as_ref().is_some_and(|c| c.is_strategic());
    println!(
        "loaded blueprint: {} infosets from {} iterations ({} card buckets, {:?} menu, {})",
        bp.strategies.len(),
        bp.iterations,
        bp.abs_cfg.postflop_buckets,
        bp.abs_cfg.menu,
        if strategic {
            "STRATEGIC clustering"
        } else if bp.centroids.is_some() {
            "EMD k-means"
        } else {
            "raw equity"
        }
    );
    let mut abs = Abstraction::with_centroids(bp.abs_cfg.clone(), bp.centroids.clone());
    if strategic {
        let prev = strat_prev.unwrap_or_else(|| {
            die("this blueprint uses strategic clustering: pass --strat-prev <previous blueprint>")
        });
        abs = abs.with_strat(load_strat_ctx(prev));
    }
    Policy::new(bp, Arc::new(abs))
}

/// Load a bot from a training or compact blueprint, with diagnostics on
/// stderr only (stdout may carry a protocol). `stack` is the depth the
/// blueprint was trained at.
fn load_bot(path: &str, stack: Stack) -> (mirror::Bot, HandConfig) {
    let fail = |e: std::io::Error| -> ! { die(&format!("cannot load blueprint '{path}': {e}")) };
    if let Some((base, pre)) = path.split_once('+') {
        let (bot, cfg) = load_bot(base, stack);
        let fail = |e: std::io::Error| -> ! { die(&format!("cannot load preflop '{pre}': {e}")) };
        let preflop: Box<dyn mirror::Strategies> = if compact::is_compact(pre) {
            Box::new(compact::CompactBlueprint::open(pre).unwrap_or_else(|e| fail(e)))
        } else {
            Box::new(Blueprint::load(pre).unwrap_or_else(|e| fail(e)))
        };
        eprintln!("{pre}: {} preflop strategies over {base}", preflop.len());
        let strategies = Box::new(mirror::PreflopOverlay { preflop, base: bot.strategies });
        return (mirror::Bot { abs: bot.abs, strategies }, cfg);
    }
    let (strategies, players, abs_cfg, centroids, table): (Box<dyn mirror::Strategies>, _, _, _, _) =
        if compact::is_compact(path) {
            let c = compact::CompactBlueprint::open(path).unwrap_or_else(|e| fail(e));
            let (n, cfg, cents) = (c.header.num_players, c.header.abs_cfg.clone(), c.header.centroids.clone());
            let table = c.bucket_table();
            (Box::new(c), n, cfg, cents, table)
        } else {
            let bp = Blueprint::load(path).unwrap_or_else(|e| fail(e));
            let (n, cfg, cents) = (bp.num_players, bp.abs_cfg.clone(), bp.centroids.clone());
            (Box::new(bp), n, cfg, cents, None)
        };
    if centroids.as_ref().is_some_and(|c| c.is_strategic()) {
        die("strategic-clustering blueprints are not supported here");
    }
    eprintln!(
        "{path}: {} infosets, {players} players, {:?} menu{}",
        strategies.len(),
        abs_cfg.menu,
        if table.is_some() { ", bucket table included" } else { "" }
    );
    let cfg = HandConfig {
        num_players: players,
        stack,
        ..HandConfig::default()
    };
    let mut abs = Abstraction::with_centroids(abs_cfg, centroids);
    if let Some(t) = table {
        abs = abs.with_table(t);
    }
    (mirror::Bot { abs: Arc::new(abs), strategies }, cfg)
}

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1)
}
