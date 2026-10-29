//! Table policy: act from the trained blueprint, with an optional
//! time-budgeted online subgame resolve for postflop decisions
//! (`play --search`).
//!
//! The resolve samples hidden cards from Bayes-tracked ranges when a
//! RangeTracker is provided (uniformly otherwise). Flop subgames are
//! depth-limited to the end of the street with biased-continuation leaf
//! values; river spots with two live players use the exact range-vs-range
//! CFR+ solver in `river.rs`; everything else solves to the end of the hand
//! with MCCFR.

use crate::abstraction::{AbsAction, AbsConfig, Abstraction, BetSize, Token};
use crate::cfr::{sample_index, Blueprint, LeafCfg, Spine, TrainConfig, Trainer};
use crate::engine::{Hand, PlayerAction, Players, Street};
use crate::mirror::{Stored, Strategies};
use crate::search::RangeTracker;
use crate::table::RoundRoot;
use crate::valuenet::ValueNet;
use rand::rngs::SmallRng;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// Count of blueprint lookups that found an entry whose action-vector length
/// disagreed with the current menu. Nonzero means the blueprint was trained
/// with a different bet abstraction than this binary uses, so play silently
/// falls back to check/call and is badly degraded — surfaced loudly once.
static MENU_MISMATCHES: AtomicU64 = AtomicU64::new(0);
/// Blueprint lookups and unseen-infoset fallbacks (to check/call), per
/// street. Diagnostic for the probes: how much of a bound is a
/// best-responder farming spots the blueprint never visited.
static LOOKUPS: [AtomicU64; 4] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static FALLBACKS: [AtomicU64; 4] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

/// (lookups, unseen fallbacks) per street since process start.
pub fn fallback_stats() -> [(u64, u64); 4] {
    std::array::from_fn(|i| (LOOKUPS[i].load(Ordering::Relaxed), FALLBACKS[i].load(Ordering::Relaxed)))
}

/// Human-readable fallback summary, or None if nothing was looked up.
pub fn fallback_report() -> Option<String> {
    let st = fallback_stats();
    let total: u64 = st.iter().map(|x| x.0).sum();
    if total == 0 {
        return None;
    }
    let names = ["preflop", "flop", "turn", "river"];
    let parts: Vec<String> = st
        .iter()
        .zip(names)
        .filter(|(x, _)| x.0 > 0)
        .map(|(&(l, f), n)| format!("{n} {f}/{l} ({:.1}%)", 100.0 * f as f64 / l as f64))
        .collect();
    let f: u64 = st.iter().map(|x| x.1).sum();
    Some(format!(
        "unseen-infoset fallbacks to check/call: {f}/{total} ({:.1}%): {}",
        100.0 * f as f64 / total as f64,
        parts.join(", ")
    ))
}
static MENU_MISMATCH_WARNED: AtomicBool = AtomicBool::new(false);

/// Street and wall time (microseconds) of every searched (postflop)
/// decision since process start. Under a parallel probe a decision's thread
/// can steal another hand's work while it waits inside the multiway
/// resolve, so for latency run the probe with RAYON_NUM_THREADS=1.
static SEARCH_MICROS: std::sync::Mutex<Vec<(u8, u64)>> = std::sync::Mutex::new(Vec::new());

/// Searched-decision latency summary, or None if nothing was searched.
pub fn search_timing_report() -> Option<String> {
    let all = SEARCH_MICROS.lock().unwrap().clone();
    if all.is_empty() {
        return None;
    }
    let summary = |mut v: Vec<u64>| {
        v.sort_unstable();
        let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize] as f64 / 1000.0;
        let mean = v.iter().sum::<u64>() as f64 / v.len() as f64 / 1000.0;
        format!(
            "{}, ms mean {mean:.1}, p50 {:.1}, p90 {:.1}, p99 {:.1}, max {:.1}",
            v.len(),
            q(0.5),
            q(0.9),
            q(0.99),
            q(1.0)
        )
    };
    let mut out = format!("searched decisions: {}", summary(all.iter().map(|x| x.1).collect()));
    for (street, name) in [(1, "flop"), (2, "turn"), (3, "river")] {
        let v: Vec<u64> = all.iter().filter(|x| x.0 == street).map(|x| x.1).collect();
        if !v.is_empty() {
            out += &format!("\n  {name}: {}", summary(v));
        }
    }
    Some(out)
}

/// How many stale-menu blueprint lookups have been seen this process.
#[cfg(test)]
pub fn menu_mismatch_count() -> u64 {
    MENU_MISMATCHES.load(Ordering::Relaxed)
}

fn note_menu_mismatch(stored: usize, menu: usize) {
    MENU_MISMATCHES.fetch_add(1, Ordering::Relaxed);
    if !MENU_MISMATCH_WARNED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "warning: blueprint stores {stored}-action strategies but the current menu \
             has {menu} actions — this blueprint was trained with a different bet \
             abstraction. Play falls back to check/call and will be badly degraded. \
             Retrain the blueprint or run it with a matching binary."
        );
    }
}

/// Bet sizes used by the value-net flop solver (33%, 75%, 150% + all-in):
/// richer than the turn solver's data-generation menu, still small enough
/// to keep leaf-refresh network queries affordable.
const FLOP_MENU: [BetSize; 3] = [BetSize::P33, BetSize::P75, BetSize::P150];
/// Bet-size menu for exact turn resolves at the table (75% pot + all-in,
/// matching the value net's training menu): full menus put the turn+river
/// vector tree at tens of GB.
const TURN_MENU: [BetSize; 1] = [BetSize::P75];
/// Turn cards sampled per leaf refresh (of 49) — bounds net queries per
/// refresh at real-time budgets.
const FLOP_QUERY_TURNS: usize = 16;

pub struct Policy {
    /// The training blueprint, or for `from_strategies` an empty one that
    /// carries only the player count and abstraction config. Strategy
    /// lookups go through `strategies`.
    pub blueprint: Arc<Blueprint>,
    /// Every strategy lookup: the training blueprint itself (shared), or a
    /// compact blueprint or overlay.
    pub strategies: Arc<dyn Strategies>,
    pub abs: Arc<Abstraction>,
    /// Belief-state value net: enables ReBeL-style flop solving.
    pub value_net: Option<Arc<ValueNet>>,
    /// Finer card abstraction for the MCCFR subgame solver's own infosets
    /// (Pluribus: lossless on the current street). None = the
    /// blueprint's. Blueprint lookups (leaf rollouts, fallbacks) always
    /// use `abs`.
    pub search_abs: Option<Arc<Abstraction>>,
}

#[derive(Debug, Clone, Copy)]
pub struct SearchParams {
    pub time_ms: u64,
    pub max_iters: u64,
    /// Model opponents as lambda-rational (logit QRE) instead of fully
    /// rational. None = solve toward equilibrium (default).
    pub qre_lambda: Option<f64>,
    /// Safe (gadget) river resolving: the opponent may take a blueprint
    /// safety value instead of entering the subgame, bounding how much
    /// resolving with wrong beliefs can be exploited.
    pub safe_resolve: bool,
    /// Metareasoning: probe with 1/8 of the budget and stop early when the
    /// hero's decision is already near-pure — trivial spots take
    /// milliseconds, the saved time is available to hard ones.
    pub adaptive: bool,
    /// Multiway MCCFR resolves are rooted at the start of the current
    /// betting round (Pluribus) rather than at the current decision.
    pub round_root: bool,
}

impl Default for SearchParams {
    fn default() -> Self {
        SearchParams {
            time_ms: 2_000,
            max_iters: 2_000_000,
            qre_lambda: None,
            safe_resolve: false,
            adaptive: false,
            round_root: true,
        }
    }
}

/// A preflop decision facing an all-in where a call ends the betting: some
/// opponent is all-in, and every other live opponent is either all-in or
/// still owes chips (has yet to respond, and is assumed to fold).
pub fn preflop_allin_spot(real: &Hand) -> bool {
    let Some(hero) = real.to_act() else { return false };
    let bet = real.current_bet();
    let live = |q: usize| q != hero && !real.folded(q);
    real.street() == Street::Preflop
        && real.to_call() > 0
        && (0..real.num_players()).any(|q| live(q) && real.all_in(q))
        && (0..real.num_players()).all(|q| !live(q) || real.all_in(q) || real.street_commit(q) < bet)
}

/// Early-exit purity threshold for adaptive search.
const ADAPTIVE_PURITY: f64 = 0.97;

/// Per-hand search state for continual resolving: the most recent exact
/// turn resolve's carry (opponent CFVs at river entries) plus where in the
/// action history that resolve was rooted. Reset every hand.
#[derive(Default)]
pub struct SearchSession {
    pub carry: Option<crate::turn::TurnCarry>,
    pub root_hist_len: usize,
}

impl SearchSession {
    pub fn new() -> SearchSession {
        SearchSession::default()
    }

    /// Carried gadget alternatives for a river resolve at `h` given the
    /// full table history: valid when the turn line since the resolve is on
    /// the solve's tree. River actions after the street separator do not
    /// invalidate the carry — the entry CFVs are the opponent's constraint
    /// values, and continual resolving keeps those fixed across the
    /// opponent's own actions (the gadget offers "opt out at river entry"),
    /// so villain-led rivers use them too instead of rollout estimates.
    fn river_alt(&self, h: &Hand, hist: &[Token]) -> Option<Vec<f64>> {
        let carry = self.carry.as_ref()?;
        let suffix = hist.get(self.root_hist_len..)?;
        let mut parts = suffix.split(|&t| t == crate::abstraction::Token::StreetSep);
        let line = parts.next()?;
        // Exactly one separator: the turn→river transition.
        parts.next()?;
        if parts.next().is_some() {
            return None;
        }
        carry.alt_for(line, *h.board().get(4)?)
    }
}

impl Policy {
    pub fn new(blueprint: Blueprint, abs: Arc<Abstraction>) -> Self {
        let blueprint = Arc::new(blueprint);
        Policy {
            strategies: blueprint.clone(),
            blueprint,
            abs,
            value_net: None,
            search_abs: None,
        }
    }

    /// A policy over any strategy store (compact blueprint, preflop
    /// overlay). Commands that need the training blueprint's map (distill,
    /// strategic clustering) cannot use it.
    pub fn from_strategies(strategies: Arc<dyn Strategies>, num_players: Players, abs: Arc<Abstraction>) -> Self {
        let blueprint = Arc::new(Blueprint {
            strategies: Default::default(),
            iterations: 0,
            num_players,
            abs_cfg: abs.cfg.clone(),
            centroids: None,
        });
        Policy {
            blueprint,
            strategies,
            abs,
            value_net: None,
            search_abs: None,
        }
    }

    /// Solve multiway subgames over `buckets` postflop card buckets
    /// (equity quantiles; exact per-board on the river) instead of the
    /// blueprint's.
    pub fn with_search_buckets(mut self, buckets: Option<std::num::NonZeroU16>) -> Self {
        self.search_abs = buckets.map(|b| {
            Arc::new(Abstraction::new(AbsConfig {
                postflop_buckets: b,
                cache_cap: 4_000_000,
                ..self.abs.cfg.clone()
            }))
        });
        self
    }

    pub fn with_value_net(mut self, net: Option<Arc<ValueNet>>) -> Self {
        self.value_net = net;
        self
    }

    /// Cheap clone (shared Arcs) with the value net removed — for paired
    /// with-net vs without-net comparisons.
    pub fn clone_without_net(&self) -> Policy {
        Policy {
            blueprint: self.blueprint.clone(),
            strategies: self.strategies.clone(),
            abs: self.abs.clone(),
            value_net: None,
            search_abs: self.search_abs.clone(),
        }
    }

    /// The blueprint's full action distribution at the current infoset —
    /// exactly what `act_blueprint` samples from, including its check/call
    /// fallback on unseen or mismatched infosets.
    pub fn blueprint_dist(&self, h: &Hand, hist: &[Token], rng: &mut SmallRng) -> (Vec<AbsAction>, Vec<f64>) {
        let p = h.to_act().expect("a blueprint decision needs a live hand");
        let acts = self.abs.abstract_actions(h);
        let bucket = self.abs.bucket(h.hole(p), h.board(), rng);
        let street = h.street() as usize;
        LOOKUPS[street].fetch_add(1, Ordering::Relaxed);
        match self.strategies.lookup(bucket, hist, &acts) {
            Stored::Fits(w) => return (acts, w.probs()),
            Stored::Unseen => {}
            // Found the infoset but the stored menu differs: a bet-menu
            // drift between training and this binary, not an unseen spot.
            Stored::OtherMenu(stored) => note_menu_mismatch(stored, acts.len()),
        }
        FALLBACKS[street].fetch_add(1, Ordering::Relaxed);
        let mut probs = vec![0.0; acts.len()];
        let call = acts
            .iter()
            .position(|&a| a == AbsAction::CheckCall)
            .expect("check/call is always legal");
        probs[call] = 1.0;
        (acts, probs)
    }

    /// Pick an abstract action for the player to act on `h` from the
    /// blueprint. `hist` is the abstract action history (tokens).
    /// Falls back to check/call on infosets the blueprint never visited or
    /// whose action menu no longer matches.
    pub fn act_blueprint(&self, h: &Hand, hist: &[Token], rng: &mut SmallRng) -> AbsAction {
        let (acts, probs) = self.blueprint_dist(h, hist, rng);
        acts[sample_index(&probs, rng)]
    }

    /// Resolve the current subgame with MCCFR for a time/iteration budget and
    /// act from the resolved strategy; falls back to the blueprint.
    /// Preflop always plays the blueprint (as Pluribus did). With a
    /// RangeTracker, opponents' hidden cards are sampled from their tracked
    /// ranges; flop subgames are depth-limited to the end of the street with
    /// biased-continuation leaf values.
    ///
    /// Subgames are rooted at `real` — the true table state — so an
    /// opponent's off-tree bet is re-solved at its actual size (nested
    /// re-solving) instead of being priced at the nearest abstract size.
    /// `shadow` (the on-tree mirror) is used only for blueprint infoset
    /// lookups; pass the same hand twice when they cannot diverge.
    #[allow(clippy::too_many_arguments)]
    pub fn act_with_search(
        &self,
        real: &Hand,
        shadow: &Hand,
        hist: &[Token],
        params: SearchParams,
        train_cfg: &TrainConfig,
        tracker: Option<&RangeTracker>,
        session: Option<&mut SearchSession>,
        round: Option<&RoundRoot>,
        rng: &mut SmallRng,
    ) -> AbsAction {
        let started = std::time::Instant::now();
        let (acts, probs) =
            self.search_dist(real, shadow, hist, params, train_cfg, tracker, session, round, rng);
        if real.street() != Street::Preflop {
            let street = real.street() as u8;
            SEARCH_MICROS.lock().unwrap().push((street, started.elapsed().as_micros() as u64));
        }
        acts[sample_index(&probs, rng)]
    }

    /// Preflop, facing an all-in with nothing left to decide after a call
    /// (see `preflop_allin_spot`): call vs fold by Monte Carlo over the
    /// all-in players' near-hard Bayes ranges and the board, players yet
    /// to respond assumed to fold. The blueprint's own strategy at these
    /// rarely reached nodes is barely trained (it called 12–16% of hands,
    /// 93s and Q2o among them, facing a jam over an open). Mixed by the
    /// z-score of the estimate; None when the spot does not qualify.
    pub fn preflop_allin_dist(
        &self,
        real: &Hand,
        shadow: &Hand,
        tracker: &RangeTracker,
        params: SearchParams,
        rng: &mut SmallRng,
    ) -> Option<(Vec<AbsAction>, Vec<f64>)> {
        if !preflop_allin_spot(real) {
            return None;
        }
        let acts = self.abs.abstract_actions(shadow);
        let fold = acts.iter().position(|&a| a == AbsAction::Fold)?;
        let call = acts.iter().position(|&a| a == AbsAction::CheckCall)?;
        let hero = real.to_act()?;
        let hole = real.hole(hero);
        let jammers: Vec<usize> = (0..real.num_players())
            .filter(|&q| q != hero && !real.folded(q) && real.all_in(q))
            .collect();
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(params.time_ms.max(1));
        let (mut n, mut sum, mut sq) = (0u32, 0.0f64, 0.0f64);
        while n < 4000 && (n < 200 || std::time::Instant::now() < deadline) {
            let mut used = [false; 52];
            used[hole[0].index()] = true;
            used[hole[1].index()] = true;
            let mut want = [None; crate::engine::MAX_PLAYERS];
            want[hero] = Some(hole);
            for &q in &jammers {
                want[q] = Some(tracker.sample_hard(q, &mut used, rng)?);
            }
            let mut h = real.clone();
            h.resample_hidden_with(&want, rng);
            let value = |first: PlayerAction| {
                let mut g = h.clone();
                g.apply(first);
                while !g.is_terminal() {
                    let a = if g.can_check() || g.to_act() == Some(hero) {
                        PlayerAction::CheckCall
                    } else {
                        PlayerAction::Fold
                    };
                    g.apply(a);
                }
                g.utilities()[hero] as f64
            };
            let d = value(PlayerAction::CheckCall) - value(PlayerAction::Fold);
            n += 1;
            sum += d;
            sq += d * d;
        }
        let mean = sum / n as f64;
        let se = ((sq / n as f64 - mean * mean).max(0.0) / n as f64).sqrt().max(1e-9);
        let p_call = 1.0 / (1.0 + (-1.7 * mean / se).exp());
        let mut probs = vec![0.0; acts.len()];
        probs[fold] = 1.0 - p_call;
        probs[call] = p_call;
        Some((acts, probs))
    }

    /// The full action distribution the search would act from — the routing
    /// behind `act_with_search`, exposed for distillation (the resolved
    /// distribution is the teacher signal, not just a sampled action).
    /// Falls back to the blueprint distribution when no solver claims the
    /// spot.
    #[allow(clippy::too_many_arguments)]
    pub fn search_dist(
        &self,
        real: &Hand,
        shadow: &Hand,
        hist: &[Token],
        params: SearchParams,
        train_cfg: &TrainConfig,
        tracker: Option<&RangeTracker>,
        session: Option<&mut SearchSession>,
        round: Option<&RoundRoot>,
        rng: &mut SmallRng,
    ) -> (Vec<AbsAction>, Vec<f64>) {
        if real.street() == Street::Preflop {
            if let Some(tr) = tracker {
                if let Some(d) = self.preflop_allin_dist(real, shadow, tr, params, rng) {
                    return d;
                }
            }
            return self.blueprint_dist(shadow, hist, rng);
        }
        // River spots with two live players: exact range-vs-range CFR+
        // (river.rs) instead of sampled MCCFR.
        if real.street() == Street::River && real.live_count() == 2 {
            if let Some(tr) = tracker {
                let sess = session.as_deref();
                if let Some(d) = self.river_dist(real, hist, tr, params, sess, rng) {
                    return d;
                }
            }
        }
        // Turn spots with two live players: exact turn+river vector solve
        // (slim bet menu), optionally inside the safety gadget. The solve's
        // river-entry CFVs are carried in the session for the river
        // resolve's gadget (continual resolving).
        if real.street() == Street::Turn && real.live_count() == 2 {
            if let Some(tr) = tracker {
                if let Some(d) = self.turn_dist(real, hist, tr, params, session, rng) {
                    return d;
                }
            }
        }
        // Flop spots with two live players and a value net: ReBeL-style
        // depth-limited vector solving with learned leaf values.
        if real.street() == Street::Flop && real.live_count() == 2 {
            if let (Some(tr), Some(net)) = (tracker, &self.value_net) {
                if let Some(d) = self.flop_net_dist(real, tr, net, params, rng) {
                    return d;
                }
            }
            // Without a net, a table-budget MCCFR resolve is worse than the
            // blueprint here (measured: ≈0 in self-play; vs Slumbot it got
            // 200bb in at ~18% equity on 51 flops in 10k hands). The exact
            // turn/river solvers take over on later streets.
            return self.blueprint_dist(shadow, hist, rng);
        }
        let leaf = (real.street() == Street::Flop).then(|| LeafCfg {
            strategies: self.strategies.clone(),
            abs: self.abs.clone(),
            limit: Street::Flop,
        });
        let sabs = self.search_abs.clone().unwrap_or_else(|| self.abs.clone());
        let p = real.to_act().expect("a search decision needs a live hand");
        // Round-start rooting: solve from the start of this betting round
        // with the actions taken so far as the spine (see cfr::Spine).
        let round = round.filter(|r| {
            params.round_root
                && !r.actions.is_empty()
                && r.real.street() == real.street()
                && r.hist_len + r.actions.len() == hist.len()
        });
        let solver = match round {
            Some(r) => {
                let spine = Spine {
                    hero: p,
                    base_len: r.hist_len,
                    steps: r.actions.clone(),
                    tokens: hist[r.hist_len..].to_vec(),
                };
                let start = tracker.map(|t| t.at_round_start());
                resolve_subgame(
                    sabs.clone(),
                    train_cfg,
                    &r.real,
                    &hist[..r.hist_len],
                    params,
                    start.as_ref(),
                    leaf,
                    Some(spine),
                )
            }
            None => resolve_subgame(sabs, train_cfg, real, hist, params, tracker, leaf, None),
        };
        let bucket = solver.abs.bucket(real.hole(p), real.board(), rng);
        if let Some(s) = solver.avg_strategy(bucket, hist) {
            let acts = solver.abs.abstract_actions(real);
            if s.len() == acts.len() {
                return (acts, s);
            }
        }
        self.blueprint_dist(shadow, hist, rng)
    }

    /// Depth-limited flop resolve over both players' tracked ranges with
    /// value-net leaves. None when the spot doesn't qualify or the hero's
    /// combo got no strategy weight (caller falls back to MCCFR search).
    fn flop_net_dist(
        &self,
        h: &Hand,
        tracker: &RangeTracker,
        net: &Arc<ValueNet>,
        params: SearchParams,
        _rng: &mut SmallRng,
    ) -> Option<(Vec<AbsAction>, Vec<f64>)> {
        let hero = h.to_act()?;
        let villain = (0..h.num_players()).find(|&p| p != hero && !h.folded(p))?;
        let net_ref: &ValueNet = net;
        let mut solver = crate::flop::FlopSolver::build_sampled(
            h,
            &self.abs,
            [tracker.seat_weights(hero), tracker.seat_weights(villain)],
            &FLOP_MENU,
            net_ref,
            25,
            FLOP_QUERY_TURNS,
        )?;
        if params.adaptive {
            solver.solve_adaptive(10_000, params.time_ms, h.hole(hero), ADAPTIVE_PURITY);
        } else {
            solver.solve(10_000, params.time_ms);
        }
        solver.root_strategy(h.hole(hero))
    }

    /// Exact turn+river resolve over both players' tracked ranges (vector
    /// CFR+ with the slim `TURN_MENU`, so the tree stays table-budget
    /// sized). With `safe_resolve`, runs inside the Burch gadget using
    /// rollout-estimated safety values. None when the spot doesn't qualify
    /// (caller falls back to sampled MCCFR).
    fn turn_dist(
        &self,
        h: &Hand,
        hist: &[Token],
        tracker: &RangeTracker,
        params: SearchParams,
        session: Option<&mut SearchSession>,
        rng: &mut SmallRng,
    ) -> Option<(Vec<AbsAction>, Vec<f64>)> {
        let hero = h.to_act()?;
        let villain = (0..h.num_players()).find(|&p| p != hero && !h.folded(p))?;
        let mut solver = crate::turn::TurnSolver::build(
            h,
            &self.abs,
            [tracker.seat_weights(hero), tracker.seat_weights(villain)],
            &TURN_MENU,
        )?;
        if params.safe_resolve {
            let alt = self.estimate_alt(h, hist, tracker, villain, rng);
            solver = solver.with_gadget(alt);
        }
        solver.solve(200, params.time_ms);
        let dist = solver.root_strategy(h.hole(hero))?;
        if let Some(sess) = session {
            sess.carry = Some(solver.extract_carry());
            sess.root_hist_len = hist.len();
        }
        Some(dist)
    }

    /// Exact river resolve over both players' tracked ranges. None when the
    /// spot doesn't qualify or the hero's combo got no strategy weight, in
    /// which case the caller falls back to sampled MCCFR. With
    /// `safe_resolve`, the solve runs the Burch resolving gadget using
    /// rollout-estimated blueprint safety values for the opponent.
    fn river_dist(
        &self,
        h: &Hand,
        hist: &[Token],
        tracker: &RangeTracker,
        params: SearchParams,
        session: Option<&SearchSession>,
        rng: &mut SmallRng,
    ) -> Option<(Vec<AbsAction>, Vec<f64>)> {
        let hero = h.to_act()?;
        let villain = (0..h.num_players()).find(|&p| p != hero && !h.folded(p))?;
        let mut solver = crate::river::RiverSolver::build(
            h,
            &self.abs,
            [tracker.seat_weights(hero), tracker.seat_weights(villain)],
        )?;
        if params.safe_resolve {
            // Continual resolving: prefer the turn resolve's carried CFVs
            // as the gadget alternatives; fall back to rollout estimates
            // when the line left the carry's tree.
            let alt = session
                .and_then(|s| s.river_alt(h, hist))
                .unwrap_or_else(|| self.estimate_alt(h, hist, tracker, villain, rng));
            solver = solver.with_gadget(alt);
        }
        if params.adaptive {
            solver.solve_adaptive(
                10_000,
                params.time_ms,
                params.qre_lambda,
                h.hole(hero),
                ADAPTIVE_PURITY,
            );
        } else {
            solver.solve(10_000, params.time_ms, params.qre_lambda);
        }
        solver.root_strategy(h.hole(hero))
    }

    /// Rollout-estimated safety values for the opponent: for each of its
    /// combos, the mean utility of blueprint-vs-blueprint playouts from this
    /// state with the opponent pinned to that combo and everyone else drawn
    /// from the tracked ranges — an estimate of what the combo was worth
    /// before we replaced the blueprint with a resolved strategy.
    /// Street-agnostic: rollouts run from `h` to the end of the hand.
    fn estimate_alt(
        &self,
        h: &Hand,
        hist: &[Token],
        tracker: &RangeTracker,
        villain: usize,
        rng: &mut SmallRng,
    ) -> Vec<f64> {
        use rayon::prelude::*;
        const ROLLOUTS: usize = 6;
        let combos = crate::search::all_combos();
        let base_seed: u64 = rand::Rng::random(rng);
        let board = h.board().to_vec();

        (0..crate::search::NUM_COMBOS)
            .into_par_iter()
            .map(|ci| {
                let c = combos[ci];
                if board.contains(&c[0]) || board.contains(&c[1]) {
                    return 0.0;
                }
                let mut rng = <SmallRng as rand::SeedableRng>::seed_from_u64(
                    base_seed ^ (ci as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                );
                let mut sum = 0.0;
                for _ in 0..ROLLOUTS {
                    let mut sim = h.clone();
                    // Sample everyone from the tracker, then pin the villain
                    // (rejecting samples that collide with the pinned combo).
                    let mut want = tracker.sample_holes(&sim, &mut rng);
                    for _ in 0..10 {
                        let clash = want
                            .iter()
                            .enumerate()
                            .any(|(p, w)| {
                                p != villain
                                    && w.is_some_and(|w| {
                                        w.contains(&c[0]) || w.contains(&c[1])
                                    })
                            });
                        if !clash {
                            break;
                        }
                        want = tracker.sample_holes(&sim, &mut rng);
                    }
                    want[villain] = Some(c);
                    let clash = want.iter().enumerate().any(|(p, w)| {
                        p != villain
                            && w.is_some_and(|w| w.contains(&c[0]) || w.contains(&c[1]))
                    });
                    if clash {
                        for (p, w) in want.iter_mut().enumerate() {
                            if p != villain {
                                *w = None; // uniform fallback
                            }
                        }
                    }
                    sim.resample_hidden_with(&want, &mut rng);

                    let mut hist2 = hist.to_vec();
                    let mut guard = 0;
                    while !sim.is_terminal() {
                        guard += 1;
                        debug_assert!(guard < 100, "alt rollout did not terminate");
                        if guard >= 100 {
                            break;
                        }
                        let a = self.act_blueprint(&sim, &hist2, &mut rng);
                        let street_before = sim.street();
                        sim.apply(self.abs.concrete(&sim, a));
                        hist2.push(a.token());
                        if !sim.is_terminal() && sim.street() != street_before {
                            hist2.push(crate::abstraction::Token::StreetSep);
                        }
                    }
                    if sim.is_terminal() {
                        sum += sim.utilities()[villain] as f64;
                    }
                }
                sum / ROLLOUTS as f64
            })
            .collect()
    }
}

/// Build a fresh subgame solver rooted at `root` (hidden cards get resampled
/// every traversal — from tracked ranges when a tracker is given) and train
/// it for the given budget. The abstraction is shared with the caller so
/// equity/bucket caches persist across decisions.
#[allow(clippy::too_many_arguments)]
pub fn resolve_subgame(
    abs: Arc<Abstraction>,
    train_cfg: &TrainConfig,
    root: &Hand,
    root_hist: &[Token],
    params: SearchParams,
    tracker: Option<&RangeTracker>,
    leaf: Option<LeafCfg>,
    spine: Option<Spine>,
) -> Trainer {
    let t = Trainer::new(abs, train_cfg.clone())
        .with_leaf(leaf)
        .with_plus(true)
        .with_qre(params.qre_lambda)
        .with_spine(spine);
    match tracker {
        Some(tr) => {
            let sampler = move |h: &mut Hand, rng: &mut SmallRng| {
                let holes = tr.sample_holes(h, rng);
                h.resample_hidden_with(&holes, rng);
            };
            t.run_subgame(root, root_hist, params.time_ms, params.max_iters, Some(&sampler));
        }
        None => t.run_subgame(root, root_hist, params.time_ms, params.max_iters, None),
    }
    t
}

// ---------------------------------------------------------------------------
// Tests (written first, TDD)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::AbsConfig;
    use crate::cards::{fresh_deck, parse_cards, Card, Hole};
    use crate::cfr::make_key;
    use crate::engine::{HandConfig, PlayerAction, Players, Stack};
    use rand::SeedableRng;

    fn abs_small() -> Abstraction {
        Abstraction::new(AbsConfig {
            postflop_buckets: std::num::NonZero::new(6).unwrap(),
            equity_rollouts: std::num::NonZero::new(50).unwrap(),
            cache_cap: 100_000,
            ..AbsConfig::default()
        })
    }

    #[test]
    fn policy_follows_blueprint_when_available() {
        let abs = abs_small();
        let h = Hand::new(&HandConfig::default(), 0, fresh_deck());
        let acts = abs.abstract_actions(&h);
        let bucket = crate::abstraction::preflop_bucket(*h.hole(3));

        // Blueprint that always folds at this exact infoset.
        let mut strategies = crate::cfr::StrategyMap::default();
        let mut probs = vec![0.0f32; acts.len()];
        probs[0] = 1.0; // Fold is always first when facing a bet
        strategies.insert(make_key(bucket, &[]).to_vec(), probs);
        let bp = Blueprint {
            strategies,
            iterations: 1,
            num_players: Players::SIX,
            abs_cfg: AbsConfig::default(),
            centroids: None,
        };

        let policy = Policy::new(bp, Arc::new(abs));
        let mut rng = SmallRng::seed_from_u64(1);
        for _ in 0..20 {
            assert_eq!(policy.act_blueprint(&h, &[], &mut rng), AbsAction::Fold);
        }
    }

    /// A blueprint whose stored strategy length no longer matches the
    /// current action menu (e.g. loaded under a binary with a different bet
    /// abstraction) must register the mismatch, not silently degrade to a
    /// calling station. The count distinguishes this from a genuinely
    /// unseen infoset (which legitimately falls back to check/call).
    #[test]
    fn stale_menu_blueprint_is_flagged_not_silently_degraded() {
        let abs = abs_small();
        let h = Hand::new(&HandConfig::default(), 0, fresh_deck());
        let acts = abs.abstract_actions(&h);
        let bucket = crate::abstraction::preflop_bucket(*h.hole(3));

        // Stored strategy has one FEWER action than the current menu — the
        // signature of a blueprint trained with a narrower bet menu.
        let mut strategies = crate::cfr::StrategyMap::default();
        strategies.insert(make_key(bucket, &[]).to_vec(), vec![0.5f32; acts.len() - 1]);
        let bp = Blueprint {
            strategies,
            iterations: 1,
            num_players: Players::SIX,
            abs_cfg: AbsConfig::default(),
            centroids: None,
        };
        let policy = Policy::new(bp, Arc::new(abs));
        let mut rng = SmallRng::seed_from_u64(1);

        let before = super::menu_mismatch_count();
        // It still returns a legal action (check/call fallback)...
        assert_eq!(policy.act_blueprint(&h, &[], &mut rng), AbsAction::CheckCall);
        // ...but the mismatch is recorded rather than swallowed.
        assert!(
            super::menu_mismatch_count() > before,
            "a length-mismatched stored strategy must be counted"
        );
    }

    #[test]
    fn policy_falls_back_to_checkcall_when_unseen() {
        let abs = abs_small();
        let h = Hand::new(&HandConfig::default(), 0, fresh_deck());
        let bp = Blueprint {
            strategies: Default::default(),
            iterations: 0,
            num_players: Players::SIX,
            abs_cfg: AbsConfig::default(),
            centroids: None,
        };
        let policy = Policy::new(bp, Arc::new(abs));
        let mut rng = SmallRng::seed_from_u64(1);
        assert_eq!(
            policy.act_blueprint(&h, &[], &mut rng),
            AbsAction::CheckCall
        );
    }

    /// Subgame resolving on a rigged river: hero holds the nuts facing an
    /// all-in. The resolved strategy must call nearly always.
    #[test]
    fn search_calls_the_nuts_on_the_river() {
        // Heads-up. p0 (button/SB) has a royal flush by the river.
        let front = parse_cards("As Ks 2c 7d Qs Js Ts 3h 4d").unwrap();
        let mut deck = fresh_deck();
        let mut used = [false; 52];
        for (i, &c) in front.iter().enumerate() {
            deck[i] = c;
            used[c.index()] = true;
        }
        let mut idx = front.len();
        for c in Card::ALL {
            if !used[c.index()] {
                deck[idx] = c;
                idx += 1;
            }
        }

        let hand_cfg = HandConfig::new(Players::TWO, Stack::new(2_000).unwrap());
        let mut h = Hand::new(&hand_cfg, 0, deck);
        let abs = abs_small();
        let mut hist: Vec<Token> = Vec::new();

        // p0 calls, p1 checks -> flop; checks to the river; p1 shoves.
        let do_act = |h: &mut Hand, hist: &mut Vec<Token>, a: AbsAction, abs: &Abstraction| {
            let street_before = h.street();
            h.apply(abs.concrete(h, a));
            hist.push(a.token());
            if !h.is_terminal() && h.street() != street_before {
                hist.push(crate::abstraction::Token::StreetSep);
            }
        };
        do_act(&mut h, &mut hist, AbsAction::CheckCall, &abs); // p0 limps
        do_act(&mut h, &mut hist, AbsAction::CheckCall, &abs); // p1 checks
        for _ in 0..2 {
            do_act(&mut h, &mut hist, AbsAction::CheckCall, &abs); // p1 checks
            do_act(&mut h, &mut hist, AbsAction::CheckCall, &abs); // p0 checks
        }
        assert_eq!(h.street(), crate::engine::Street::River);
        assert_eq!(h.to_act(), Some(1));
        do_act(&mut h, &mut hist, AbsAction::AllIn, &abs); // p1 shoves
        assert_eq!(h.to_act(), Some(0));
        assert!(!h.is_terminal());

        let train_cfg = TrainConfig {
            hand: hand_cfg,
            prune_after: u64::MAX,
            ..TrainConfig::default()
        };
        let solver = resolve_subgame(
            Arc::new(Abstraction::new(AbsConfig {
                postflop_buckets: std::num::NonZero::new(6).unwrap(),
                equity_rollouts: std::num::NonZero::new(50).unwrap(),
                cache_cap: 100_000,
                ..AbsConfig::default()
            })),
            &train_cfg,
            &h,
            &hist,
            SearchParams {
                time_ms: 60_000, // generous: survives CPU contention in CI
                max_iters: 20_000,
                qre_lambda: None,
                safe_resolve: false,
                adaptive: false,
                round_root: true,
            },
            None,
            None,
        None,
        );

        let mut rng = SmallRng::seed_from_u64(2);
        let bucket = solver.abs.bucket(h.hole(0), h.board(), &mut rng);
        assert_eq!(bucket, 5, "royal flush must be in the top bucket");
        let strat = solver
            .avg_strategy(bucket, &hist)
            .expect("root infoset must be trained");
        // Actions facing a shove: [Fold, CheckCall].
        assert!(
            strat[1] > 0.9,
            "must call all-in with the nuts, got {:?}",
            strat
        );
        // Sanity: engine agrees calling ends the hand with hero winning.
        let mut done = h.clone();
        done.apply(PlayerAction::CheckCall);
        assert!(done.is_terminal());
        assert!(done.utilities()[0] > 0);
    }

    /// Round-rooted three-way river resolve: the subgame is rooted at the
    /// start of the river with the two opponents' actions (shove, call)
    /// as the spine; the hero holding the nuts must still call.
    #[test]
    fn round_rooted_search_calls_the_nuts_three_way() {
        use crate::table::Table;
        let front = parse_cards("As Ks 2c 7d 8h 9c Qs Js Ts 3h 4d").unwrap();
        let mut deck = fresh_deck();
        let mut used = [false; 52];
        for (i, &c) in front.iter().enumerate() {
            deck[i] = c;
            used[c.index()] = true;
        }
        let mut idx = front.len();
        for c in Card::ALL {
            if !used[c.index()] {
                deck[idx] = c;
                idx += 1;
            }
        }
        let hand_cfg = HandConfig::new(Players::THREE, Stack::new(2_000).unwrap());
        // The blueprint's 6 river buckets are too coarse here: against
        // the resolved (strong) shoving range the top bucket mixes the
        // nuts with hands that correctly fold (~18% fold measured), so
        // the solver gets its own finer abstraction (Pluribus: lossless
        // on the current street).
        let abs = Arc::new(Abstraction::new(AbsConfig {
            postflop_buckets: std::num::NonZero::new(6).unwrap(),
            equity_rollouts: std::num::NonZero::new(50).unwrap(),
            cache_cap: 100_000,
            ..AbsConfig::default()
        }));
        let policy = Policy::new(
            Blueprint {
                strategies: Default::default(),
                iterations: 0,
                num_players: Players::THREE,
                abs_cfg: AbsConfig::default(),
                centroids: None,
            },
            abs.clone(),
        )
        .with_search_buckets(std::num::NonZero::new(30));
        let mut table = Table::new(&hand_cfg, 0, deck);
        let mut tracker = RangeTracker::new(3);
        // p0 calls, p1 calls, p2 checks; checked around to the river.
        for _ in 0..3 {
            table.apply_abs(AbsAction::CheckCall, &abs);
        }
        tracker.exclude(table.real.board());
        for _ in 0..2 {
            for _ in 0..3 {
                table.apply_abs(AbsAction::CheckCall, &abs);
            }
            tracker.exclude(table.real.board());
        }
        assert_eq!(table.real.street(), Street::River);
        assert_eq!(table.real.to_act(), Some(1));
        table.apply_abs(AbsAction::AllIn, &abs); // p1 shoves
        table.apply_abs(AbsAction::CheckCall, &abs); // p2 calls
        assert_eq!(table.real.to_act(), Some(0));
        assert_eq!(table.real.live_count(), 3);
        assert_eq!(table.round.actions.len(), 2);

        let train_cfg = TrainConfig {
            hand: hand_cfg,
            prune_after: u64::MAX,
            ..TrainConfig::default()
        };
        let params = SearchParams {
            time_ms: 60_000,
            max_iters: 40_000,
            ..SearchParams::default()
        };
        let mut rng = SmallRng::seed_from_u64(3);
        let (acts, probs) = policy.search_dist(
            &table.real,
            &table.shadow,
            &table.hist,
            params,
            &train_cfg,
            Some(&tracker),
            None,
            Some(&table.round),
            &mut rng,
        );
        assert_eq!(acts, vec![AbsAction::Fold, AbsAction::CheckCall]);
        assert!(probs[1] > 0.9, "must call with the nuts three-way, got {probs:?}");
    }

    /// Safe (gadget) river resolving end to end through act_with_search:
    /// rollout-estimated safety values + gadget solve must still call the
    /// shove with the nuts.
    #[test]
    fn safe_resolve_calls_the_nuts_on_the_river() {
        use crate::search::RangeTracker;
        let front = parse_cards("As Ks 2c 7d Qs Js Ts 3h 4d").unwrap();
        let mut deck = fresh_deck();
        let mut used = [false; 52];
        for (i, &c) in front.iter().enumerate() {
            deck[i] = c;
            used[c.index()] = true;
        }
        let mut idx = front.len();
        for c in Card::ALL {
            if !used[c.index()] {
                deck[idx] = c;
                idx += 1;
            }
        }
        let hand_cfg = HandConfig::new(Players::TWO, Stack::new(2_000).unwrap());
        let mut h = Hand::new(&hand_cfg, 0, deck);
        let abs = Arc::new(abs_small());
        let policy = Policy::new(
            Blueprint {
                strategies: Default::default(),
                iterations: 0,
                num_players: Players::TWO,
                abs_cfg: AbsConfig::default(),
                centroids: None,
            },
            abs.clone(),
        );
        let mut hist: Vec<Token> = Vec::new();
        let mut tracker = RangeTracker::new(2);
        let act = |h: &mut Hand, hist: &mut Vec<Token>, a: AbsAction| {
            let street_before = h.street();
            h.apply(abs.concrete(h, a));
            hist.push(a.token());
            if !h.is_terminal() && h.street() != street_before {
                hist.push(crate::abstraction::Token::StreetSep);
            }
        };
        for _ in 0..3 {
            act(&mut h, &mut hist, AbsAction::CheckCall);
            act(&mut h, &mut hist, AbsAction::CheckCall);
        }
        assert_eq!(h.street(), Street::River);
        act(&mut h, &mut hist, AbsAction::AllIn); // p1 shoves
        assert_eq!(h.to_act(), Some(0));
        tracker.exclude(h.board());

        let train_cfg = TrainConfig {
            hand: hand_cfg,
            prune_after: u64::MAX,
            ..TrainConfig::default()
        };
        let mut rng = SmallRng::seed_from_u64(5);
        let a = policy.act_with_search(
            &h,
            &h,
            &hist,
            SearchParams {
                time_ms: 5_000,
                max_iters: 300,
                qre_lambda: None,
                safe_resolve: true,
                adaptive: false,
                round_root: true,
            },
            &train_cfg,
            Some(&tracker),
            None,
            None,
            &mut rng,
        );
        assert_eq!(a, AbsAction::CheckCall, "royal must call even with the gadget");
    }

    /// Turn decisions with two live players route to the exact turn+river
    /// vector solver: a made royal facing a turn shove must call, in both
    /// plain and gadget (safe) resolving modes.
    #[test]
    fn exact_turn_resolve_calls_the_nuts_facing_a_shove() {
        use crate::search::RangeTracker;
        use crate::table::Table;
        let front = parse_cards("As Ks 2c 7d Qs Js Ts 3h 4d").unwrap();
        let mut deck = fresh_deck();
        let mut used = [false; 52];
        for (i, &c) in front.iter().enumerate() {
            deck[i] = c;
            used[c.index()] = true;
        }
        let mut idx = front.len();
        for c in Card::ALL {
            if !used[c.index()] {
                deck[idx] = c;
                idx += 1;
            }
        }
        let hand_cfg = HandConfig::new(Players::TWO, Stack::new(2_000).unwrap());
        let abs = Arc::new(abs_small());
        let policy = Policy::new(
            Blueprint {
                strategies: Default::default(),
                iterations: 0,
                num_players: Players::TWO,
                abs_cfg: AbsConfig::default(),
                centroids: None,
            },
            abs.clone(),
        );
        let train_cfg = TrainConfig {
            hand: hand_cfg.clone(),
            prune_after: u64::MAX,
            ..TrainConfig::default()
        };
        for safe in [false, true] {
            let mut table = Table::new(&hand_cfg, 0, deck);
            for _ in 0..4 {
                table.apply_abs(AbsAction::CheckCall, &abs);
            }
            assert_eq!(table.real.street(), Street::Turn);
            table.apply_abs(AbsAction::AllIn, &abs); // p1 shoves the turn
            assert_eq!(table.real.to_act(), Some(0));
            let mut tracker = RangeTracker::new(2);
            tracker.exclude(table.real.board());

            let mut rng = SmallRng::seed_from_u64(11);
            let a = policy.act_with_search(
                &table.real,
                &table.shadow,
                &table.hist,
                SearchParams {
                    time_ms: 30_000,
                    max_iters: 20_000,
                    qre_lambda: None,
                    safe_resolve: safe,
                    adaptive: false,
                    round_root: true,
                },
                &train_cfg,
                Some(&tracker),
                None,
                None,
                &mut rng,
            );
            assert_eq!(
                a,
                AbsAction::CheckCall,
                "made royal must call the turn shove (safe_resolve={safe})"
            );
        }
    }

    /// Continual resolving end to end: the bot's exact turn resolve must
    /// populate the session carry, and its river decision must be able to
    /// consume the carried CFVs as gadget alternatives (the line/river
    /// lookup succeeds on the play path actually taken).
    #[test]
    fn turn_resolve_carries_cfvs_into_the_river_gadget() {
        use crate::search::RangeTracker;
        use crate::table::Table;
        // Bot is p1 (first to act postflop) with a medium hand; we are p0.
        let front = parse_cards("2c 7d Ah Kd Qs Js Ts 3h 4d").unwrap();
        let mut deck = fresh_deck();
        let mut used = [false; 52];
        for (i, &c) in front.iter().enumerate() {
            deck[i] = c;
            used[c.index()] = true;
        }
        let mut idx = front.len();
        for c in Card::ALL {
            if !used[c.index()] {
                deck[idx] = c;
                idx += 1;
            }
        }
        let hand_cfg = HandConfig::new(Players::TWO, Stack::new(2_000).unwrap());
        let abs = Arc::new(abs_small());
        let policy = Policy::new(
            Blueprint {
                strategies: Default::default(),
                iterations: 0,
                num_players: Players::TWO,
                abs_cfg: AbsConfig::default(),
                centroids: None,
            },
            abs.clone(),
        );
        let train_cfg = TrainConfig {
            hand: hand_cfg.clone(),
            prune_after: u64::MAX,
            ..TrainConfig::default()
        };
        let params = SearchParams {
            time_ms: 30_000,
            max_iters: 20_000,
            qre_lambda: None,
            safe_resolve: true,
            adaptive: false,
            round_root: true,
        };
        let mut table = Table::new(&hand_cfg, 0, deck);
        for _ in 0..4 {
            table.apply_abs(AbsAction::CheckCall, &abs);
        }
        assert_eq!(table.real.street(), Street::Turn);
        assert_eq!(table.real.to_act(), Some(1), "bot (p1) opens the turn");

        let mut tracker = RangeTracker::new(2);
        tracker.exclude(table.real.board());
        let mut session = SearchSession::new();
        let mut rng = SmallRng::seed_from_u64(3);

        // Bot's turn decision: must populate the carry.
        let a1 = policy.act_with_search(
            &table.real,
            &table.shadow,
            &table.hist,
            params,
            &train_cfg,
            Some(&tracker),
            Some(&mut session),
            None,
            &mut rng,
        );
        assert!(session.carry.is_some(), "turn resolve must store the carry");
        let root_len = session.root_hist_len;
        assert_eq!(root_len, table.hist.len());
        table.apply_abs(a1, &abs);
        if table.real.street() == Street::Turn && table.real.to_act() == Some(0) {
            // We call/check behind to close the turn.
            table.apply_abs(AbsAction::CheckCall, &abs);
        }
        if table.real.is_terminal() || table.real.street() != Street::River {
            panic!("rig must reach the river, bot chose {a1:?}");
        }
        assert_eq!(table.real.to_act(), Some(1), "bot opens the river");

        // The exact lookup the river resolve performs must succeed on this
        // line: the carried alternatives exist for the realized river card.
        let alt = session
            .river_alt(&table.real, &table.hist)
            .expect("carried CFVs must cover the line actually played");
        assert_eq!(alt.len(), crate::search::NUM_COMBOS);
        assert!(alt.iter().all(|v| v.is_finite()));

        // Villain river actions must not invalidate the carry: the entry
        // CFVs are the opponent's constraint values and those are unchanged
        // by the opponent's own actions (continual resolving). A villain
        // check and a villain bet ahead of us must both still match, and
        // must return the same entry values as the unled lookup.
        for tok in [AbsAction::CheckCall.token(), AbsAction::Bet(BetSize::P75).token()] {
            let mut led = table.hist.clone();
            led.push(tok);
            let led_alt = session
                .river_alt(&table.real, &led)
                .expect("villain-led river must still use the carried CFVs");
            assert_eq!(led_alt, alt);
        }
        // A second street separator can never occur after a turn resolve;
        // such a line is off the carry and must miss cleanly.
        let mut two_seps = table.hist.clone();
        two_seps.push(crate::abstraction::Token::StreetSep);
        assert!(session.river_alt(&table.real, &two_seps).is_none());

        // And the river decision itself completes through the gadget path.
        let a2 = policy.act_with_search(
            &table.real,
            &table.shadow,
            &table.hist,
            params,
            &train_cfg,
            Some(&tracker),
            Some(&mut session),
            None,
            &mut rng,
        );
        let menu = abs.abstract_actions(&table.real);
        assert!(menu.contains(&a2), "river action must be legal, got {a2:?}");
    }

    /// Heads-up flop with no value net: search must play the blueprint, not
    /// an 800ms MCCFR resolve. Measured twice: no-net flop resolves ≈ 0 vs
    /// blueprint in self-play (+30±129), and vs Slumbot they produced 51
    /// flop all-ins at ~18% average equity (−6600 bb / 10k hands).
    #[test]
    fn no_net_flop_search_falls_back_to_the_blueprint() {
        use crate::search::RangeTracker;
        use crate::table::Table;
        let front = parse_cards("2c 7d Ah Kd Qs Js Ts").unwrap();
        let mut deck = fresh_deck();
        let mut used = [false; 52];
        for (i, &c) in front.iter().enumerate() {
            deck[i] = c;
            used[c.index()] = true;
        }
        let mut idx = front.len();
        for c in Card::ALL {
            if !used[c.index()] {
                deck[idx] = c;
                idx += 1;
            }
        }
        let hand_cfg = HandConfig::new(Players::TWO, Stack::new(2_000).unwrap());
        let abs = Arc::new(abs_small());
        let policy = Policy::new(
            Blueprint {
                strategies: Default::default(),
                iterations: 0,
                num_players: Players::TWO,
                abs_cfg: AbsConfig::default(),
                centroids: None,
            },
            abs.clone(),
        );
        let train_cfg = TrainConfig {
            hand: hand_cfg.clone(),
            prune_after: u64::MAX,
            ..TrainConfig::default()
        };
        let params = SearchParams {
            time_ms: 100,
            max_iters: 5_000,
            qre_lambda: None,
            safe_resolve: true,
            adaptive: false,
            round_root: true,
        };
        let mut table = Table::new(&hand_cfg, 0, deck);
        table.apply_abs(AbsAction::CheckCall, &abs);
        table.apply_abs(AbsAction::CheckCall, &abs);
        assert_eq!(table.real.street(), Street::Flop);
        let tracker = RangeTracker::new(2);
        let mut rng = SmallRng::seed_from_u64(3);
        let (acts, probs) = policy.search_dist(
            &table.real,
            &table.shadow,
            &table.hist,
            params,
            &train_cfg,
            Some(&tracker),
            None,
            None,
            &mut rng,
        );
        let mut rng2 = SmallRng::seed_from_u64(3);
        let (bp_acts, bp_probs) = policy.blueprint_dist(&table.shadow, &table.hist, &mut rng2);
        assert_eq!(acts, bp_acts);
        assert_eq!(probs, bp_probs, "no-net flop must be pure blueprint");
    }

    /// Nested re-solving of off-tree bets: the bot must price an off-tree
    /// bet at its REAL amount, not the nearest abstract size. Villain bets
    /// a real 800 into a 200 pot, which log-maps to the 400 abstract bet.
    /// Hero's bluff-catcher has 42% equity against the tracked range: a
    /// call at the mapped price (needs 40%) but a fold at the real price
    /// (needs 44.4%). Solving from the real state must fold; the control
    /// (solving from the shadow, the old behavior) must call — which also
    /// validates that the pricing window is set up correctly.
    #[test]
    fn search_prices_off_tree_bets_from_the_real_state() {
        use crate::search::RangeTracker;
        use crate::table::Table;
        let front = parse_cards("Ah 3c Kd 9c Qs Js Ts 3h 4d").unwrap();
        let mut deck = fresh_deck();
        let mut used = [false; 52];
        for (i, &c) in front.iter().enumerate() {
            deck[i] = c;
            used[c.index()] = true;
        }
        let mut idx = front.len();
        for c in Card::ALL {
            if !used[c.index()] {
                deck[idx] = c;
                idx += 1;
            }
        }
        let hand_cfg = HandConfig::new(Players::TWO, Stack::new(2_000).unwrap());
        // The 800-into-200 off-tree bet maps to the callable 400 size only on
        // the wide menu; the test is about pricing, so pin that menu.
        let mut cfg = abs_small().cfg.clone();
        cfg.menu = crate::abstraction::MenuShape::Wide;
        let abs = Arc::new(Abstraction::new(cfg));
        let policy = Policy::new(
            Blueprint {
                strategies: Default::default(),
                iterations: 0,
                num_players: Players::TWO,
                abs_cfg: AbsConfig::default(),
                centroids: None,
            },
            abs.clone(),
        );
        let mut table = Table::new(&hand_cfg, 0, deck);
        // Check down to the river.
        for _ in 0..6 {
            table.apply_abs(AbsAction::CheckCall, &abs);
        }
        assert_eq!(table.real.street(), Street::River);
        assert_eq!(table.real.to_act(), Some(1));
        // Villain bets an OFF-TREE 800 into the 200 pot.
        table.apply_concrete(PlayerAction::RaiseTo(800), &abs);
        assert_eq!(table.real.to_act(), Some(0));
        assert_eq!(table.real.to_call(), 800, "real price is the off-tree bet");
        assert_eq!(
            table.shadow.to_call(),
            400,
            "shadow must have mapped the bet to the 200%-pot abstract size"
        );

        // Villain's tracked range: 42% a bluff hero beats (pair of twos),
        // 58% the nuts (royal). Hero (Ah 3c, pair of threes) has 42% equity.
        let mut tracker = RangeTracker::new(2);
        tracker.set_all(1, 0.0);
        let bluff = parse_cards("2c 2h").unwrap();
        let nuts = parse_cards("As Ks").unwrap();
        tracker.set_weight(1, Hole::new(bluff[0], bluff[1]).unwrap(), 0.42);
        tracker.set_weight(1, Hole::new(nuts[0], nuts[1]).unwrap(), 0.58);

        let train_cfg = TrainConfig {
            hand: hand_cfg,
            prune_after: u64::MAX,
            ..TrainConfig::default()
        };
        let params = SearchParams {
            time_ms: 10_000,
            max_iters: 20_000,
            qre_lambda: None,
            safe_resolve: false,
            adaptive: false,
            round_root: true,
        };
        // EV of calling 800 real: .42(+900) - .58(900) = -144 < fold -100.
        let mut rng = SmallRng::seed_from_u64(9);
        let a = policy.act_with_search(
            &table.real,
            &table.shadow,
            &table.hist,
            params,
            &train_cfg,
            Some(&tracker),
            None,
            None,
            &mut rng,
        );
        assert_eq!(a, AbsAction::Fold, "must fold at the real 800 price");

        // Control (old behavior): priced at the mapped 400, calling is
        // correct: .42(+500) - .58(500) = -80 > fold -100.
        let mut rng = SmallRng::seed_from_u64(9);
        let a = policy.act_with_search(
            &table.shadow,
            &table.shadow,
            &table.hist,
            params,
            &train_cfg,
            Some(&tracker),
            None,
            None,
            &mut rng,
        );
        assert_eq!(
            a,
            AbsAction::CheckCall,
            "at the mapped price the same spot is a call — pricing window sanity"
        );
    }

    /// Depth-limited, range-tracked flop resolve: must produce a trained
    /// root strategy (leaf rollouts + range sampling wired end to end).
    #[test]
    fn range_tracked_depth_limited_flop_search_smoke() {
        use crate::search::RangeTracker;
        let abs = Arc::new(abs_small());
        let hand_cfg = HandConfig {
            num_players: Players::TWO,
            ..HandConfig::default()
        };
        let train_cfg = TrainConfig {
            hand: hand_cfg.clone(),
            prune_after: u64::MAX,
            ..TrainConfig::default()
        };

        // Tiny blueprint so leaf rollouts and range updates have something
        // to look up (missing infosets fall back to uniform anyway).
        let trainer = Trainer::new(abs.clone(), train_cfg.clone());
        trainer.run(2_000, &|_| {});
        let bp = Arc::new(trainer.to_blueprint());

        // Play to the flop: button calls, BB checks.
        let mut rng = SmallRng::seed_from_u64(77);
        let mut h = Hand::new(&hand_cfg, 0, fresh_deck());
        let mut hist: Vec<Token> = Vec::new();
        let mut tracker = RangeTracker::new(2);
        for a in [AbsAction::CheckCall, AbsAction::CheckCall] {
            tracker.observe(h.to_act().unwrap(), a, &h, &hist, &bp, &abs);
            let street_before = h.street();
            h.apply(abs.concrete(&h, a));
            hist.push(a.token());
            if h.street() != street_before {
                hist.push(crate::abstraction::Token::StreetSep);
                tracker.exclude(h.board());
            }
        }
        assert_eq!(h.street(), Street::Flop);

        let solver = resolve_subgame(
            abs.clone(),
            &train_cfg,
            &h,
            &hist,
            SearchParams {
                time_ms: 5_000,
                max_iters: 3_000,
                qre_lambda: None,
                safe_resolve: false,
                adaptive: false,
                round_root: true,
            },
            Some(&tracker),
            Some(LeafCfg {
                strategies: bp.clone(),
                abs: abs.clone(),
                limit: Street::Flop,
            }),
        None,
        );
        assert!(solver.node_count() > 0, "search must create infosets");
        let p = h.to_act().unwrap();
        let bucket = solver.abs.bucket(h.hole(p), h.board(), &mut rng);
        let strat = solver
            .avg_strategy(bucket, &hist)
            .expect("root infoset must be trained");
        let acts = abs.abstract_actions(&h);
        assert_eq!(strat.len(), acts.len());
        assert!((strat.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }
}
