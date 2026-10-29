//! Machine interface for embedding the bot in another program: one JSON
//! request per stdin line, one JSON response per stdout line. Diagnostics
//! go to stderr so stdout carries nothing but the protocol.
//!
//! Requests are stateless. Each describes the hand so far — starting
//! stacks, button, the acting seat's hole cards, the board and every
//! concrete action — and the table is replayed from scratch, so the
//! process keeps no per-table state and a restart loses nothing. The
//! response is the blueprint's mixed strategy as concrete actions; the
//! caller samples it with its own RNG. Every stack depth is played as if it
//! were the training depth; see `mirror`. With blueprints trained at several
//! depths, each request uses the one nearest the acting seat's effective
//! stack (`pick`). With `--search`, postflop decisions are re-solved online
//! (`Policy::search_dist`) from ranges rebuilt by replaying the request.

use crate::bot::{Policy, SearchParams};
use crate::cards::{card_names, hole_cards, Card, Deck, Hole};
use crate::cfr::TrainConfig;
use crate::engine::{HandConfig, PlayerAction, Players, Stack, Street};
use crate::mirror::{Bot, Mirror, Source};
use crate::search::RangeTracker;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};

/// Online search for one served blueprint: the policy shares the bot's
/// training-format blueprint.
pub struct Searcher {
    pub policy: Policy,
    pub params: SearchParams,
    pub train_cfg: TrainConfig,
}

/// One blueprint the server answers from, at its training depth.
pub struct Served {
    pub bot: Bot,
    pub cfg: HandConfig,
    pub search: Option<Searcher>,
}

/// A concrete action, as the caller's table saw or will apply it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Act {
    Fold,
    /// Check when nothing is owed, otherwise call.
    Call,
    /// Raise TO this total street commitment.
    Raise { to: u32 },
}

/// A request as it arrives. Cards are parsed while deserializing, so a
/// `Request` holds only real cards; `Spot::new` checks it against the table.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub id: u64,
    pub button: usize,
    /// Stacks at the start of the hand, before blinds, in chips (bb = 100).
    pub stacks: Vec<Stack>,
    /// The seat to act; must match the replayed hand.
    pub seat: usize,
    #[serde(deserialize_with = "hole_cards")]
    pub hole: Hole,
    #[serde(deserialize_with = "card_names")]
    pub board: Vec<Card>,
    pub actions: Vec<Act>,
}

/// A request checked against the table it is played at: one stack per
/// seat, the button and the acting seat on the table, no card
/// twice, and a board of 0, 3, 4 or 5 cards. Only `Spot::new` makes one,
/// so the decision code can rely on all of that; whether the actions are
/// legal is found by replaying them.
#[derive(Debug)]
pub struct Spot {
    id: u64,
    button: usize,
    stacks: Vec<Stack>,
    seat: usize,
    hole: Hole,
    board: Vec<Card>,
    actions: Vec<Act>,
}

impl Spot {
    pub fn new(req: Request, players: Players) -> Result<Spot, String> {
        let players = players.get();
        if req.stacks.len() != players {
            return Err(format!("expected {players} stacks, got {}", req.stacks.len()));
        }
        if req.button >= players || req.seat >= players {
            return Err(format!("button and seat must be below {players}"));
        }
        if ![0, 3, 4, 5].contains(&req.board.len()) {
            return Err(format!("a board has 0, 3, 4 or 5 cards, not {}", req.board.len()));
        }
        let mut used = [false; 52];
        for &c in req.hole.iter().chain(&req.board) {
            if std::mem::replace(&mut used[c.index()], true) {
                return Err(format!("card {c} appears twice"));
            }
        }
        Ok(Spot {
            id: req.id,
            button: req.button,
            stacks: req.stacks,
            seat: req.seat,
            hole: req.hole,
            board: req.board,
            actions: req.actions,
        })
    }

    /// A deck that deals the hole cards to the acting seat and the board as
    /// the board; every other position gets a card no one in the request
    /// has seen. Opponents' holes never affect the acting seat's blueprint
    /// lookup.
    fn deck(&self) -> Deck {
        let n = self.stacks.len();
        let hole = (2 * self.seat..).zip(self.hole);
        let board = (2 * n..).zip(self.board.iter().copied());
        Deck::place(hole.chain(board)).expect("`Spot::new` refuses a card given twice")
    }
}

/// Read one request line into a `Spot`. The id is read first, so every
/// error about a request that carries an id echoes it.
fn read_request(line: &str, players: Players) -> Result<Spot, (Option<u64>, String)> {
    let value: serde_json::Value =
        serde_json::from_str(line).map_err(|e| (None, format!("bad request: {e}")))?;
    let id = value.get("id").and_then(serde_json::Value::as_u64);
    let req = Request::deserialize(value).map_err(|e| (id, format!("bad request: {e}")))?;
    Spot::new(req, players).map_err(|e| (id, e))
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Choice {
    #[serde(flatten)]
    pub act: Act,
    pub p: f64,
}

#[derive(Debug, PartialEq)]
pub struct Decision {
    pub strategy: Vec<Choice>,
    pub source: Source,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Response {
    Strategy {
        id: u64,
        strategy: Vec<Choice>,
        source: &'static str,
    },
    Error {
        id: Option<u64>,
        error: String,
    },
}

/// The blueprint for this request: the one whose training depth is nearest
/// the acting seat's effective stack (its own stack, capped by the deepest
/// opponent's), ties going to the shallower. Starting stacks are fixed for
/// the hand, so a seat keeps one blueprint from its first decision to its
/// last.
pub fn pick(served: &[Served], spot: &Spot) -> usize {
    let deepest_other = spot
        .stacks
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != spot.seat)
        .map(|(_, &s)| s.get())
        .max()
        .unwrap_or(0);
    let eff = spot.stacks[spot.seat].get().min(deepest_other);
    (0..served.len())
        .min_by_key(|&i| (served[i].cfg.stack.get().abs_diff(eff), served[i].cfg.stack))
        .expect("serve has at least one blueprint")
}

/// Answer requests until stdin closes.
pub fn serve(served: &[Served]) {
    let mut out = std::io::stdout().lock();
    write_line(
        &mut out,
        &serde_json::json!({
            "ready": true,
            "players": served[0].cfg.num_players,
            "infosets": served.iter().map(|s| s.bot.strategies.len()).sum::<usize>(),
            "depths": served.iter().map(|s| s.cfg.stack).collect::<Vec<_>>(),
            "search": served.iter().map(|s| s.search.is_some()).collect::<Vec<_>>(),
        }),
    );
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let response = match read_request(&line, served[0].cfg.num_players) {
            Err((id, error)) => Response::Error { id, error },
            Ok(spot) => {
                let s = &served[pick(served, &spot)];
                let decision = match &s.search {
                    Some(searcher) => decide_with_search(&s.bot, &s.cfg, searcher, &spot),
                    None => decide(&s.bot, &s.cfg, &spot),
                };
                match decision {
                    Ok(d) => Response::Strategy {
                        id: spot.id,
                        strategy: d.strategy,
                        source: match d.source {
                            Source::Blueprint => "blueprint",
                            Source::Unseen => "unseen",
                            Source::Capped => "capped",
                            Source::Search => "search",
                        },
                    },
                    Err(error) => Response::Error {
                        id: Some(spot.id),
                        error,
                    },
                }
            }
        };
        write_line(&mut out, &response);
    }
}

fn write_line(out: &mut impl Write, value: &impl Serialize) {
    let line = serde_json::to_string(value).expect("responses always serialize");
    // A closed stdout means the parent is gone; nothing left to answer.
    if writeln!(out, "{line}").and_then(|_| out.flush()).is_err() {
        std::process::exit(0);
    }
}

/// The blueprint strategy for the seat to act in the described hand.
pub fn decide(bot: &Bot, cfg: &HandConfig, spot: &Spot) -> Result<Decision, String> {
    let mut mirror = replay(bot, cfg, spot, false)?;
    let (choices, source) = mirror.strategy(bot);
    Ok(Decision { strategy: choices_json(choices), source })
}

/// Postflop, and preflop facing an all-in that a call would close, the
/// searched strategy for the seat to act: the ranges come from replaying
/// every action of the request through the blueprint, and the solve is
/// rooted at the current decision (stateless requests carry no turn solve
/// into the river). Other preflop decisions, the blueprint's.
pub fn decide_with_search(bot: &Bot, cfg: &HandConfig, s: &Searcher, spot: &Spot) -> Result<Decision, String> {
    let mut mirror = replay(bot, cfg, spot, true)?;
    if mirror.real.street() == Street::Preflop && !crate::bot::preflop_allin_spot(&mirror.real) {
        let (choices, source) = mirror.strategy(bot);
        return Ok(Decision { strategy: choices_json(choices), source });
    }
    if !mirror.sync(&bot.abs) {
        let (choices, source) = mirror.strategy(bot);
        return Ok(Decision { strategy: choices_json(choices), source });
    }
    let policy = &s.policy;
    let tracker = ranges(policy, &mirror, cfg.num_players.get());
    let mut rng = SmallRng::seed_from_u64(spot.id ^ 0x5EA2_C4ED);
    let (acts, probs) = policy.search_dist(
        &mirror.real,
        mirror.shadow(),
        mirror.hist(),
        s.params,
        &s.train_cfg,
        Some(&tracker),
        None,
        None,
        &mut rng,
    );
    let total: f64 = probs.iter().sum();
    let probs: Vec<f64> = probs.iter().map(|p| p / total).collect();
    let choices = mirror.concrete(&policy.abs, &acts, &probs);
    Ok(Decision { strategy: choices_json(choices), source: Source::Search })
}

/// Every seat's range after the request's actions, observed through the
/// blueprint. Widened: the table's other seats include a human, who is not
/// known to play the blueprint, and hard Bayes collapses a human's off-model
/// line onto a few combos that the exact river solve then answers to blocker
/// by blocker. The bot seats are widened too until the protocol says which
/// seats are bots. The near-hard weights the preflop all-in resolve reads
/// are never widened.
fn ranges(policy: &Policy, mirror: &Mirror, n: usize) -> RangeTracker {
    let mut tracker = RangeTracker::new(n).with_widening();
    let mut board = 0;
    for o in mirror.log() {
        if o.shadow.board().len() > board {
            board = o.shadow.board().len();
            tracker.exclude(o.shadow.board());
        }
        tracker.observe(o.seat, o.action, &o.shadow, &o.hist, &*policy.strategies, &policy.abs);
    }
    tracker.exclude(mirror.real.board());
    tracker
}

fn choices_json(choices: Vec<(PlayerAction, f64)>) -> Vec<Choice> {
    choices
        .into_iter()
        .map(|(a, p)| Choice {
            act: match a {
                PlayerAction::Fold => Act::Fold,
                PlayerAction::CheckCall => Act::Call,
                PlayerAction::RaiseTo(to) => Act::Raise { to },
            },
            p,
        })
        .collect()
}

/// Replay the spot's actions and check that the result is the decision it
/// asks about: the hand is live, its seat is to act, and its board is the
/// board of the street the actions reach.
fn replay(bot: &Bot, cfg: &HandConfig, spot: &Spot, log: bool) -> Result<Mirror, String> {
    debug_assert_eq!(spot.stacks.len(), cfg.num_players.get(), "a spot built for another table");
    let mut mirror = Mirror::new(cfg, spot.button, spot.deck(), &spot.stacks);
    if log {
        mirror = mirror.with_log();
    }
    for (i, &a) in spot.actions.iter().enumerate() {
        let h = &mirror.real;
        if h.is_terminal() {
            return Err(format!("action {i} comes after the hand ended"));
        }
        let action = match a {
            Act::Fold if h.can_check() => {
                return Err(format!("action {i} folds with nothing to call"));
            }
            Act::Fold => PlayerAction::Fold,
            Act::Call => PlayerAction::CheckCall,
            Act::Raise { to } => match h.raise_bounds() {
                Some((lo, hi)) if (lo..=hi).contains(&to) => PlayerAction::RaiseTo(to),
                Some((lo, hi)) => {
                    return Err(format!("action {i} raises to {to}, outside {lo}..={hi}"));
                }
                None => return Err(format!("action {i} raises, but no raise is legal")),
            },
        };
        mirror.apply(action, &bot.abs);
    }

    let real = &mirror.real;
    let Some(actor) = real.to_act() else {
        return Err("the hand is over".into());
    };
    if actor != spot.seat {
        return Err(format!("seat {actor} is to act, not seat {}", spot.seat));
    }
    if real.board().len() != spot.board.len() {
        return Err(format!(
            "the board has {} cards but the hand is on the {:?}",
            spot.board.len(),
            real.street()
        ));
    }
    Ok(mirror)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::{preflop_bucket, AbsAction, AbsConfig, Abstraction};
    use crate::cfr::{make_key, Blueprint, StrategyMap};
    use crate::engine::Hand;
    use std::sync::Arc;

    const STACK: u32 = 10_000;

    fn bot(strategies: StrategyMap) -> Bot {
        let bp = Blueprint {
            strategies,
            iterations: 0,
            num_players: Players::SIX,
            abs_cfg: AbsConfig::default(),
            centroids: None,
        };
        Bot {
            abs: Arc::new(Abstraction::new(bp.abs_cfg.clone())),
            strategies: Box::new(bp),
        }
    }

    fn request(stacks: u32, seat: usize, hole: [&str; 2], board: &[&str], actions: Vec<Act>) -> Request {
        Request {
            id: 1,
            button: 0,
            stacks: vec![Stack::new(stacks).unwrap(); 6],
            seat,
            hole: Hole::try_from(hole.map(|c| c.parse().unwrap())).unwrap(),
            board: board.iter().map(|c| c.parse().unwrap()).collect(),
            actions,
        }
    }

    fn spot(req: Request) -> Spot {
        Spot::new(req, Players::SIX).unwrap()
    }

    fn cfg() -> HandConfig {
        HandConfig::default()
    }

    // Button 0, blinds in 1 and 2: seat 3 opens the action preflop.
    const UTG: usize = 3;

    #[test]
    fn unseen_infosets_fall_back_to_check_call() {
        let req = request(STACK, UTG, ["Ah", "Kd"], &[], vec![]);
        let d = decide(&bot(StrategyMap::default()), &cfg(), &spot(req)).unwrap();
        assert_eq!(d.source, Source::Unseen);
        assert_eq!(d.strategy, vec![Choice { act: Act::Call, p: 1.0 }]);
    }

    /// A bot whose only infoset is UTG's first decision with `hole`,
    /// weighted per abstract action by `weights(menu length)`.
    fn utg_bot(hole: [&str; 2], weights: impl FnOnce(usize) -> Vec<f32>) -> Bot {
        let s = spot(request(STACK, UTG, hole, &[], vec![]));
        let cards = s.hole;
        let h = Hand::new(&cfg(), 0, s.deck());
        let menu = bot(StrategyMap::default()).abs.abstract_actions(&h).len();
        let mut map = StrategyMap::default();
        map.insert(make_key(preflop_bucket(*cards), &[]).to_vec(), weights(menu));
        bot(map)
    }

    /// Everything on the last abstract action, which is always all-in.
    fn all_in(menu: usize) -> Vec<f32> {
        let mut s = vec![0.0; menu];
        s[menu - 1] = 1.0;
        s
    }

    #[test]
    fn blueprint_mix_comes_back_as_concrete_actions() {
        let p = utg_bot(["Ah", "Kd"], |menu| {
            let mut s = vec![0.0; menu];
            s[0] = 0.5; // fold
            s[menu - 1] = 0.5;
            s
        });
        let d = decide(&p, &cfg(), &spot(request(STACK, UTG, ["Ah", "Kd"], &[], vec![]))).unwrap();
        assert_eq!(d.source, Source::Blueprint);
        assert_eq!(
            d.strategy,
            vec![
                Choice { act: Act::Fold, p: 0.5 },
                Choice { act: Act::Raise { to: STACK }, p: 0.5 },
            ]
        );
    }

    #[test]
    fn deep_stacks_shove_only_the_training_stack() {
        let p = utg_bot(["Ah", "Kd"], all_in);
        let d = decide(&p, &cfg(), &spot(request(3 * STACK, UTG, ["Ah", "Kd"], &[], vec![]))).unwrap();
        assert_eq!(d.source, Source::Blueprint, "a deep stack still finds the training-depth infoset");
        assert_eq!(d.strategy, vec![Choice { act: Act::Raise { to: STACK }, p: 1.0 }]);
    }

    #[test]
    fn short_stacks_shove_their_real_stack() {
        let p = utg_bot(["Ah", "Kd"], all_in);
        let d = decide(&p, &cfg(), &spot(request(STACK / 2, UTG, ["Ah", "Kd"], &[], vec![]))).unwrap();
        assert_eq!(d.source, Source::Blueprint, "a short stack still finds the training-depth infoset");
        assert_eq!(d.strategy, vec![Choice { act: Act::Raise { to: STACK / 2 }, p: 1.0 }]);
    }

    #[test]
    fn deep_stacks_check_call_once_a_training_stack_is_in() {
        // At 300bb, UTG raises to 100bb and the BB calls: both are all-in in
        // the 100bb shadow, so on the flop the BB is capped.
        let mut actions = vec![Act::Raise { to: STACK }];
        actions.extend([Act::Fold; 4]);
        actions.push(Act::Call);
        let req = request(3 * STACK, 2, ["7c", "7d"], &["2h", "9s", "Kc"], actions);
        let d = decide(&bot(StrategyMap::default()), &cfg(), &spot(req)).unwrap();
        assert_eq!(d.source, Source::Capped);
        assert_eq!(d.strategy, vec![Choice { act: Act::Call, p: 1.0 }]);
    }

    #[test]
    fn short_all_ins_keep_callers_on_tree() {
        // UTG shoves 20bb, seats 4 and 5 call. On the flop UTG is all-in in
        // the real hand only; the shadow checks for it and seat 4 acts.
        let mut actions = vec![Act::Raise { to: STACK / 5 }, Act::Call, Act::Call];
        actions.extend([Act::Fold; 3]);
        let mut req = request(STACK, 4, ["7c", "7d"], &["2h", "9s", "Kc"], actions);
        req.stacks[UTG] = Stack::new(STACK / 5).unwrap();
        let d = decide(&bot(StrategyMap::default()), &cfg(), &spot(req)).unwrap();
        assert_eq!(d.source, Source::Unseen, "looked up on-tree, not capped");
    }

    #[test]
    fn replays_actions_across_streets() {
        // UTG..BTN fold, SB completes, BB checks: the flop goes to SB first.
        let mut actions = vec![Act::Fold; 4];
        actions.extend([Act::Call, Act::Call]);
        let req = request(STACK, 1, ["7c", "7d"], &["2h", "9s", "Kc"], actions);
        let d = decide(&bot(StrategyMap::default()), &cfg(), &spot(req)).unwrap();
        assert_eq!(d.strategy, vec![Choice { act: Act::Call, p: 1.0 }]);
    }

    #[test]
    fn off_menu_raises_are_accepted_and_followed() {
        let req = request(STACK, 4, ["Qs", "Js"], &[], vec![Act::Raise { to: 333 }]);
        assert!(decide(&bot(StrategyMap::default()), &cfg(), &spot(req)).is_ok());
    }

    #[test]
    fn rejects_requests_that_do_not_match_the_hand() {
        let p = bot(StrategyMap::default());
        let err = |req: Request| Spot::new(req, Players::SIX).and_then(|s| decide(&p, &cfg(), &s)).unwrap_err();
        assert!(err(request(STACK, 4, ["Ah", "Kd"], &[], vec![])).contains("seat 3 is to act"));
        assert!(err(request(STACK, UTG, ["Ah", "Kd"], &["Ah", "3c", "4c"], vec![])).contains("twice"));
        assert!(err(request(STACK, UTG, ["Ah", "Kd"], &["2c", "3c", "4c"], vec![])).contains("board"));
        assert!(err(request(STACK, UTG, ["Ah", "Kd"], &["2c", "3c"], vec![])).contains("0, 3, 4 or 5"));
        assert!(err(request(STACK, UTG, ["Ah", "Kd"], &[], vec![Act::Raise { to: 150 }])).contains("outside"));
        let mut short = request(STACK, UTG, ["Ah", "Kd"], &[], vec![]);
        short.stacks.pop();
        assert!(err(short).contains("expected 6 stacks"));
    }

    #[test]
    fn picks_the_blueprint_nearest_the_effective_stack() {
        let depths = [10_000, 8_000];
        let served: Vec<Served> = depths
            .iter()
            .map(|&stack| Served {
                bot: bot(StrategyMap::default()),
                cfg: HandConfig { stack: Stack::new(stack).unwrap(), ..cfg() },
                search: None,
            })
            .collect();
        let at = |stacks: Vec<u32>, seat: usize| {
            let mut req = request(STACK, seat, ["Ah", "Kd"], &[], vec![]);
            req.stacks = stacks.into_iter().map(|s| Stack::new(s).unwrap()).collect();
            depths[pick(&served, &spot(req))]
        };
        assert_eq!(at(vec![10_000; 6], UTG), 10_000);
        assert_eq!(at(vec![8_000; 6], UTG), 8_000);
        assert_eq!(at(vec![9_000; 6], UTG), 8_000, "a tie goes to the shallower");
        assert_eq!(at(vec![9_100; 6], UTG), 10_000);
        assert_eq!(at(vec![12_000; 6], UTG), 10_000);
        assert_eq!(at(vec![5_000; 6], UTG), 8_000);
        // A deep seat facing only 80bb opponents plays 80bb, and the short
        // seat plays its own stack.
        assert_eq!(at(vec![8_000, 8_000, 8_000, 12_000, 8_000, 8_000], UTG), 8_000);
        let mut stacks = vec![12_000; 6];
        stacks[UTG] = 8_200;
        assert_eq!(at(stacks, UTG), 8_000);
    }

    /// A searcher over an empty blueprint, with a budget small enough for
    /// a unit test.
    fn searcher() -> (Bot, Searcher) {
        let bp = Blueprint {
            strategies: StrategyMap::default(),
            iterations: 0,
            num_players: Players::SIX,
            abs_cfg: AbsConfig::default(),
            centroids: None,
        };
        let abs = Arc::new(Abstraction::new(bp.abs_cfg.clone()));
        let policy = Policy::new(bp, abs.clone());
        let bot = Bot { abs, strategies: Box::new(policy.blueprint.clone()) };
        let params = SearchParams { time_ms: 50, max_iters: 300, ..SearchParams::default() };
        let train_cfg = TrainConfig { hand: cfg(), prune_after: u64::MAX, ..TrainConfig::default() };
        (bot, Searcher { policy, params, train_cfg })
    }

    #[test]
    fn search_answers_postflop_with_legal_actions() {
        let (b, s) = searcher();
        // UTG opens, two callers, everyone else folds: a 3-way flop.
        let mut actions = vec![Act::Raise { to: 290 }, Act::Call, Act::Call];
        actions.extend([Act::Fold; 3]);
        let req = request(STACK, UTG, ["Ah", "Kd"], &["2h", "9s", "Kc"], actions);
        let d = decide_with_search(&b, &cfg(), &s, &spot(req)).unwrap();
        assert_eq!(d.source, Source::Search);
        let total: f64 = d.strategy.iter().map(|c| c.p).sum();
        assert!((total - 1.0).abs() < 1e-9, "probabilities sum to {total}");
        for c in &d.strategy {
            if let Act::Raise { to } = c.act {
                assert!((1..=STACK).contains(&to), "raise to {to}");
            }
            assert_ne!(c.act, Act::Fold, "nothing to call, so no fold");
        }
    }

    #[test]
    fn search_leaves_preflop_to_the_blueprint() {
        let (b, s) = searcher();
        let req = request(STACK, UTG, ["Ah", "Kd"], &[], vec![]);
        let d = decide_with_search(&b, &cfg(), &s, &spot(req)).unwrap();
        assert_eq!(d.source, Source::Unseen);
    }

    #[test]
    fn facing_a_preflop_jam_is_resolved_by_equity() {
        let (b, s) = searcher();
        // UTG opens, MP jams 100bb, folds to the big blind. The empty
        // blueprint leaves MP's range uniform, so AA calls and 72o folds.
        let mut actions = vec![Act::Raise { to: 290 }, Act::Raise { to: STACK }];
        actions.extend([Act::Fold; 3]);
        let call = |hole: [&str; 2]| {
            let d = decide_with_search(&b, &cfg(), &s, &spot(request(STACK, 2, hole, &[], actions.clone()))).unwrap();
            assert_eq!(d.source, Source::Search);
            d.strategy.iter().filter(|c| c.act == Act::Call).map(|c| c.p).sum::<f64>()
        };
        assert!(call(["Ah", "As"]) > 0.99);
        assert!(call(["7h", "2d"]) < 0.01);
    }

    #[test]
    fn the_replay_log_has_one_entry_per_shadow_action() {
        let (b, _) = searcher();
        let mut actions = vec![Act::Raise { to: 290 }, Act::Call, Act::Call];
        actions.extend([Act::Fold; 3]);
        let req = request(STACK, UTG, ["Ah", "Kd"], &["2h", "9s", "Kc"], actions);
        let m = replay(&b, &cfg(), &spot(req), true).unwrap();
        let tokens = m.hist().iter().filter(|&&t| t != crate::abstraction::Token::StreetSep).count();
        assert_eq!(m.log().len(), tokens);
        assert_eq!(m.log()[0].seat, UTG);
        assert!(m.log().iter().all(|o| o.shadow.street() == crate::engine::Street::Preflop));
    }

    /// A Pluton hand where v0.3.0's river answers swung on single blockers:
    /// the big blind (seat 0, the human) raises a limp, check-raises the
    /// flop and bets the pot on the river into 88. With a blueprint where
    /// only aces make that preflop raise, hard Bayes would favour AhAd over
    /// any other combo by the floor ratio of 50; serve's ranges must widen
    /// instead.
    #[test]
    fn a_human_off_model_line_widens_its_range() {
        let line = r#"{"id":1,"button":4,"stacks":[10000,10000,10000,10000,10000,10000],"seat":1,"hole":["8h","8c"],"board":["2c","As","6c","Ac","9d"],"actions":[{"type":"call"},{"type":"fold"},{"type":"fold"},{"type":"fold"},{"type":"fold"},{"type":"raise","to":300},{"type":"call"},{"type":"call"},{"type":"raise","to":600},{"type":"raise","to":2450},{"type":"call"},{"type":"call"},{"type":"call"},{"type":"raise","to":5550}]}"#;
        let spot = read_request(line, Players::SIX).unwrap();
        let (b, _) = searcher();
        let m = replay(&b, &cfg(), &spot, true).unwrap();
        let raise = m
            .log()
            .iter()
            .find(|o| o.seat == 0 && o.shadow.street() == Street::Preflop)
            .unwrap();
        assert!(matches!(raise.action, AbsAction::Bet(_) | AbsAction::AllIn), "{:?}", raise.action);

        let acts = b.abs.abstract_actions(&raise.shadow);
        let raise_idx = acts.iter().position(|&a| a == raise.action).unwrap();
        let check_idx = acts.iter().position(|&a| a == AbsAction::CheckCall).unwrap();
        let hole = |s: &str| Hole::try_from([&s[..2], &s[2..]].map(|c| c.parse().unwrap())).unwrap();
        let aces = preflop_bucket(*hole("AhAd"));
        let mut strategies = StrategyMap::default();
        for bucket in 0..169u16 {
            let mut s = vec![0.0f32; acts.len()];
            s[if bucket == aces { raise_idx } else { check_idx }] = 1.0;
            strategies.insert(make_key(bucket, &raise.hist).to_vec(), s);
        }
        let bp = Blueprint {
            strategies,
            iterations: 1,
            num_players: Players::SIX,
            abs_cfg: AbsConfig::default(),
            centroids: None,
        };
        let policy = Policy::new(bp, b.abs.clone());

        let t = ranges(&policy, &m, 6);
        let ratio = t.prob(0, hole("AhAd")) / t.prob(0, hole("KdQd"));
        assert!(ratio > 1.5, "aces must still be favoured, got {ratio}");
        assert!(ratio < 15.0, "the off-model raise must widen, not reach hard Bayes' 50: {ratio}");
        // Cards on the board leave the range.
        assert_eq!(t.prob(0, hole("AsKd")), 0.0);
    }

    #[test]
    fn requests_and_choices_use_the_documented_json() {
        let req: Request = serde_json::from_str(
            r#"{"id":7,"button":0,"stacks":[10000,10000,10000,10000,10000,10000],
                "seat":4,"hole":["Ah","Kd"],"board":[],
                "actions":[{"type":"raise","to":250}]}"#,
        )
        .unwrap();
        assert_eq!(req.actions, vec![Act::Raise { to: 250 }]);
        assert_eq!(req.hole.map(|c| c.to_string()), ["Ah".to_string(), "Kd".to_string()]);
        let json = serde_json::to_string(&Choice { act: Act::Raise { to: 250 }, p: 0.5 }).unwrap();
        assert_eq!(json, r#"{"type":"raise","to":250,"p":0.5}"#);
    }

    #[test]
    fn malformed_requests_are_answered_with_their_id() {
        let line = |hole: &str| {
            format!(
                r#"{{"id":7,"button":0,"stacks":[10000,10000,10000,10000,10000,10000],
                    "seat":3,"hole":{hole},"board":[],"actions":[]}}"#
            )
        };
        let (id, e) = read_request(&line(r#"["Ah","Xd"]"#), Players::SIX).unwrap_err();
        assert_eq!(id, Some(7));
        assert!(e.contains("bad card"), "{e}");
        let (id, e) = read_request(&line(r#"["Ah"]"#), Players::SIX).unwrap_err();
        assert_eq!(id, Some(7));
        assert!(e.contains("two hole cards"), "{e}");
        let (id, e) = read_request(&line(r#"["Ah","Ah"]"#), Players::SIX).unwrap_err();
        assert_eq!(id, Some(7));
        assert!(e.contains("card Ah appears twice"), "{e}");
        let (id, e) = read_request(&line(r#"["Ah","Kd"]"#), Players::new(5).unwrap()).unwrap_err();
        assert_eq!(id, Some(7));
        assert!(e.contains("expected 5 stacks"), "{e}");
        let empty_seat = line(r#"["Ah","Kd"]"#).replacen("10000", "0", 1);
        let (id, e) = read_request(&empty_seat, Players::SIX).unwrap_err();
        assert_eq!(id, Some(7));
        assert!(e.contains("at least one chip"), "{e}");
        assert_eq!(read_request("not json", Players::SIX).unwrap_err().0, None);
        assert!(read_request(&line(r#"["Ah","Kd"]"#), Players::SIX).is_ok());
    }
}
