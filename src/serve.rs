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
//! were the training depth; see `mirror`.

use crate::bot::Policy;
use crate::cards::{parse_card, Card};
use crate::engine::{HandConfig, PlayerAction};
use crate::mirror::{Mirror, Source};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};

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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub id: u64,
    pub button: usize,
    /// Stacks at the start of the hand, before blinds, in chips (bb = 100).
    pub stacks: Vec<u32>,
    /// The seat to act; must match the replayed hand.
    pub seat: usize,
    pub hole: [String; 2],
    pub board: Vec<String>,
    pub actions: Vec<Act>,
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

/// Answer requests until stdin closes.
pub fn serve(policy: &Policy, cfg: &HandConfig) {
    let mut out = std::io::stdout().lock();
    write_line(
        &mut out,
        &serde_json::json!({
            "ready": true,
            "players": cfg.num_players,
            "infosets": policy.blueprint.strategies.len(),
        }),
    );
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Request>(&line) {
            Err(e) => Response::Error {
                id: None,
                error: format!("bad request: {e}"),
            },
            Ok(req) => match decide(policy, cfg, &req) {
                Ok(d) => Response::Strategy {
                    id: req.id,
                    strategy: d.strategy,
                    source: match d.source {
                        Source::Blueprint => "blueprint",
                        Source::Unseen => "unseen",
                        Source::Capped => "capped",
                    },
                },
                Err(error) => Response::Error {
                    id: Some(req.id),
                    error,
                },
            },
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
pub fn decide(policy: &Policy, cfg: &HandConfig, req: &Request) -> Result<Decision, String> {
    let n = cfg.num_players;
    if req.stacks.len() != n {
        return Err(format!("expected {n} stacks, got {}", req.stacks.len()));
    }
    if req.button >= n || req.seat >= n {
        return Err(format!("button and seat must be below {n}"));
    }
    if req.stacks.contains(&0) {
        return Err("stacks must be positive".into());
    }
    let hole = [card(&req.hole[0])?, card(&req.hole[1])?];
    let board = req.board.iter().map(|c| card(c)).collect::<Result<Vec<_>, _>>()?;
    if board.len() > 5 {
        return Err("the board has at most 5 cards".into());
    }
    let deck = deal(n, req.seat, hole, &board)?;
    let mut mirror = replay(policy, cfg, req, deck)?;

    let real = &mirror.real;
    if real.is_terminal() {
        return Err("the hand is over".into());
    }
    if real.to_act() != req.seat {
        return Err(format!("seat {} is to act, not seat {}", real.to_act(), req.seat));
    }
    if real.board().len() != board.len() {
        return Err(format!(
            "the board has {} cards but the hand is on the {:?}",
            board.len(),
            real.street()
        ));
    }
    let (choices, source) = mirror.strategy(policy);
    let strategy = choices
        .into_iter()
        .map(|(a, p)| Choice {
            act: match a {
                PlayerAction::Fold => Act::Fold,
                PlayerAction::CheckCall => Act::Call,
                PlayerAction::RaiseTo(to) => Act::Raise { to },
            },
            p,
        })
        .collect();
    Ok(Decision { strategy, source })
}

fn card(s: &str) -> Result<Card, String> {
    parse_card(s).ok_or_else(|| format!("bad card {s:?}"))
}

/// A deck that deals `hole` to `seat` and `board` as the board; every other
/// position gets a card that no one in the request has seen. Opponents'
/// holes never affect the acting seat's blueprint lookup.
fn deal(n: usize, seat: usize, hole: [Card; 2], board: &[Card]) -> Result<[Card; 52], String> {
    let mut used = [false; 52];
    for &c in hole.iter().chain(board) {
        if std::mem::replace(&mut used[c as usize], true) {
            return Err(format!("card {} appears twice", crate::cards::card_str(c)));
        }
    }
    let mut unseen = (0..52u8).filter(|&c| !used[c as usize]);
    let mut deck = [0u8; 52];
    for (i, slot) in deck.iter_mut().enumerate() {
        *slot = if i == 2 * seat {
            hole[0]
        } else if i == 2 * seat + 1 {
            hole[1]
        } else if (2 * n..2 * n + board.len()).contains(&i) {
            board[i - 2 * n]
        } else {
            unseen.next().expect("52 cards cover every position")
        };
    }
    Ok(deck)
}

fn replay(policy: &Policy, cfg: &HandConfig, req: &Request, deck: [Card; 52]) -> Result<Mirror, String> {
    let mut mirror = Mirror::new(cfg, req.button, deck, &req.stacks);
    for (i, &a) in req.actions.iter().enumerate() {
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
        mirror.apply(action, &policy.abs);
    }
    Ok(mirror)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::{preflop_bucket, AbsConfig, Abstraction};
    use crate::cfr::{make_key, Blueprint, StrategyMap};
    use crate::engine::Hand;
    use std::sync::Arc;

    const STACK: u32 = 10_000;

    fn policy(strategies: StrategyMap) -> Policy {
        let bp = Blueprint {
            strategies,
            iterations: 0,
            num_players: 6,
            abs_cfg: AbsConfig::default(),
            centroids: None,
        };
        let abs = Abstraction::new(bp.abs_cfg.clone());
        Policy::new(bp, Arc::new(abs))
    }

    fn request(stacks: u32, seat: usize, hole: [&str; 2], board: &[&str], actions: Vec<Act>) -> Request {
        Request {
            id: 1,
            button: 0,
            stacks: vec![stacks; 6],
            seat,
            hole: hole.map(String::from),
            board: board.iter().map(|c| c.to_string()).collect(),
            actions,
        }
    }

    fn cfg() -> HandConfig {
        HandConfig::default()
    }

    // Button 0, blinds in 1 and 2: seat 3 opens the action preflop.
    const UTG: usize = 3;

    #[test]
    fn unseen_infosets_fall_back_to_check_call() {
        let req = request(STACK, UTG, ["Ah", "Kd"], &[], vec![]);
        let d = decide(&policy(StrategyMap::default()), &cfg(), &req).unwrap();
        assert_eq!(d.source, Source::Unseen);
        assert_eq!(d.strategy, vec![Choice { act: Act::Call, p: 1.0 }]);
    }

    /// A policy whose only infoset is UTG's first decision with `hole`,
    /// weighted per abstract action by `weights(menu length)`.
    fn utg_policy(hole: [&str; 2], weights: impl FnOnce(usize) -> Vec<f32>) -> Policy {
        let cards = hole.map(|c| parse_card(c).unwrap());
        let h = Hand::new(&cfg(), 0, deal(6, UTG, cards, &[]).unwrap());
        let menu = policy(StrategyMap::default()).abs.abstract_actions(&h).len();
        let mut map = StrategyMap::default();
        map.insert(make_key(preflop_bucket(cards), &[]).to_vec(), weights(menu));
        policy(map)
    }

    /// Everything on the last abstract action, which is always all-in.
    fn all_in(menu: usize) -> Vec<f32> {
        let mut s = vec![0.0; menu];
        s[menu - 1] = 1.0;
        s
    }

    #[test]
    fn blueprint_mix_comes_back_as_concrete_actions() {
        let p = utg_policy(["Ah", "Kd"], |menu| {
            let mut s = vec![0.0; menu];
            s[0] = 0.5; // fold
            s[menu - 1] = 0.5;
            s
        });
        let d = decide(&p, &cfg(), &request(STACK, UTG, ["Ah", "Kd"], &[], vec![])).unwrap();
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
        let p = utg_policy(["Ah", "Kd"], all_in);
        let d = decide(&p, &cfg(), &request(3 * STACK, UTG, ["Ah", "Kd"], &[], vec![])).unwrap();
        assert_eq!(d.source, Source::Blueprint, "a deep stack still finds the training-depth infoset");
        assert_eq!(d.strategy, vec![Choice { act: Act::Raise { to: STACK }, p: 1.0 }]);
    }

    #[test]
    fn short_stacks_shove_their_real_stack() {
        let p = utg_policy(["Ah", "Kd"], all_in);
        let d = decide(&p, &cfg(), &request(STACK / 2, UTG, ["Ah", "Kd"], &[], vec![])).unwrap();
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
        let d = decide(&policy(StrategyMap::default()), &cfg(), &req).unwrap();
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
        req.stacks[UTG] = STACK / 5;
        let d = decide(&policy(StrategyMap::default()), &cfg(), &req).unwrap();
        assert_eq!(d.source, Source::Unseen, "looked up on-tree, not capped");
    }

    #[test]
    fn replays_actions_across_streets() {
        // UTG..BTN fold, SB completes, BB checks: the flop goes to SB first.
        let mut actions = vec![Act::Fold; 4];
        actions.extend([Act::Call, Act::Call]);
        let req = request(STACK, 1, ["7c", "7d"], &["2h", "9s", "Kc"], actions);
        let d = decide(&policy(StrategyMap::default()), &cfg(), &req).unwrap();
        assert_eq!(d.strategy, vec![Choice { act: Act::Call, p: 1.0 }]);
    }

    #[test]
    fn off_menu_raises_are_accepted_and_followed() {
        let req = request(STACK, 4, ["Qs", "Js"], &[], vec![Act::Raise { to: 333 }]);
        assert!(decide(&policy(StrategyMap::default()), &cfg(), &req).is_ok());
    }

    #[test]
    fn rejects_requests_that_do_not_match_the_hand() {
        let p = policy(StrategyMap::default());
        let err = |req: Request| decide(&p, &cfg(), &req).unwrap_err();
        assert!(err(request(STACK, 4, ["Ah", "Kd"], &[], vec![])).contains("seat 3 is to act"));
        assert!(err(request(STACK, UTG, ["Ah", "Ah"], &[], vec![])).contains("twice"));
        assert!(err(request(STACK, UTG, ["Ah", "Kd"], &["2c", "3c", "4c"], vec![])).contains("board"));
        assert!(err(request(STACK, UTG, ["Ah", "Xd"], &[], vec![])).contains("bad card"));
        assert!(err(request(STACK, UTG, ["Ah", "Kd"], &[], vec![Act::Raise { to: 150 }])).contains("outside"));
        let mut short = request(STACK, UTG, ["Ah", "Kd"], &[], vec![]);
        short.stacks.pop();
        assert!(err(short).contains("expected 6 stacks"));
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
        let json = serde_json::to_string(&Choice { act: Act::Raise { to: 250 }, p: 0.5 }).unwrap();
        assert_eq!(json, r#"{"type":"raise","to":250,"p":0.5}"#);
    }
}
