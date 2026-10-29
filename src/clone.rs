//! Behavioral clone of Slumbot from logged hands (`slumbot --log`).
//!
//! Each JSONL line is a hand's final API response: the cumulative action
//! string, full board, both players' hole cards (Slumbot reveals its cards
//! every hand), and `client_pos`. Replaying the hand through the table
//! reconstructs the exact abstract infoset at each of Slumbot's decisions;
//! counting its chosen abstract actions per infoset and normalizing yields
//! a Blueprint-format strategy — a playable, exploitable model of the
//! static opponent (the target for restricted Nash response training).
//!
//! The abstraction must come from the blueprint the exploiter will use, so
//! buckets (and serialized centroids) line up at train and play time.

use crate::abstraction::Abstraction;
use crate::cards::{hole_cards, Card, Deck, Hole};
use crate::cfr::{make_key, Blueprint, StrategyMap};
use crate::engine::{HandConfig, PlayerAction, Players};
use crate::slumbot::{board, parse_incr, ClientPos, SlumAct, STACK};
use crate::table::Table;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde::Deserialize;
use std::io::BufRead;

/// One of Slumbot's decisions: infoset key, chosen menu index, menu size.
struct Decision {
    key: Box<[u8]>,
    act_ix: usize,
    menu_len: usize,
}

#[derive(Default)]
pub struct CloneStats {
    pub hands: u64,
    pub skipped: u64,
    pub decisions: u64,
    pub infosets: usize,
    /// Held-out top-1 agreement per street [pre, flop, turn, river]:
    /// (matches, total).
    pub agree: [(u64, u64); 4],
}

/// One line of a `slumbot --log` file: a finished hand with both
/// players' cards. Lines that do not read as one are skipped.
#[derive(Debug, Deserialize)]
struct LoggedHand {
    client_pos: ClientPos,
    #[serde(rename = "hole_cards", deserialize_with = "hole_cards")]
    our: Hole,
    #[serde(rename = "bot_hole_cards", deserialize_with = "hole_cards")]
    bot: Hole,
    #[serde(default, deserialize_with = "board")]
    board: Vec<Card>,
    #[serde(default)]
    action: String,
}

/// Replay one logged hand; returns Slumbot's decisions as (key, action).
fn replay_hand(hand: &LoggedHand, abs: &Abstraction) -> Result<Vec<Decision>, String> {
    let LoggedHand { client_pos, our, bot, board, action } = hand;
    let (acts, _) = parse_incr(action, 0)?;

    let cfg = HandConfig::new(Players::TWO, STACK);
    // Seat 0 = us, seat 1 = Slumbot (mirrors the live adapter); the
    // logged board sits where the engine deals it from.
    let deck = Deck::place(our.iter().chain(bot.iter()).chain(board).copied().enumerate())?;
    let mut table = Table::new(&cfg, client_pos.button(), deck);

    let mut rng = SmallRng::seed_from_u64(0); // bucket draws are key-seeded
    let mut out = Vec::new();
    for inc in acts {
        if matches!(inc, SlumAct::StreetSep) {
            continue; // the engine advances streets itself
        }
        let Some(p) = table.real.to_act() else {
            return Err("action after terminal".into());
        };
        let concrete = match inc {
            SlumAct::Check | SlumAct::Call => PlayerAction::CheckCall,
            SlumAct::Fold => PlayerAction::Fold,
            SlumAct::BetTo(x) => PlayerAction::RaiseTo(x),
            SlumAct::StreetSep => unreachable!(),
        };
        if p == 1 {
            let abs_a = table.map_concrete(concrete, abs);
            let menu = abs.abstract_actions(&table.shadow);
            let act_ix = menu
                .iter()
                .position(|&a| a == abs_a)
                .ok_or("mapped action not on menu")?;
            let bucket = abs.bucket(*bot, table.shadow.board(), &mut rng);
            out.push(Decision {
                key: make_key(bucket, &table.hist),
                act_ix,
                menu_len: menu.len(),
            });
        }
        table.apply_concrete(concrete, abs);
    }
    Ok(out)
}

/// Build a clone Blueprint from a JSONL log. `holdout` in [0,1) reserves
/// that fraction of hands (deterministic on hand index) for measuring
/// top-1 agreement of the trained clone. The abstraction (and the
/// centroids serialized into the output) must be the exploiter's.
pub fn build(
    path: &str,
    abs: &Abstraction,
    holdout: f64,
) -> Result<(Blueprint, CloneStats), String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut stats = CloneStats::default();
    let mut counts: StrategyMap = StrategyMap::default();
    let mut held: Vec<Decision> = Vec::new();

    for (i, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let hand: LoggedHand = match serde_json::from_str(&line) {
            Ok(hand) => hand,
            Err(_) => {
                stats.skipped += 1;
                continue;
            }
        };
        let decisions = match replay_hand(&hand, abs) {
            Ok(d) => d,
            Err(_) => {
                stats.skipped += 1;
                continue;
            }
        };
        stats.hands += 1;
        // Deterministic even split over the hand index.
        let is_held = holdout > 0.0
            && ((i + 1) as f64 * holdout).floor() > (i as f64 * holdout).floor();
        if is_held {
            held.extend(decisions);
            continue;
        }
        for d in decisions {
            stats.decisions += 1;
            let e = counts
                .entry(d.key.to_vec())
                .or_insert_with(|| vec![0.0f32; d.menu_len]);
            if e.len() == d.menu_len {
                e[d.act_ix] += 1.0;
            }
        }
    }
    stats.infosets = counts.len();

    // Normalize counts to probabilities (Blueprint::get renormalizes too,
    // but keep the stored form a proper distribution).
    for probs in counts.values_mut() {
        let total: f32 = probs.iter().sum();
        if total > 0.0 {
            probs.iter_mut().for_each(|p| *p /= total);
        }
    }

    // Held-out top-1 agreement, by street (street = '/'-count in the key's
    // history is not stored; recover from hist length is fragile — use the
    // key directly and bin by the decision order instead: preflop keys have
    // the shortest histories). Simpler and exact: recompute street from the
    // stored key by counting street-separator tokens in its hist suffix.
    for d in &held {
        let hist = &d.key[2..];
        let street = hist
            .iter()
            .filter(|&&t| t == crate::abstraction::Token::StreetSep as u8)
            .count()
            .min(3);
        let slot = &mut stats.agree[street];
        slot.1 += 1;
        if let Some(probs) = counts.get(&d.key[..]) {
            let best = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i);
            if best == Some(d.act_ix) {
                slot.0 += 1;
            }
        }
    }

    let bp = Blueprint {
        strategies: counts,
        iterations: stats.decisions,
        num_players: Players::TWO,
        abs_cfg: abs.cfg.clone(),
        centroids: abs.centroids.clone(),
    };
    Ok((bp, stats))
}

// ---------------------------------------------------------------------------
// Tests (written first, TDD)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::{AbsConfig, Token};
    use serde_json::json;

    fn abs_small() -> Abstraction {
        Abstraction::new(AbsConfig {
            postflop_buckets: std::num::NonZero::new(6).unwrap(),
            equity_rollouts: std::num::NonZero::new(40).unwrap(),
            dist_runouts: std::num::NonZero::new(8).unwrap(),
            runout_rollouts: std::num::NonZero::new(20).unwrap(),
            cache_cap: 200_000,
            menu: crate::abstraction::MenuShape::Wide,
        })
    }

    /// client_pos 0: Slumbot is the small blind and acts first preflop.
    /// "b200f" = Slumbot raises to 200, we fold: exactly one Slumbot
    /// decision, recorded at an empty history with a raise on the menu.
    #[test]
    fn replays_a_preflop_raise_fold() {
        let abs = abs_small();
        let v = json!({
            "action": "b200f",
            "client_pos": 0,
            "hole_cards": ["6c", "4d"],
            "bot_hole_cards": ["Ts", "2c"],
            "board": [],
            "winnings": -100,
        });
        let ds = replay_hand(&serde_json::from_value(v).unwrap(), &abs).unwrap();
        assert_eq!(ds.len(), 1, "only Slumbot's raise is its decision");
        assert!(ds[0].menu_len >= 3);
        assert!(ds[0].act_ix > 1, "a raise is neither fold(0) nor call(1)");
        assert_eq!(&ds[0].key[2..], &[] as &[u8], "preflop first-to-act");
    }

    /// client_pos 1: we are the button; a checked-down hand records one
    /// Slumbot decision per street (BB checks first postflop... Slumbot
    /// checks last preflop and first on every later street).
    #[test]
    fn replays_a_checked_down_hand_across_streets() {
        let abs = abs_small();
        let v = json!({
            "action": "cc/kk/kk/kk",
            "client_pos": 1,
            "hole_cards": ["2c", "2d"],
            "bot_hole_cards": ["7c", "4d"],
            "board": ["Ah", "Kd", "Qs", "Js", "9h"],
            "winnings": -100,
        });
        let ds = replay_hand(&serde_json::from_value(v).unwrap(), &abs).unwrap();
        assert_eq!(ds.len(), 4, "one Slumbot decision per street");
        // Street separators accumulate in the history keys.
        let seps: Vec<usize> = ds
            .iter()
            .map(|d| {
                d.key[2..]
                    .iter()
                    .filter(|&&t| t == Token::StreetSep as u8)
                    .count()
            })
            .collect();
        assert_eq!(seps, vec![0, 1, 2, 3]);

        // The river decision is bucketed on the logged board, where
        // Slumbot's 7-high is the bottom bucket, not on cards the engine
        // dealt itself (here 2h 2s 3c 3d 3h, a board that chops).
        let bot = Hole::new("7c".parse().unwrap(), "4d".parse().unwrap()).unwrap();
        let board: Vec<Card> = ["Ah", "Kd", "Qs", "Js", "9h"].map(|c| c.parse().unwrap()).into();
        let mut rng = SmallRng::seed_from_u64(0);
        let want = make_key(abs.bucket(bot, &board, &mut rng), &[]);
        assert_eq!(ds[3].key[..2], want[..2]);
    }

    /// build() over a small file: counts land in the blueprint and repeated
    /// identical decisions concentrate the distribution.
    #[test]
    fn builds_a_normalized_clone_from_a_log_file() {
        let abs = abs_small();
        let line = r#"{"action":"b200f","client_pos":0,"hole_cards":["6c","4d"],"bot_hole_cards":["Ts","2c"],"board":[],"winnings":-100}"#;
        let path = std::env::temp_dir().join("clone_build_test.jsonl");
        let bad_card = line.replace("Ts", "Xs");
        let twice = line.replace("Ts", "6c");
        std::fs::write(&path, format!("{line}\n{line}\nnot json\n{bad_card}\n{twice}\n")).unwrap();
        let (bp, stats) = build(path.to_str().unwrap(), &abs, 0.0).unwrap();
        assert_eq!(stats.hands, 2);
        assert_eq!(stats.skipped, 3);
        assert_eq!(stats.decisions, 2);
        assert_eq!(stats.infosets, 1);
        let probs = bp.strategies.values().next().unwrap();
        let total: f32 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "stored dist must normalize");
        assert_eq!(probs.iter().cloned().fold(0.0, f32::max), 1.0);
        let _ = std::fs::remove_file(&path);
    }
}
