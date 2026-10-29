//! Preflop diagnostics on a served blueprint.
//!
//! - `probe`: the blueprint's strategy for every starting hand at one
//!   preflop node, plus the all-in share overall and for the premiums.
//! - `action_ev`: what each action at that node is worth to a given hand
//!   when everyone, the hero included, plays the blueprint afterwards.
//!   Opponents' cards are dealt at random and weighted by how likely the
//!   blueprint makes the line that led to the node (importance weights), so
//!   the values are against the blueprint's own ranges. If the blueprint
//!   were a best response to itself here, every action it plays would be
//!   worth the same and no action would be worth more.
//!
//! A node is a line of preflop actions from the start of the hand, with
//! the button in seat 0 as in training: UTG (seat 3) acts first, then MP,
//! CO, BTN, SB, BB. Tokens: `f` fold, `c` check/call, `rN` the menu's
//! raise to BET_SIZES[N] of the pot, `a` all-in.

use crate::abstraction::{AbsAction, Abstraction, BetSize, TOKEN_STREET_SEP};
use crate::cards::{fresh_deck, Card, Rank, Suit};
use crate::cfr::sample_index;
use crate::engine::{Hand, HandConfig, Street};
use crate::mirror::Bot;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Seat names with the button in seat 0 (six-handed).
pub const SEATS: [&str; 6] = ["BTN", "SB", "BB", "UTG", "MP", "CO"];

/// The premium hands whose all-in share players notice.
pub const PREMIUMS: [&str; 6] = ["KK", "QQ", "JJ", "TT", "AKs", "AKo"];

/// A preflop node: the hand after `line`, and the tokens it recorded.
pub struct Node {
    pub hand: Hand,
    pub hist: Vec<u8>,
    /// (seat, abstract action) for each step of the line.
    pub steps: Vec<(usize, AbsAction)>,
}

pub fn parse_action(tok: &str) -> Result<AbsAction, String> {
    match tok {
        "f" => Ok(AbsAction::Fold),
        "c" => Ok(AbsAction::CheckCall),
        "a" => Ok(AbsAction::AllIn),
        _ => tok
            .strip_prefix('r')
            .and_then(|n| n.parse::<u8>().ok())
            .and_then(BetSize::new)
            .map(AbsAction::Bet)
            .ok_or_else(|| format!("bad action token '{tok}' (f, c, rN, a)")),
    }
}

pub fn action_name(h: &Hand, abs: &Abstraction, a: AbsAction) -> String {
    let bb = h.big_blind() as f64;
    match a {
        AbsAction::Fold => "fold".into(),
        AbsAction::CheckCall if h.can_check() => "check".into(),
        AbsAction::CheckCall => "call".into(),
        AbsAction::AllIn => "all-in".into(),
        AbsAction::Bet(_) => match abs.concrete(h, a) {
            crate::engine::PlayerAction::RaiseTo(t) => format!("r{:.1}", t as f64 / bb),
            _ => unreachable!(),
        },
    }
}

/// Replay `line` (whitespace-separated tokens) from a fresh hand. The
/// deck only fixes the cards; the line's validity does not depend on it.
pub fn node(cfg: &HandConfig, abs: &Abstraction, line: &str, deck: [Card; 52]) -> Result<Node, String> {
    let mut h = Hand::new(cfg, 0, deck);
    let mut hist = Vec::new();
    let mut steps = Vec::new();
    for tok in line.split_whitespace() {
        let leaves = || format!("the line leaves preflop before '{tok}'");
        let seat = h.to_act().ok_or_else(leaves)?;
        if h.street() != Street::Preflop {
            return Err(leaves());
        }
        let a = parse_action(tok)?;
        let acts = abs.abstract_actions(&h);
        if !acts.contains(&a) {
            return Err(format!("'{tok}' is not on the menu for {}", SEATS[seat]));
        }
        steps.push((seat, a));
        h.apply(abs.concrete(&h, a));
        hist.push(a.token());
    }
    if h.is_terminal() || h.street() != Street::Preflop {
        return Err("the line does not end at a preflop decision".into());
    }
    Ok(Node { hand: h, hist, steps })
}

/// The 169 hand classes in preflop_bucket order are not contiguous by
/// name, so classes are named from a representative combo.
pub fn class_name(hi: Rank, lo: Rank, suited: bool) -> String {
    let (h, l) = (hi.to_char(), lo.to_char());
    if hi == lo {
        format!("{h}{l}")
    } else {
        format!("{h}{l}{}", if suited { 's' } else { 'o' })
    }
}

/// Every combo of a class like "KK", "AKs" or "AKo".
pub fn class_combos(name: &str) -> Result<Vec<[Card; 2]>, String> {
    let chars: Vec<char> = name.chars().collect();
    let r = Rank::from_char;
    let bad = || format!("bad hand class '{name}' (e.g. KK, AKs, AKo)");
    let (a, b) = match (chars.first().and_then(|&c| r(c)), chars.get(1).and_then(|&c| r(c))) {
        (Some(a), Some(b)) => (a.max(b), a.min(b)),
        _ => return Err(bad()),
    };
    let kind = chars.get(2).copied();
    let mut out = Vec::new();
    for s1 in Suit::ALL {
        for s2 in Suit::ALL {
            let ok = match (a == b, kind) {
                (true, None) => s1 < s2,
                (false, Some('s')) => s1 == s2,
                (false, Some('o')) => s1 != s2,
                _ => return Err(bad()),
            };
            if ok {
                out.push([Card::new(a, s1), Card::new(b, s2)]);
            }
        }
    }
    Ok(out)
}

/// Every class name, strongest-looking first (pairs, then by ranks).
pub fn all_classes() -> Vec<String> {
    let mut out = Vec::new();
    for hi in Rank::ALL.into_iter().rev() {
        out.push(class_name(hi, hi, false));
    }
    for (h, &hi) in Rank::ALL.iter().enumerate().rev() {
        for &lo in Rank::ALL[..h].iter().rev() {
            out.push(class_name(hi, lo, true));
            out.push(class_name(hi, lo, false));
        }
    }
    out
}

/// The blueprint's normalized strategy at the player to act's infoset;
/// check/call when the infoset is unseen or its menu differs (as served).
pub fn dist(bot: &Bot, h: &Hand, hist: &[u8], rng: &mut SmallRng) -> (Vec<AbsAction>, Vec<f64>, bool) {
    let acts = bot.abs.abstract_actions(h);
    let bucket = bot.abs.bucket(h.hole(h.to_act().expect("a live decision")), h.board(), rng);
    if let Some(w) = bot.strategies.lookup(bucket, hist, &acts).fits() {
        return (acts, w.probs(), true);
    }
    let mut p = vec![0.0; acts.len()];
    p[acts.iter().position(|&a| a == AbsAction::CheckCall).unwrap()] = 1.0;
    (acts, p, false)
}

/// Strategy rows of `probe`: per class, the action shares at the node.
pub struct Probe {
    pub actions: Vec<String>,
    pub rows: Vec<(String, Vec<f64>)>,
    /// All-in share over all 1,326 combos (unweighted by reach).
    pub all_in: f64,
    /// Unweighted mean all-in share of the PREMIUMS.
    pub premium_all_in: f64,
    /// Share of the all-in combos that come from classes going all-in less
    /// than 5% of the time: averaging residue rather than a real range.
    pub residue: f64,
}

pub fn probe(bot: &Bot, cfg: &HandConfig, line: &str) -> Result<Probe, String> {
    let n0 = node(cfg, &bot.abs, line, fresh_deck())?;
    let hero = n0.hand.to_act().expect("node ends at a decision");
    let acts = bot.abs.abstract_actions(&n0.hand);
    let all_in_at = acts.iter().position(|&a| a == AbsAction::AllIn);
    let mut rng = SmallRng::seed_from_u64(0);
    let mut rows = Vec::new();
    let (mut ai_sum, mut ai_residue, mut combos) = (0.0, 0.0, 0usize);
    for name in all_classes() {
        let cs = class_combos(&name)?;
        // Preflop buckets depend on the class only.
        let mut deck = fresh_deck();
        place(&mut deck, hero, cs[0]);
        let n = node(cfg, &bot.abs, line, deck)?;
        let (_, p, _) = dist(bot, &n.hand, &n.hist, &mut rng);
        if let Some(i) = all_in_at {
            ai_sum += p[i] * cs.len() as f64;
            if p[i] < 0.05 {
                ai_residue += p[i] * cs.len() as f64;
            }
        }
        combos += cs.len();
        rows.push((name, p));
    }
    let share = |name: &str| {
        let p = &rows.iter().find(|r| r.0 == name).unwrap().1;
        all_in_at.map_or(0.0, |i| p[i])
    };
    let premium_all_in = PREMIUMS.iter().map(|n| share(n)).sum::<f64>() / PREMIUMS.len() as f64;
    Ok(Probe {
        actions: acts.iter().map(|&a| action_name(&n0.hand, &bot.abs, a)).collect(),
        rows,
        all_in: ai_sum / combos as f64,
        premium_all_in,
        residue: if ai_sum > 0.0 { ai_residue / ai_sum } else { 0.0 },
    })
}

/// Put `hole` into `seat`'s deal slots, swapping out whatever was there.
fn place(deck: &mut [Card; 52], seat: usize, hole: [Card; 2]) {
    for (k, &c) in hole.iter().enumerate() {
        let at = deck.iter().position(|&x| x == c).unwrap();
        deck.swap(2 * seat + k, at);
    }
}

/// One sampled deal of `action_ev`: its importance weight (the reach of the
/// line for everyone but the hero) and the hero's value of each action and
/// of the blueprint's mix, in big blinds.
struct Deal {
    weight: f64,
    vals: Vec<f64>,
    mixed: f64,
}

/// Values of every action at the node for one hand class, in big blinds.
pub struct ActionEv {
    pub class: String,
    pub actions: Vec<String>,
    /// The blueprint's mix at the node.
    pub mix: Vec<f64>,
    /// Weighted mean value of each action, bb, with a 95% half-width.
    pub ev: Vec<(f64, f64)>,
    /// Each action minus the blueprint's mix (paired per deal), bb, 95%.
    pub vs_mix: Vec<(f64, f64)>,
    /// Effective sample size of the importance weights.
    pub ess: f64,
    pub deals: u64,
    /// The hero's own probability of having played the line with this
    /// class (preflop strategies depend on the class only).
    pub hero_reach: f64,
}

impl ActionEv {
    /// What playing the best action instead of the mix gains, bb.
    pub fn gain(&self) -> f64 {
        self.vs_mix.iter().map(|v| v.0).fold(0.0, f64::max)
    }
}

/// Play the hand out with everyone on the blueprint; the hero's chips.
fn playout(bot: &Bot, mut h: Hand, mut hist: Vec<u8>, hero: usize, rng: &mut SmallRng) -> i64 {
    while !h.is_terminal() {
        let (acts, p, _) = dist(bot, &h, &hist, rng);
        let a = acts[sample_index(&p, rng)];
        let street = h.street();
        h.apply(bot.abs.concrete(&h, a));
        hist.push(a.token());
        if !h.is_terminal() && h.street() != street {
            hist.push(TOKEN_STREET_SEP);
        }
    }
    h.utilities()[hero]
}

pub fn action_ev(bot: &Bot, cfg: &HandConfig, line: &str, class: &str, deals: u64, seed: u64) -> Result<ActionEv, String> {
    let combos = class_combos(class)?;
    let n0 = node(cfg, &bot.abs, line, fresh_deck())?;
    let hero = n0.hand.to_act().expect("node ends at a decision");
    let acts = bot.abs.abstract_actions(&n0.hand);
    let k = acts.len();
    let bb = cfg.blinds.bb() as f64;
    let rows: Vec<Deal> = (0..deals)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut deck = fresh_deck();
            deck.shuffle(&mut rng);
            place(&mut deck, hero, combos[rng.random_range(0..combos.len())]);
            let n = node(cfg, &bot.abs, line, deck).expect("line checked above");
            // Reach of the line for everyone but the hero, whose own reach
            // is the same for every combo of the class.
            let mut h = Hand::new(cfg, 0, deck);
            let mut w = 1.0;
            for (j, &(seat, a)) in n.steps.iter().enumerate() {
                if seat != hero {
                    let (acts, p, _) = dist(bot, &h, &n.hist[..j], &mut rng);
                    w *= p[acts.iter().position(|&x| x == a).unwrap()];
                }
                h.apply(bot.abs.concrete(&h, a));
            }
            let (_, mix, _) = dist(bot, &n.hand, &n.hist, &mut rng);
            if w == 0.0 {
                return Deal { weight: 0.0, vals: vec![0.0; k], mixed: 0.0 };
            }
            // Common random numbers: every action is played out with the
            // same deck and the same sampling stream.
            let play_seed: u64 = rng.random();
            let vals: Vec<f64> = acts
                .iter()
                .map(|&a| {
                    let mut child = n.hand.clone();
                    child.apply(bot.abs.concrete(&n.hand, a));
                    let mut hist = n.hist.clone();
                    hist.push(a.token());
                    if !child.is_terminal() && child.street() != Street::Preflop {
                        hist.push(TOKEN_STREET_SEP);
                    }
                    let mut prng = SmallRng::seed_from_u64(play_seed);
                    playout(bot, child, hist, hero, &mut prng) as f64 / bb
                })
                .collect();
            let mixed: f64 = vals.iter().zip(&mix).map(|(v, p)| v * p).sum();
            Deal { weight: w, vals, mixed }
        })
        .collect();
    let mut rng = SmallRng::seed_from_u64(0);
    let mut deck = fresh_deck();
    place(&mut deck, hero, combos[0]);
    let nh = node(cfg, &bot.abs, line, deck)?;
    let (_, mix, _) = dist(bot, &nh.hand, &nh.hist, &mut rng);
    let mut hero_reach = 1.0;
    let mut h = Hand::new(cfg, 0, deck);
    for (j, &(seat, a)) in nh.steps.iter().enumerate() {
        if seat == hero {
            let (acts, p, _) = dist(bot, &h, &nh.hist[..j], &mut rng);
            hero_reach *= p[acts.iter().position(|&x| x == a).unwrap()];
        }
        h.apply(bot.abs.concrete(&h, a));
    }
    let sw: f64 = rows.iter().map(|r| r.weight).sum();
    let sw2: f64 = rows.iter().map(|r| r.weight * r.weight).sum();
    if sw == 0.0 {
        return Err("no deal reached the node".into());
    }
    // Self-normalized importance-sampling mean and its delta-method
    // standard error.
    let est = |f: &dyn Fn(&Deal) -> f64| {
        let m = rows.iter().map(|r| r.weight * f(r)).sum::<f64>() / sw;
        let var = rows.iter().map(|r| (r.weight * (f(r) - m)).powi(2)).sum::<f64>() / (sw * sw);
        (m, 1.96 * var.sqrt())
    };
    Ok(ActionEv {
        class: class.to_string(),
        actions: acts.iter().map(|&a| action_name(&n0.hand, &bot.abs, a)).collect(),
        mix,
        ev: (0..k).map(|j| est(&|r| r.vals[j])).collect(),
        vs_mix: (0..k).map(|j| est(&|r| r.vals[j] - r.mixed)).collect(),
        ess: sw * sw / sw2,
        deals,
        hero_reach,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::AbsConfig;

    #[test]
    fn classes_have_the_right_combo_counts() {
        assert_eq!(class_combos("KK").unwrap().len(), 6);
        assert_eq!(class_combos("AKs").unwrap().len(), 4);
        assert_eq!(class_combos("AKo").unwrap().len(), 12);
        assert!(class_combos("AK").is_err());
        let all = all_classes();
        assert_eq!(all.len(), 169);
        let total: usize = all.iter().map(|c| class_combos(c).unwrap().len()).sum();
        assert_eq!(total, 1326);
    }

    #[test]
    fn button_open_reaches_the_big_blind() {
        let abs = Abstraction::new(AbsConfig::default());
        let n = node(&HandConfig::default(), &abs, "f f f r3 f", fresh_deck()).unwrap();
        assert_eq!(SEATS[n.hand.to_act().unwrap()], "BB");
        let names: Vec<String> = abs
            .abstract_actions(&n.hand)
            .iter()
            .map(|&a| action_name(&n.hand, &abs, a))
            .collect();
        assert_eq!(names, ["fold", "call", "r6.0", "r9.1", "r15.3", "all-in"]);
        assert!(node(&HandConfig::default(), &abs, "f f f r9", fresh_deck()).is_err());
    }
}
