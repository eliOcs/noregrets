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

use crate::abstraction::{AbsAction, Abstraction, BetSize, Token};
use crate::cards::{fresh_deck, Card, Deck, Hole, Rank, Suit};
use crate::cfr::sample_index;
use crate::engine::{Hand, HandConfig, Street};
use crate::mirror::Bot;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Seat names with the button in seat 0 (six-handed).
pub const SEATS: [&str; 6] = ["BTN", "SB", "BB", "UTG", "MP", "CO"];

/// The premium hands whose all-in share players notice.
pub const PREMIUMS: [&str; 6] = ["KK", "QQ", "JJ", "TT", "AKs", "AKo"];

/// A preflop node: the hand after `line`, and the tokens it recorded.
pub struct Node {
    pub hand: Hand,
    pub hist: Vec<Token>,
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
pub fn node(cfg: &HandConfig, abs: &Abstraction, line: &str, deck: Deck) -> Result<Node, String> {
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
pub fn class_combos(name: &str) -> Result<Vec<Hole>, String> {
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
                out.push(Hole::new(Card::new(a, s1), Card::new(b, s2)).expect("a class names two different cards"));
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
pub fn dist(bot: &Bot, h: &Hand, hist: &[Token], rng: &mut SmallRng) -> (Vec<AbsAction>, Vec<f64>, bool) {
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
        deck.deal_to(hero, cs[0]);
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
fn playout(bot: &Bot, mut h: Hand, mut hist: Vec<Token>, hero: usize, rng: &mut SmallRng) -> i64 {
    while !h.is_terminal() {
        let (acts, p, _) = dist(bot, &h, &hist, rng);
        let a = acts[sample_index(&p, rng)];
        let street = h.street();
        h.apply(bot.abs.concrete(&h, a));
        hist.push(a.token());
        if !h.is_terminal() && h.street() != street {
            hist.push(Token::StreetSep);
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
            deck.deal_to(hero, combos[rng.random_range(0..combos.len())]);
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
                        hist.push(Token::StreetSep);
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
    deck.deal_to(hero, combos[0]);
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

/// One preflop decision of `reach`.
pub struct ReachRow {
    /// The line in `--line` tokens.
    pub tokens: String,
    /// The line with seats and sizes, e.g. "UTG r2.9, MP a".
    pub line: String,
    pub seat: usize,
    /// Probability that every other seat plays the line (card removal
    /// ignored). External sampling enumerates the actor's own actions and
    /// samples everyone else's, so this is the share of the actor's
    /// traversals that reach the node: regret updates per combo are about
    /// iterations / players * opp_reach / 1326.
    pub opp_reach: f64,
    /// Probability that the actor's own hand plays the line: with
    /// `opp_reach`, how often the bot arrives here in self-play.
    pub own_reach: f64,
    /// `serve --search` answers here with the preflop all-in resolve, not
    /// the blueprint.
    pub resolved: bool,
    /// Combos (of 1,326) whose infoset holds no usable strategy.
    pub unseen: usize,
    /// With a second blueprint: the mean total variation distance between
    /// the two strategies over the actor's classes, weighted by how often
    /// each class plays the line under the first. None when no class does.
    pub diff: Option<f64>,
}

/// Every preflop decision of the menu's tree, with how often training and
/// self-play reach it under the blueprint's own strategies. Ranges are per
/// hand class, unseen infosets check/call as served.
pub fn reach(bot: &Bot, other: Option<&Bot>, cfg: &HandConfig) -> Vec<ReachRow> {
    let classes = all_classes()
        .iter()
        .map(|name| {
            let cs = class_combos(name).expect("all_classes names valid classes");
            (crate::abstraction::preflop_bucket(*cs[0]), cs.len())
        })
        .collect();
    let mut walk = ReachWalk { bot, other, classes, rows: Vec::new() };
    let h = Hand::new(cfg, 0, fresh_deck());
    let mut ranges = vec![vec![1.0; 169]; h.num_players()];
    walk.node(&h, &mut Vec::new(), &mut Vec::new(), &mut ranges);
    walk.rows
}

struct ReachWalk<'a> {
    bot: &'a Bot,
    other: Option<&'a Bot>,
    /// (preflop bucket, combos) per class.
    classes: Vec<(u16, usize)>,
    rows: Vec<ReachRow>,
}

impl ReachWalk<'_> {
    /// `ranges[q][c]`: the probability that seat q holding class c played
    /// q's actions of the line so far.
    fn node(&mut self, h: &Hand, hist: &mut Vec<Token>, line: &mut Vec<(String, String)>, ranges: &mut [Vec<f64>]) {
        let Some(p) = h.to_act() else { return };
        if h.street() != Street::Preflop {
            return;
        }
        let mass = |r: &[f64]| r.iter().zip(&self.classes).map(|(&w, &(_, n))| w * n as f64).sum::<f64>() / 1326.0;
        let opp_reach = (0..ranges.len()).filter(|&q| q != p).map(|q| mass(&ranges[q])).product();
        let own_reach = mass(&ranges[p]);
        let acts = self.bot.abs.abstract_actions(h);
        let served = |bot: &Bot, bucket: u16| -> Option<Vec<f64>> {
            bot.strategies.lookup(bucket, hist, &acts).fits().map(|w| w.probs())
        };
        let check_call: Vec<f64> = acts.iter().map(|&a| if a == AbsAction::CheckCall { 1.0 } else { 0.0 }).collect();
        let mut unseen = 0;
        let sigma: Vec<Vec<f64>> = self
            .classes
            .iter()
            .map(|&(bucket, n)| {
                served(self.bot, bucket).unwrap_or_else(|| {
                    unseen += n;
                    check_call.clone()
                })
            })
            .collect();
        let diff = self.other.and_then(|other| {
            let (mut sum, mut weight) = (0.0, 0.0);
            for (c, &(bucket, n)) in self.classes.iter().enumerate() {
                let w = ranges[p][c] * n as f64;
                let theirs = served(other, bucket).unwrap_or_else(|| check_call.clone());
                let tv: f64 = sigma[c].iter().zip(&theirs).map(|(a, b)| (a - b).abs()).sum::<f64>() / 2.0;
                sum += w * tv;
                weight += w;
            }
            (weight > 0.0).then(|| sum / weight)
        });
        self.rows.push(ReachRow {
            tokens: line.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>().join(" "),
            line: line.iter().map(|(_, l)| l.as_str()).collect::<Vec<_>>().join(", "),
            seat: p,
            opp_reach,
            own_reach,
            resolved: crate::bot::preflop_allin_spot(h),
            unseen,
            diff,
        });
        let before = ranges[p].clone();
        for (i, &a) in acts.iter().enumerate() {
            for (c, w) in ranges[p].iter_mut().enumerate() {
                *w = before[c] * sigma[c][i];
            }
            let token = match a {
                AbsAction::Fold => "f".to_string(),
                AbsAction::CheckCall => "c".to_string(),
                AbsAction::AllIn => "a".to_string(),
                AbsAction::Bet(s) => format!("r{}", s.index()),
            };
            line.push((token, format!("{} {}", SEATS[p], action_name(h, &self.bot.abs, a))));
            hist.push(a.token());
            let mut child = h.clone();
            child.apply(self.bot.abs.concrete(h, a));
            self.node(&child, hist, line, ranges);
            hist.pop();
            line.pop();
        }
        ranges[p] = before;
    }
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

    #[test]
    fn reach_follows_the_served_check_call_of_an_empty_blueprint() {
        use crate::cfr::{Blueprint, StrategyMap};
        use crate::engine::{Players, Stack};
        let bp = Blueprint {
            strategies: StrategyMap::default(),
            iterations: 0,
            num_players: Players::new(2).unwrap(),
            abs_cfg: AbsConfig::default(),
            centroids: None,
        };
        let bot = Bot { abs: std::sync::Arc::new(Abstraction::new(bp.abs_cfg.clone())), strategies: Box::new(bp) };
        let cfg = HandConfig::new(Players::new(2).unwrap(), Stack::new(1000).unwrap());
        let rows = reach(&bot, None, &cfg);
        assert!(rows.len() > 2);
        // Heads-up preflop the seats alternate: the last token is the
        // opponent's, the one before it the actor's, and so on.
        for r in &rows {
            let toks: Vec<&str> = r.tokens.split_whitespace().collect();
            let played = |parity: usize| toks.iter().rev().skip(parity).step_by(2).all(|&t| t == "c");
            assert_eq!(r.opp_reach, if played(0) { 1.0 } else { 0.0 }, "{}", r.tokens);
            assert_eq!(r.own_reach, if played(1) { 1.0 } else { 0.0 }, "{}", r.tokens);
            assert_eq!(r.unseen, 1326);
        }
    }
}
