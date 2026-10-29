//! What do off-menu bet sizes win against bots that translate them to the
//! log-nearest menu size? One hero seat plays the blueprint but resizes
//! every bet it makes; the field plays the same blueprint and sees each
//! resized bet through the usual translation. Every deal is played twice
//! with the same cards, button and sampling seed: once resized, once on
//! the menu. The per-deal difference is what the resizing gains.
//!
//! Resized amounts are chosen against the bots' shadow, which keeps the
//! menu amounts, so the real pot drifts from the shadow's after the first
//! resized bet, as it does at a real table.

use crate::abstraction::{AbsAction, Abstraction};
use crate::cards::{fresh_deck, Card};
use crate::cfr::sample_index;
use crate::engine::{HandConfig, PlayerAction};
use crate::mirror::{Bot, Mirror};
use crate::table::mean_ci;
use clap::ValueEnum;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Resize {
    /// The largest amount that still maps to the chosen size.
    Up,
    /// The smallest amount that still maps to the chosen size.
    Down,
    /// Log-uniform between `down` and `up`: arbitrary human sizes.
    Random,
    /// Every bet and raise is a min-raise.
    Min,
}

pub struct Report {
    pub hands: u64,
    /// Resized hero's winrate minus on-menu hero's, mbb/hand, 95% half-width.
    pub gain: (f64, f64),
    pub resized: (f64, f64),
    pub on_menu: (f64, f64),
    /// Share of hero bets whose amount changed.
    pub resized_share: f64,
}

pub fn run(bot: &Bot, cfg: &HandConfig, resize: Resize, hands: u64, seed: u64) -> Report {
    let n = cfg.num_players.get();
    let bb = cfg.bb as f64;
    let rows: Vec<(f64, f64, u64, u64)> = (0..hands)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed ^ i.wrapping_mul(0x2545_F491_4F6C_DD1D));
            let hero = (i % n as u64) as usize;
            let button = rng.random_range(0..n);
            let mut deck = fresh_deck();
            deck.shuffle(&mut rng);
            let play_seed = rng.random();
            let (ur, bets, changed) = play(bot, cfg, Some(resize), hero, button, deck, play_seed);
            let (um, _, _) = play(bot, cfg, None, hero, button, deck, play_seed);
            (ur as f64 / bb * 1000.0, um as f64 / bb * 1000.0, bets, changed)
        })
        .collect();
    let col = |f: fn(&(f64, f64, u64, u64)) -> f64| mean_ci(&rows.iter().map(f).collect::<Vec<_>>());
    let bets: u64 = rows.iter().map(|r| r.2).sum();
    let changed: u64 = rows.iter().map(|r| r.3).sum();
    Report {
        hands,
        gain: col(|r| r.0 - r.1),
        resized: col(|r| r.0),
        on_menu: col(|r| r.1),
        resized_share: changed as f64 / bets.max(1) as f64,
    }
}

/// One hand. Returns the hero's chip result, the hero's bets, and how many
/// of them were resized.
fn play(
    bot: &Bot,
    cfg: &HandConfig,
    resize: Option<Resize>,
    hero: usize,
    button: usize,
    deck: [Card; 52],
    seed: u64,
) -> (i64, u64, u64) {
    let mut rng = SmallRng::seed_from_u64(seed);
    // Sizing draws its own randomness, so both runs sample actions alike.
    let mut size_rng = SmallRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
    let stacks = vec![cfg.stack; cfg.num_players.get()];
    let mut m = Mirror::new(cfg, button, deck, &stacks);
    let (mut bets, mut changed) = (0, 0);
    for _ in 0..500 {
        if m.real.is_terminal() {
            return (m.real.utilities()[hero], bets, changed);
        }
        let to_act = m.real.to_act();
        let choices = m.strategy(bot).0;
        let probs: Vec<f64> = choices.iter().map(|c| c.1).collect();
        let mut a: PlayerAction = choices[sample_index(&probs, &mut rng)].0;
        if let (true, Some(r), PlayerAction::RaiseTo(x)) = (to_act == hero, resize, a) {
            bets += 1;
            let y = resized(&m, &bot.abs, x, r, &mut size_rng);
            changed += (y != x) as u64;
            a = PlayerAction::RaiseTo(y);
        }
        m.apply(a, &bot.abs);
    }
    unreachable!("offmenu-match hand did not terminate")
}

/// Resize a bet of `x` (a menu amount clamped into the real bounds) so
/// that it still translates to the same menu size. All-ins stay all-ins.
fn resized(m: &Mirror, abs: &Abstraction, x: u32, r: Resize, rng: &mut SmallRng) -> u32 {
    let (lo, hi) = m.real.raise_bounds().expect("hero chose a raise");
    let shadow = m.shadow();
    let want = abs.map_raise(shadow, x);
    if x == hi || want == AbsAction::AllIn {
        return x;
    }
    if r == Resize::Min {
        return lo;
    }
    let mut menu: Vec<u32> = abs
        .abstract_actions(shadow)
        .into_iter()
        .filter_map(|a| match abs.concrete(shadow, a) {
            PlayerAction::RaiseTo(t) => Some(t),
            _ => None,
        })
        .collect();
    menu.sort_unstable();
    menu.dedup();
    let PlayerAction::RaiseTo(t) = abs.concrete(shadow, want) else {
        unreachable!("a bet translates to a raise")
    };
    let i = menu.iter().position(|&u| u == t).expect("the menu holds its own sizes");
    let mid = |a: u32, b: u32| ((a as f64) * (b as f64)).sqrt();
    // Strictly inside the geometric midpoints, so translation keeps `t`.
    let down = if i == 0 { lo } else { (mid(menu[i - 1], t).floor() as u32 + 1).max(lo) };
    let up = match menu.get(i + 1) {
        Some(&next) => (mid(t, next).ceil() as u32 - 1).min(hi),
        None => hi,
    };
    if down > up {
        return x;
    }
    let y = match r {
        Resize::Up => up,
        Resize::Down => down,
        Resize::Random => {
            let (l, u) = ((down as f64).ln(), (up as f64).ln());
            ((l + rng.random::<f64>() * (u - l)).exp().round() as u32).clamp(down, up)
        }
        Resize::Min => unreachable!(),
    };
    if abs.map_raise(shadow, y) == want { y } else { x }
}
