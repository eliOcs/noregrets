//! What does playing a blueprint at the wrong stack depth cost? One hero
//! seat plays bot A through a mirror at A's training depth against a
//! field of bot B at B's own depth, with real stacks of `stack` chips.
//! Every deal is played twice with the same cards, button and sampling
//! seed: once with A as the hero, once with B. The per-deal difference is
//! A's cost of playing off-depth, with card luck cancelled.

use crate::cards::{fresh_deck, Card};
use crate::cfr::sample_index;
use crate::engine::{HandConfig, PlayerAction};
use crate::mirror::{Bot, Mirror, Source};
use crate::table::mean_ci;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// A bot and the hand config it was trained at (`stack` = depth).
pub struct Side<'a> {
    pub bot: &'a Bot,
    pub cfg: HandConfig,
}

pub struct Report {
    pub hands: u64,
    /// Hero A's winrate minus hero B's, mbb/hand, with a 95% half-width.
    pub cost: (f64, f64),
    pub a: (f64, f64),
    pub b: (f64, f64),
    /// Share of A-hero decisions that were unseen / capped.
    pub a_unseen: f64,
    pub a_capped: f64,
}

#[derive(Default, Clone, Copy)]
struct Counts {
    decisions: u64,
    unseen: u64,
    capped: u64,
}

pub fn run(a: &Side, b: &Side, stack: u32, hands: u64, seed: u64) -> Report {
    let n = a.cfg.num_players;
    let bb = a.cfg.bb as f64;
    let rows: Vec<(f64, f64, Counts)> = (0..hands)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed ^ i.wrapping_mul(0x2545_F491_4F6C_DD1D));
            let hero = (i % n as u64) as usize;
            let button = rng.random_range(0..n);
            let mut deck = fresh_deck();
            deck.shuffle(&mut rng);
            let play_seed = rng.random();
            let stacks = vec![stack; n];
            let (ua, counts) = play(a, b, hero, button, deck, &stacks, play_seed);
            let (ub, _) = play(b, b, hero, button, deck, &stacks, play_seed);
            (ua as f64 / bb * 1000.0, ub as f64 / bb * 1000.0, counts)
        })
        .collect();
    let col = |f: fn(&(f64, f64, Counts)) -> f64| mean_ci(&rows.iter().map(f).collect::<Vec<_>>());
    let total = rows.iter().fold(Counts::default(), |t, r| Counts {
        decisions: t.decisions + r.2.decisions,
        unseen: t.unseen + r.2.unseen,
        capped: t.capped + r.2.capped,
    });
    let share = |k: u64| k as f64 / total.decisions.max(1) as f64;
    Report {
        hands,
        cost: col(|r| r.0 - r.1),
        a: col(|r| r.0),
        b: col(|r| r.1),
        a_unseen: share(total.unseen),
        a_capped: share(total.capped),
    }
}

/// One hand: `hero` plays `hero_side`, everyone else `field`. Returns the
/// hero's chip result and the hero's decision sources.
fn play(
    hero_side: &Side,
    field: &Side,
    hero: usize,
    button: usize,
    deck: [Card; 52],
    stacks: &[u32],
    seed: u64,
) -> (i64, Counts) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut hm = Mirror::new(&hero_side.cfg, button, deck, stacks);
    let mut fm = Mirror::new(&field.cfg, button, deck, stacks);
    let mut counts = Counts::default();
    for _ in 0..500 {
        if hm.real.is_terminal() {
            return (hm.real.utilities()[hero], counts);
        }
        let choices = if hm.real.to_act() == hero {
            let (c, s) = hm.strategy(hero_side.bot);
            counts.decisions += 1;
            counts.unseen += (s == Source::Unseen) as u64;
            counts.capped += (s == Source::Capped) as u64;
            c
        } else {
            fm.strategy(field.bot).0
        };
        let probs: Vec<f64> = choices.iter().map(|c| c.1).collect();
        let a: PlayerAction = choices[sample_index(&probs, &mut rng)].0;
        hm.apply(a, &hero_side.bot.abs);
        fm.apply(a, &field.bot.abs);
    }
    unreachable!("depth-match hand did not terminate")
}
