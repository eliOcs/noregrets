//! A real hand at arbitrary stacks, mirrored by an on-tree shadow hand at
//! the blueprint's training depth.
//!
//! `Table` keeps its shadow at the real stacks, so any depth other than the
//! training one changes the bet menu and blueprint lookups miss. Here every
//! shadow stack is the training stack, whatever the real stacks are:
//!
//! - Bet sizes come from the shadow and are clamped into the real hand's
//!   bounds, so a short stack's large bet becomes its all-in.
//! - A short stack's all-in maps to the nearest bet size, not to the
//!   shadow's (much larger) all-in, so callers keep their shadow chips.
//! - A player the real hand skips because they are all-in checks/calls in
//!   the shadow, as if they had chips and just called along.
//! - A player who is all-in in the shadow but not in the real hand (a deep
//!   stack that has committed a training stack's worth) is "capped": the
//!   shadow ignores their actions and they check/call from then on.

use crate::abstraction::{AbsAction, Abstraction, TOKEN_STREET_SEP};
use crate::cards::Card;
use crate::cfr::Blueprint;
use crate::engine::{Hand, HandConfig, PlayerAction};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::borrow::Cow;
use std::sync::Arc;

/// Blueprint strategies by infoset, whatever their storage.
pub trait Strategies: Send + Sync {
    /// The stored (possibly unnormalized) action weights, if any:
    /// borrowed where they are stored as floats, decoded otherwise.
    fn get(&self, bucket: u16, hist: &[u8]) -> Option<Cow<'_, [f32]>>;
    fn len(&self) -> usize;
}

impl Strategies for Blueprint {
    fn get(&self, bucket: u16, hist: &[u8]) -> Option<Cow<'_, [f32]>> {
        Blueprint::get(self, bucket, hist).map(|s| Cow::Borrowed(s.as_slice()))
    }
    fn len(&self) -> usize {
        self.strategies.len()
    }
}

/// Strategies shared between a served bot and its search `Policy`.
impl<T: Strategies + ?Sized> Strategies for Arc<T> {
    fn get(&self, bucket: u16, hist: &[u8]) -> Option<Cow<'_, [f32]>> {
        (**self).get(bucket, hist)
    }
    fn len(&self) -> usize {
        (**self).len()
    }
}

/// Preflop strategies from one source over everything else from another
/// (a preflop refinement served over the blueprint it refined). A key is
/// preflop when its history has no street separator.
pub struct PreflopOverlay {
    /// Training or compact; only its preflop keys are read.
    pub preflop: Box<dyn Strategies>,
    pub base: Box<dyn Strategies>,
}

impl Strategies for PreflopOverlay {
    fn get(&self, bucket: u16, hist: &[u8]) -> Option<Cow<'_, [f32]>> {
        if !hist.contains(&TOKEN_STREET_SEP) {
            if let Some(s) = self.preflop.get(bucket, hist) {
                return Some(s);
            }
        }
        self.base.get(bucket, hist)
    }
    fn len(&self) -> usize {
        self.base.len()
    }
}

/// What a bot needs to act: its card abstraction and its strategies.
pub struct Bot {
    pub abs: Arc<Abstraction>,
    pub strategies: Box<dyn Strategies>,
}

/// Where a strategy came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Blueprint,
    /// The blueprint never visited this infoset (or its menu differs):
    /// the bot checks/calls.
    Unseen,
    /// The bot already committed a training stack's worth: it checks/calls.
    Capped,
    /// Online search resolved the spot.
    Search,
}

/// One shadow action and the state it was taken from: what a range
/// tracker observes.
#[derive(Clone)]
pub struct Observed {
    pub seat: usize,
    pub action: AbsAction,
    pub shadow: Hand,
    pub hist: Vec<u8>,
}

pub struct Mirror {
    pub real: Hand,
    shadow: Hand,
    hist: Vec<u8>,
    log: Option<Vec<Observed>>,
}

impl Mirror {
    /// `cfg.stack` is the training depth the shadow plays at.
    pub fn new(cfg: &HandConfig, button: usize, deck: [Card; 52], stacks: &[u32]) -> Mirror {
        Mirror {
            real: Hand::new_with_stacks(cfg, button, deck, stacks),
            shadow: Hand::new(cfg, button, deck),
            hist: Vec::with_capacity(32),
            log: None,
        }
    }

    /// Record every shadow action from here on (see `log`).
    pub fn with_log(mut self) -> Mirror {
        self.log = Some(Vec::new());
        self
    }

    /// The shadow actions so far, auto-calls included, in order; empty
    /// unless built `with_log`.
    pub fn log(&self) -> &[Observed] {
        self.log.as_deref().unwrap_or(&[])
    }

    /// The shadow's abstract action history.
    pub fn hist(&self) -> &[u8] {
        &self.hist
    }

    /// Bring the shadow to the real hand's player to act. False when that
    /// player is capped and can only check/call.
    pub fn sync(&mut self, abs: &Abstraction) -> bool {
        let p = self.real.to_act();
        self.sync_to(p, abs)
    }

    /// Apply the real hand's next action; the shadow follows on-tree.
    pub fn apply(&mut self, a: PlayerAction, abs: &Abstraction) {
        let p = self.real.to_act();
        if self.sync_to(p, abs) {
            let mapped = match a {
                PlayerAction::Fold if self.shadow.can_check() => AbsAction::CheckCall,
                PlayerAction::Fold => AbsAction::Fold,
                PlayerAction::CheckCall => AbsAction::CheckCall,
                PlayerAction::RaiseTo(x) => match self.shadow.raise_bounds() {
                    // A real all-in that is shorter than the shadow's.
                    Some((_, hi))
                        if x < hi && self.real.raise_bounds().is_some_and(|(_, all_in)| all_in == x) =>
                    {
                        self.nearest_bet(x, abs)
                    }
                    Some(_) => abs.map_raise(&self.shadow, x),
                    None => AbsAction::CheckCall,
                },
            };
            self.shadow_apply(mapped, abs);
        }
        self.real.apply(a);
    }

    /// The strategy for the player to act on the real hand, as concrete
    /// actions with probabilities (duplicates merged).
    pub fn strategy(&mut self, bot: &Bot) -> (Vec<(PlayerAction, f64)>, Source) {
        let p = self.real.to_act();
        if !self.sync_to(p, &bot.abs) {
            return (vec![(PlayerAction::CheckCall, 1.0)], Source::Capped);
        }
        let shadow = &self.shadow;
        let acts = bot.abs.abstract_actions(shadow);
        // Bucketing seeds its Monte Carlo draws from the cards, so this RNG
        // is never consumed; a fixed seed keeps answers reproducible anyway.
        let mut rng = SmallRng::seed_from_u64(0);
        let bucket = bot.abs.bucket(shadow.hole(p), shadow.board(), &mut rng);
        let stored = bot.strategies.get(bucket, &self.hist).filter(|s| s.len() == acts.len());
        let total: f64 = stored.as_deref().into_iter().flatten().map(|&x| x as f64).sum();
        let (probs, source) = match stored {
            Some(s) if total > 0.0 => (s.iter().map(|&x| x as f64 / total).collect(), Source::Blueprint),
            // Unseen (or menu-mismatched) infosets check/call, like the
            // engine's own blueprint player.
            _ => {
                let call = acts.iter().position(|&a| a == AbsAction::CheckCall);
                let mut probs = vec![0.0; acts.len()];
                probs[call.expect("check/call is always legal")] = 1.0;
                (probs, Source::Unseen)
            }
        };

        (self.concrete(&bot.abs, &acts, &probs), source)
    }

    /// Abstract shadow actions with probabilities as concrete actions on the
    /// real hand (duplicates merged). Call after `sync`.
    pub fn concrete(&self, abs: &Abstraction, acts: &[AbsAction], probs: &[f64]) -> Vec<(PlayerAction, f64)> {
        let (real, shadow) = (&self.real, &self.shadow);
        let mut out: Vec<(PlayerAction, f64)> = Vec::with_capacity(acts.len());
        for (&a, &p) in acts.iter().zip(probs) {
            if p <= 0.0 {
                continue;
            }
            let concrete = match abs.concrete(shadow, a) {
                PlayerAction::Fold if real.can_check() => PlayerAction::CheckCall,
                PlayerAction::RaiseTo(x) => match real.raise_bounds() {
                    Some((lo, hi)) => PlayerAction::RaiseTo(x.clamp(lo, hi)),
                    None => PlayerAction::CheckCall,
                },
                other => other,
            };
            match out.iter_mut().find(|(c, _)| *c == concrete) {
                Some((_, q)) => *q += p,
                None => out.push((concrete, p)),
            }
        }
        out
    }

    /// The on-tree shadow, as of the last `strategy` call.
    pub fn shadow(&self) -> &Hand {
        &self.shadow
    }

    /// Advance the shadow to `p`'s turn on the real hand's street, calling
    /// along for players the real hand skips. False when the shadow cannot
    /// get there: `p` is capped, or the shadow's hand or round already ended.
    fn sync_to(&mut self, p: usize, abs: &Abstraction) -> bool {
        // Each auto-call moves the shadow forward; a hand has far fewer
        // turns than this.
        for _ in 0..64 {
            let s = &self.shadow;
            if s.is_terminal() || s.street() > self.real.street() {
                return false;
            }
            let q = s.to_act();
            if q == p && s.street() == self.real.street() {
                return true;
            }
            let skipped = self.real.all_in(q) || self.real.folded(q);
            if !skipped && s.street() == self.real.street() {
                return false;
            }
            self.shadow_apply(AbsAction::CheckCall, abs);
        }
        unreachable!("shadow did not reach seat {p}")
    }

    /// The log-nearest non-all-in raise on the shadow's menu.
    fn nearest_bet(&self, x: u32, abs: &Abstraction) -> AbsAction {
        let distance = |t: u32| ((x.max(1) as f64).ln() - (t.max(1) as f64).ln()).abs();
        abs.abstract_actions(&self.shadow)
            .into_iter()
            .filter(|a| matches!(a, AbsAction::Bet(_)))
            .filter_map(|a| match abs.concrete(&self.shadow, a) {
                PlayerAction::RaiseTo(t) => Some((a, distance(t))),
                _ => None,
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map_or(AbsAction::AllIn, |(a, _)| a)
    }

    fn shadow_apply(&mut self, a: AbsAction, abs: &Abstraction) {
        if let Some(log) = &mut self.log {
            log.push(Observed {
                seat: self.shadow.to_act(),
                action: a,
                shadow: self.shadow.clone(),
                hist: self.hist.clone(),
            });
        }
        let street = self.shadow.street();
        self.shadow.apply(abs.concrete(&self.shadow, a));
        self.hist.push(a.token());
        if self.shadow.street() != street && !self.shadow.is_terminal() {
            self.hist.push(TOKEN_STREET_SEP);
        }
    }
}
