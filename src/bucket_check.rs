//! Does lazy bucketing, as `serve` does it without a table, agree with the
//! bucket table the blueprint was trained with? Buckets random flop and
//! turn hands from scratch and compares them with the table.

use crate::abstraction::{canonical_cards_key, Abstraction, BucketTable};
use crate::cards::{fresh_deck, Hole};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::HashMap;

/// Mismatch shares for flop and turn.
pub fn run(table: &BucketTable, hands: u64, seed: u64) -> [(u64, u64); 2] {
    let abs = Abstraction::with_centroids(table.cfg.clone(), table.centroids.clone());
    let want: HashMap<u64, u16> = table.entries.iter().copied().collect();
    let rows: Vec<(usize, bool)> = (0..hands)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed ^ i.wrapping_mul(0x2545_F491_4F6C_DD1D));
            let mut deck = fresh_deck();
            deck.shuffle(&mut rng);
            let street = (i % 2) as usize;
            let hole = Hole::new(deck[0], deck[1]).expect("a shuffled deck holds each card once");
            let board = &deck[2..5 + street];
            let got = abs.bucket(hole, board, &mut rng);
            let key = canonical_cards_key(*hole, board);
            (street, want[&key] != got)
        })
        .collect();
    let mut out = [(0, 0); 2];
    for (s, miss) in rows {
        out[s].0 += miss as u64;
        out[s].1 += 1;
    }
    out
}
