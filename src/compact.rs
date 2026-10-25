//! A compact, memory-mapped, read-only blueprint for serving.
//!
//! The training blueprint is a hash map of boxed keys to `Vec<f32>`; once
//! loaded, per-entry allocation overhead dominates its memory. This format
//! stores the same strategies sorted by infoset key, in blocks like an
//! SSTable:
//!
//! - each record is `[shared u8][suffix_len u8][suffix][n u8][n x u8]`, where
//!   `shared` is the prefix length it shares with the previous key in its
//!   block (sorted keys share bucket and history prefixes), and the action
//!   weights are quantized to bytes summing to 255;
//! - every `BLOCK` records a block starts with a full key. A dense index of
//!   block offsets plus a blob of each block's first key lets a lookup
//!   binary-search blocks touching only index pages, then scan one block.
//!
//! The file is memory-mapped, so only pages that lookups touch become
//! resident, and they are file-backed: the kernel can evict them instead of
//! needing the whole table to fit in RAM.

use crate::abstraction::{AbsConfig, Centroids};
use crate::cfr::{make_key, Blueprint};
use crate::mirror::Strategies;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::io::Write;

const MAGIC: &[u8; 8] = b"NRCOMP02";
const BLOCK: usize = 64;
/// Index entry: data offset u64, first-key offset u32, first-key length u8.
const INDEX_ENTRY: usize = 13;

#[derive(Serialize, Deserialize)]
pub struct Header {
    pub iterations: u64,
    pub num_players: usize,
    pub abs_cfg: AbsConfig,
    pub centroids: Option<Centroids>,
    pub records: u64,
    pub blocks: u64,
    /// Bytes of first-key blob after the index entries.
    pub first_keys: u64,
}

pub struct CompactBlueprint {
    pub header: Header,
    mmap: Mmap,
    /// Byte offsets of the index entries (data offset u64, first-key
    /// offset u32, first-key length u8), the first-key blob and the data.
    index_at: usize,
    keys_at: usize,
    data_at: usize,
}

/// Is the file at `path` a compact blueprint?
pub fn is_compact(path: &str) -> bool {
    let mut magic = [0u8; 8];
    std::fs::File::open(path)
        .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut magic))
        .is_ok_and(|_| &magic == MAGIC)
}

/// Quantize weights to bytes summing to 255 by largest remainder. `None`
/// when the weights carry no strategy (all zero).
pub fn quantize(weights: &[f32]) -> Option<Vec<u8>> {
    let total: f64 = weights.iter().map(|&w| w.max(0.0) as f64).sum();
    if total <= 0.0 {
        return None;
    }
    let scaled: Vec<f64> = weights.iter().map(|&w| w.max(0.0) as f64 / total * 255.0).collect();
    let mut q: Vec<u8> = scaled.iter().map(|x| x.floor() as u8).collect();
    let short = 255 - q.iter().map(|&b| b as u32).sum::<u32>();
    let mut order: Vec<usize> = (0..q.len()).collect();
    order.sort_by(|&a, &b| (scaled[b] - scaled[b].floor()).total_cmp(&(scaled[a] - scaled[a].floor())));
    for &i in order.iter().take(short as usize) {
        q[i] += 1;
    }
    Some(q)
}

/// Write `bp` in the compact format. Infosets without a strategy are
/// dropped; lookups treat them as unseen, as the hash map does.
pub fn write(bp: &Blueprint, path: &str) -> std::io::Result<u64> {
    let mut entries: Vec<(&[u8], Vec<u8>)> = bp
        .strategies
        .iter()
        .filter_map(|(k, v)| quantize(v).map(|q| (k.as_slice(), q)))
        .collect();
    entries.sort_unstable_by(|a, b| a.0.cmp(b.0));

    let mut data: Vec<u8> = Vec::new();
    let mut index: Vec<(u64, u32, u8)> = Vec::with_capacity(entries.len() / BLOCK + 1);
    let mut first_keys: Vec<u8> = Vec::new();
    let mut prev: &[u8] = &[];
    for (i, (key, q)) in entries.iter().enumerate() {
        let shared = if i % BLOCK == 0 {
            index.push((data.len() as u64, first_keys.len() as u32, key.len() as u8));
            first_keys.extend_from_slice(key);
            0
        } else {
            key.iter().zip(prev).take_while(|(a, b)| a == b).count().min(255)
        };
        let suffix = &key[shared..];
        assert!(suffix.len() <= 255 && q.len() <= 255, "infoset key or menu too long");
        data.push(shared as u8);
        data.push(suffix.len() as u8);
        data.extend_from_slice(suffix);
        data.push(q.len() as u8);
        data.extend_from_slice(q);
        prev = key;
    }

    let header = bincode::serialize(&Header {
        iterations: bp.iterations,
        num_players: bp.num_players,
        abs_cfg: bp.abs_cfg.clone(),
        centroids: bp.centroids.clone(),
        records: entries.len() as u64,
        blocks: index.len() as u64,
        first_keys: first_keys.len() as u64,
    })
    .map_err(std::io::Error::other)?;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(MAGIC)?;
    f.write_all(&(header.len() as u64).to_le_bytes())?;
    f.write_all(&header)?;
    for (data_off, key_off, key_len) in &index {
        f.write_all(&data_off.to_le_bytes())?;
        f.write_all(&key_off.to_le_bytes())?;
        f.write_all(&[*key_len])?;
    }
    f.write_all(&first_keys)?;
    f.write_all(&data)?;
    f.flush()?;
    Ok(entries.len() as u64)
}

impl CompactBlueprint {
    pub fn open(path: &str) -> std::io::Result<CompactBlueprint> {
        let file = std::fs::File::open(path)?;
        // SAFETY: the blueprint file is written once and never modified
        // while served.
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < 16 || &mmap[..8] != MAGIC {
            return Err(std::io::Error::other("not a compact blueprint"));
        }
        let header_len = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let header: Header =
            bincode::deserialize(&mmap[16..16 + header_len]).map_err(std::io::Error::other)?;
        let index_at = 16 + header_len;
        let keys_at = index_at + INDEX_ENTRY * header.blocks as usize;
        let data_at = keys_at + header.first_keys as usize;
        if data_at > mmap.len() {
            return Err(std::io::Error::other("truncated compact blueprint"));
        }
        Ok(CompactBlueprint {
            header,
            mmap,
            index_at,
            keys_at,
            data_at,
        })
    }

    fn index_entry(&self, b: usize) -> &[u8] {
        let at = self.index_at + INDEX_ENTRY * b;
        &self.mmap[at..at + INDEX_ENTRY]
    }

    fn block_start(&self, b: usize) -> usize {
        self.data_at + u64::from_le_bytes(self.index_entry(b)[..8].try_into().unwrap()) as usize
    }

    /// The full key of a block's first record, from the dense index.
    fn first_key(&self, b: usize) -> &[u8] {
        let e = self.index_entry(b);
        let at = self.keys_at + u32::from_le_bytes(e[8..12].try_into().unwrap()) as usize;
        &self.mmap[at..at + e[12] as usize]
    }

    pub fn lookup(&self, key: &[u8]) -> Option<&[u8]> {
        let blocks = self.header.blocks as usize;
        // Last block whose first key is <= key.
        let (mut lo, mut hi) = (0usize, blocks);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.first_key(mid) <= key {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let b = lo.checked_sub(1)?;
        let in_block = if b + 1 == blocks {
            self.header.records as usize - b * BLOCK
        } else {
            BLOCK
        };

        let mut at = self.block_start(b);
        let mut cur: Vec<u8> = Vec::with_capacity(key.len() + 8);
        for _ in 0..in_block {
            let shared = self.mmap[at] as usize;
            let len = self.mmap[at + 1] as usize;
            cur.truncate(shared);
            cur.extend_from_slice(&self.mmap[at + 2..at + 2 + len]);
            let n_at = at + 2 + len;
            let n = self.mmap[n_at] as usize;
            match cur.as_slice().cmp(key) {
                Ordering::Equal => return Some(&self.mmap[n_at + 1..n_at + 1 + n]),
                Ordering::Greater => return None,
                Ordering::Less => at = n_at + 1 + n,
            }
        }
        None
    }
}

impl Strategies for CompactBlueprint {
    fn get(&self, bucket: u16, hist: &[u8]) -> Option<Vec<f32>> {
        self.lookup(&make_key(bucket, hist))
            .map(|q| q.iter().map(|&b| b as f32).collect())
    }
    fn len(&self) -> usize {
        self.header.records as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfr::StrategyMap;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn blueprint(entries: usize, seed: u64) -> Blueprint {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut strategies = StrategyMap::default();
        while strategies.len() < entries {
            let bucket = rng.random_range(0..200u16);
            let hist: Vec<u8> = (0..rng.random_range(0..12)).map(|_| rng.random_range(0..8)).collect();
            let n = rng.random_range(2..7);
            let weights: Vec<f32> = (0..n).map(|_| rng.random::<f32>()).collect();
            strategies.insert(make_key(bucket, &hist).to_vec(), weights);
        }
        Blueprint {
            strategies,
            iterations: 7,
            num_players: 6,
            abs_cfg: AbsConfig::default(),
            centroids: None,
        }
    }

    fn roundtrip(bp: &Blueprint, name: &str) -> CompactBlueprint {
        let path = std::env::temp_dir().join(format!("nr-compact-{name}-{}.bin", std::process::id()));
        let path = path.to_str().unwrap();
        write(bp, path).unwrap();
        assert!(is_compact(path));
        let c = CompactBlueprint::open(path).unwrap();
        std::fs::remove_file(path).unwrap();
        c
    }

    #[test]
    fn every_infoset_reads_back_quantized() {
        let bp = blueprint(5_000, 1);
        let c = roundtrip(&bp, "all");
        assert_eq!(c.header.records, 5_000);
        assert_eq!(c.header.iterations, 7);
        for (key, weights) in &bp.strategies {
            assert_eq!(c.lookup(key), Some(quantize(weights).unwrap().as_slice()), "key {key:?}");
        }
    }

    #[test]
    fn unknown_infosets_are_absent() {
        let bp = blueprint(2_000, 2);
        let c = roundtrip(&bp, "absent");
        let mut rng = SmallRng::seed_from_u64(3);
        for _ in 0..5_000 {
            let key = make_key(rng.random_range(0..300u16), &[rng.random_range(0..16)]).to_vec();
            assert_eq!(c.lookup(&key).is_some(), bp.strategies.contains_key(&key));
        }
        assert_eq!(c.lookup(&[]), None, "before the first key");
        assert_eq!(c.lookup(&[255; 40]), None, "after the last key");
    }

    #[test]
    fn quantized_weights_sum_to_255_and_keep_the_top_action() {
        let mut rng = SmallRng::seed_from_u64(4);
        for _ in 0..1_000 {
            let w: Vec<f32> = (0..rng.random_range(1..8)).map(|_| rng.random::<f32>().powi(3)).collect();
            let q = quantize(&w).unwrap();
            assert_eq!(q.iter().map(|&b| b as u32).sum::<u32>(), 255);
            let top = |v: &[f32]| (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
            assert!(q[top(&w)] > 0);
        }
        assert_eq!(quantize(&[0.0, 0.0]), None);
    }
}
