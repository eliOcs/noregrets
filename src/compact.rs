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
//!   weights are each action's share rounded to whole steps (see
//!   [`quantize`]), so they sum to about the step count;
//! - every `BLOCK` records a block starts with a full key. A dense index of
//!   block offsets plus a blob of each block's first key lets a lookup
//!   binary-search blocks touching only index pages, then scan one block.
//!
//! The file is memory-mapped, so only pages that lookups touch become
//! resident, and they are file-backed: the kernel can evict them instead of
//! needing the whole table to fit in RAM.
//!
//! A blueprint may carry its flop/turn bucket table at the end of the file
//! (`buckets` sorted u64 canonical keys, then as many u16 buckets), read in
//! place the same way, so serving needs one file and no table in memory.

use crate::abstraction::{AbsConfig, BucketTable, CanonTable, Centroids, Token};
use crate::cfr::{make_key, Blueprint};
use crate::mirror::Strategies;
use crate::engine::Players;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::io::Write;

const MAGIC: &[u8; 8] = b"NRCOMP03";
/// The format before embedded bucket tables; still read.
const MAGIC_V2: &[u8; 8] = b"NRCOMP02";
const BLOCK: usize = 64;
/// Index entry: data offset u64, first-key offset u32, first-key length u8.
const INDEX_ENTRY: usize = 13;

#[derive(Serialize, Deserialize)]
pub struct Header {
    pub iterations: u64,
    pub num_players: Players,
    pub abs_cfg: AbsConfig,
    pub centroids: Option<Centroids>,
    pub records: u64,
    pub blocks: u64,
    /// Bytes of first-key blob after the index entries.
    pub first_keys: u64,
    /// Entries in the embedded bucket table (0: none).
    pub buckets: u64,
}

#[derive(Serialize, Deserialize)]
struct HeaderV2 {
    iterations: u64,
    num_players: Players,
    abs_cfg: AbsConfig,
    centroids: Option<Centroids>,
    records: u64,
    blocks: u64,
    first_keys: u64,
}

pub struct CompactBlueprint {
    pub header: Header,
    mmap: std::sync::Arc<Mmap>,
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
        .is_ok_and(|_| &magic == MAGIC || &magic == MAGIC_V2)
}

/// Round each action's share to the nearest of `steps` steps and store the
/// step counts. Readers normalize weights, so the bytes need not sum to
/// `steps` exactly. Each share is rounded on its own, up as readily as
/// down, so coarse steps (20 = 5%) remove the residue that MCCFR's average
/// strategy leaves on every action without favouring any of them. If every
/// share rounds to zero, the top action keeps one step. `None` when the
/// weights carry no strategy (all zero).
pub fn quantize(weights: &[f32], steps: u8) -> Option<Vec<u8>> {
    let total: f64 = weights.iter().map(|&w| w.max(0.0) as f64).sum();
    if total <= 0.0 {
        return None;
    }
    let mut q: Vec<u8> =
        weights.iter().map(|&w| (w.max(0.0) as f64 / total * steps as f64).round() as u8).collect();
    if q.iter().all(|&b| b == 0) {
        let top = (0..weights.len()).max_by(|&a, &b| weights[a].total_cmp(&weights[b])).unwrap();
        q[top] = 1;
    }
    Some(q)
}

/// Write `bp` in the compact format, with strategies in `steps` steps (see
/// [`quantize`]) and, if given, its bucket table, which must have been
/// built for `bp`'s abstraction. Infosets without a strategy are dropped;
/// lookups treat them as unseen, as the hash map does.
pub fn write(bp: &Blueprint, path: &str, steps: u8, table: Option<&BucketTable>) -> std::io::Result<u64> {
    if let Some(t) = table {
        if !t.matches(&bp.abs_cfg) || t.centroids != bp.centroids {
            return Err(std::io::Error::other("the bucket table was built for a different abstraction"));
        }
    }
    let mut entries: Vec<(&[u8], Vec<u8>)> = bp
        .strategies
        .iter()
        .filter_map(|(k, v)| quantize(v, steps).map(|q| (k.as_slice(), q)))
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
        buckets: table.map_or(0, |t| t.entries.len() as u64),
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
    if let Some(t) = table {
        let mut entries = t.entries.clone();
        entries.sort_unstable();
        for (k, _) in &entries {
            f.write_all(&k.to_le_bytes())?;
        }
        for (_, b) in &entries {
            f.write_all(&b.to_le_bytes())?;
        }
    }
    f.flush()?;
    Ok(entries.len() as u64)
}

impl CompactBlueprint {
    pub fn open(path: &str) -> std::io::Result<CompactBlueprint> {
        let file = std::fs::File::open(path)?;
        // SAFETY: the blueprint file is written once and never modified
        // while served.
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < 16 || (&mmap[..8] != MAGIC && &mmap[..8] != MAGIC_V2) {
            return Err(std::io::Error::other("not a compact blueprint"));
        }
        let header_len = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let bytes = mmap.get(16..16 + header_len).ok_or_else(|| std::io::Error::other("truncated compact blueprint"))?;
        let header: Header = if &mmap[..8] == MAGIC {
            bincode::deserialize(bytes).map_err(std::io::Error::other)?
        } else {
            let h: HeaderV2 = bincode::deserialize(bytes).map_err(std::io::Error::other)?;
            Header {
                iterations: h.iterations,
                num_players: h.num_players,
                abs_cfg: h.abs_cfg,
                centroids: h.centroids,
                records: h.records,
                blocks: h.blocks,
                first_keys: h.first_keys,
                buckets: 0,
            }
        };
        let index_at = 16 + header_len;
        let keys_at = index_at + INDEX_ENTRY * header.blocks as usize;
        let data_at = keys_at + header.first_keys as usize;
        if data_at + BUCKET_ENTRY * header.buckets as usize > mmap.len() {
            return Err(std::io::Error::other("truncated compact blueprint"));
        }
        Ok(CompactBlueprint {
            header,
            mmap: std::sync::Arc::new(mmap),
            index_at,
            keys_at,
            data_at,
        })
    }

    /// The embedded bucket table, if the file carries one.
    pub fn bucket_table(&self) -> Option<std::sync::Arc<dyn CanonTable>> {
        let n = self.header.buckets as usize;
        (n > 0).then(|| {
            let keys_at = self.mmap.len() - BUCKET_ENTRY * n;
            std::sync::Arc::new(EmbeddedBuckets { mmap: self.mmap.clone(), keys_at, buckets_at: keys_at + 8 * n, n })
                as std::sync::Arc<dyn CanonTable>
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

/// Bytes per embedded bucket-table entry: a u64 key and a u16 bucket.
const BUCKET_ENTRY: usize = 10;

/// A compact blueprint's bucket table, read in place: binary search over
/// the sorted keys, then the bucket at the same position.
struct EmbeddedBuckets {
    mmap: std::sync::Arc<Mmap>,
    keys_at: usize,
    buckets_at: usize,
    n: usize,
}

impl CanonTable for EmbeddedBuckets {
    fn get(&self, ckey: u64) -> Option<u16> {
        let key = |i: usize| u64::from_le_bytes(self.mmap[self.keys_at + 8 * i..][..8].try_into().unwrap());
        let (mut lo, mut hi) = (0, self.n);
        while lo < hi {
            let mid = (lo + hi) / 2;
            match key(mid).cmp(&ckey) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => {
                    let at = self.buckets_at + 2 * mid;
                    return Some(u16::from_le_bytes(self.mmap[at..at + 2].try_into().unwrap()));
                }
            }
        }
        None
    }
}

impl Strategies for CompactBlueprint {
    fn get(&self, bucket: u16, hist: &[Token]) -> Option<std::borrow::Cow<'_, [f32]>> {
        self.lookup(&make_key(bucket, hist))
            .map(|q| std::borrow::Cow::Owned(q.iter().map(|&b| b as f32).collect()))
    }
    fn len(&self) -> usize {
        self.header.records as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::{AbsAction, BetSize};
    use crate::cfr::StrategyMap;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn blueprint(entries: usize, seed: u64) -> Blueprint {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut strategies = StrategyMap::default();
        while strategies.len() < entries {
            let bucket = rng.random_range(0..200u16);
            let hist: Vec<Token> = (0..rng.random_range(0..12))
                .map(|_| AbsAction::Bet(BetSize::new(rng.random_range(0..8)).unwrap()).token())
                .collect();
            let n = rng.random_range(2..7);
            let weights: Vec<f32> = (0..n).map(|_| rng.random::<f32>()).collect();
            strategies.insert(make_key(bucket, &hist).to_vec(), weights);
        }
        Blueprint {
            strategies,
            iterations: 7,
            num_players: Players::SIX,
            abs_cfg: AbsConfig::default(),
            centroids: None,
        }
    }

    fn roundtrip(bp: &Blueprint, name: &str) -> CompactBlueprint {
        roundtrip_with(bp, name, None)
    }

    fn roundtrip_with(bp: &Blueprint, name: &str, table: Option<&BucketTable>) -> CompactBlueprint {
        let path = std::env::temp_dir().join(format!("nr-compact-{name}-{}.bin", std::process::id()));
        let path = path.to_str().unwrap();
        write(bp, path, 255, table).unwrap();
        assert!(is_compact(path));
        let c = CompactBlueprint::open(path).unwrap();
        std::fs::remove_file(path).unwrap();
        c
    }

    fn table(entries: Vec<(u64, u16)>) -> BucketTable {
        BucketTable { cfg: AbsConfig::default(), centroids: None, entries }
    }

    #[test]
    fn an_embedded_bucket_table_reads_back() {
        let bp = blueprint(3_000, 5);
        let mut rng = SmallRng::seed_from_u64(6);
        let mut entries: Vec<(u64, u16)> = (0..20_000).map(|_| (rng.random::<u64>() >> 1, rng.random_range(0..12))).collect();
        entries.sort_unstable();
        entries.dedup_by_key(|e| e.0);
        let c = roundtrip_with(&bp, "table", Some(&table(entries.clone())));
        for (key, weights) in &bp.strategies {
            assert_eq!(c.lookup(key), Some(quantize(weights, 255).unwrap().as_slice()));
        }
        let t = c.bucket_table().expect("an embedded table");
        for &(k, b) in &entries {
            assert_eq!(t.get(k), Some(b));
        }
        for _ in 0..5_000 {
            let k = rng.random::<u64>() >> 1;
            assert_eq!(t.get(k), entries.binary_search_by_key(&k, |e| e.0).ok().map(|i| entries[i].1));
        }
        assert!(roundtrip(&bp, "no-table").bucket_table().is_none());
    }

    #[test]
    fn bucketing_reads_the_embedded_table_for_every_suit_variant() {
        use crate::abstraction::{canonical_cards_key, Abstraction};
        use crate::cards::Hole;
        // A deliberately wrong bucket shows the table, not Monte Carlo,
        // answered: hole 2c 3d, board 4h 7c 9h, and the same with clubs and
        // diamonds swapped.
        let c = |s: &str| crate::cards::parse_card(s).unwrap();
        let (hole, board) = ([c("2c"), c("3d")], [c("4h"), c("7c"), c("9h")]);
        let bp = roundtrip_with(&blueprint(100, 7), "abs", Some(&table(vec![(canonical_cards_key(hole, &board), 11)])));
        let abs = Abstraction::with_centroids(AbsConfig::default(), None).with_table(bp.bucket_table().unwrap());
        let mut rng = SmallRng::seed_from_u64(8);
        assert_eq!(abs.bucket(Hole::try_from(hole).unwrap(), &board, &mut rng), 11);
        assert_eq!(abs.bucket(Hole::try_from([c("2d"), c("3c")]).unwrap(), &[c("4h"), c("7d"), c("9h")], &mut rng), 11);
        assert_eq!(abs.canon_len(), 0, "table hits are not copied into the cache");
    }

    #[test]
    fn a_table_for_another_abstraction_is_refused() {
        let bp = blueprint(100, 9);
        let mut t = table(vec![(1, 1)]);
        t.cfg.postflop_buckets = t.cfg.postflop_buckets.checked_add(1).unwrap();
        let path = std::env::temp_dir().join(format!("nr-compact-refused-{}.bin", std::process::id()));
        assert!(write(&bp, path.to_str().unwrap(), 255, Some(&t)).is_err());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn version_2_files_still_open() {
        let bp = blueprint(2_000, 10);
        let path = std::env::temp_dir().join(format!("nr-compact-v2-{}.bin", std::process::id()));
        let path = path.to_str().unwrap();
        write(&bp, path, 255, None).unwrap();
        // Rewrite the header as version 2 wrote it, without the table count.
        let bytes = std::fs::read(path).unwrap();
        let len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let h: Header = bincode::deserialize(&bytes[16..16 + len]).unwrap();
        let v2 = bincode::serialize(&HeaderV2 {
            iterations: h.iterations,
            num_players: h.num_players,
            abs_cfg: h.abs_cfg,
            centroids: h.centroids,
            records: h.records,
            blocks: h.blocks,
            first_keys: h.first_keys,
        })
        .unwrap();
        let mut old = MAGIC_V2.to_vec();
        old.extend_from_slice(&(v2.len() as u64).to_le_bytes());
        old.extend_from_slice(&v2);
        old.extend_from_slice(&bytes[16 + len..]);
        std::fs::write(path, old).unwrap();
        assert!(is_compact(path));
        let c = CompactBlueprint::open(path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert!(c.bucket_table().is_none());
        for (key, weights) in &bp.strategies {
            assert_eq!(c.lookup(key), Some(quantize(weights, 255).unwrap().as_slice()));
        }
    }

    #[test]
    fn every_infoset_reads_back_quantized() {
        let bp = blueprint(5_000, 1);
        let c = roundtrip(&bp, "all");
        assert_eq!(c.header.records, 5_000);
        assert_eq!(c.header.iterations, 7);
        for (key, weights) in &bp.strategies {
            assert_eq!(c.lookup(key), Some(quantize(weights, 255).unwrap().as_slice()), "key {key:?}");
        }
    }

    #[test]
    fn unknown_infosets_are_absent() {
        let bp = blueprint(2_000, 2);
        let c = roundtrip(&bp, "absent");
        let mut rng = SmallRng::seed_from_u64(3);
        for _ in 0..5_000 {
            let bucket = rng.random_range(0..300u16);
            let key = vec![bucket as u8, (bucket >> 8) as u8, rng.random_range(0..16u8)];
            assert_eq!(c.lookup(&key).is_some(), bp.strategies.contains_key(&key));
        }
        assert_eq!(c.lookup(&[]), None, "before the first key");
        assert_eq!(c.lookup(&[255; 40]), None, "after the last key");
    }

    #[test]
    fn quantized_shares_round_to_the_nearest_step_and_keep_the_top_action() {
        let mut rng = SmallRng::seed_from_u64(4);
        for steps in [255u8, 20] {
            for _ in 0..1_000 {
                let w: Vec<f32> = (0..rng.random_range(1..8)).map(|_| rng.random::<f32>().powi(3)).collect();
                let total: f32 = w.iter().sum();
                let q = quantize(&w, steps).unwrap();
                for (&x, &b) in w.iter().zip(&q) {
                    assert!((x / total * steps as f32 - b as f32).abs() <= 0.5 + 1e-4, "{w:?} -> {q:?}");
                }
                let top = |v: &[f32]| (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
                assert!(q[top(&w)] > 0);
            }
        }
        assert_eq!(quantize(&[0.0, 0.0], 20), None);
        assert_eq!(quantize(&[1.0; 50], 20).unwrap().iter().filter(|&&b| b > 0).count(), 1);
    }

    #[test]
    fn coarse_steps_drop_residue_and_round_small_shares_both_ways() {
        // Big blind with K-J offsuit facing a 3bb button open, 800M blueprint:
        // fold, call, raises to 6, 9.1 and 15.4bb, all-in. The 2.3% shove and
        // 2% raise round down; the 3.5% raise rounds up.
        let w = [0.071, 0.8, 0.035, 0.051, 0.02, 0.023];
        assert_eq!(quantize(&w, 20).unwrap(), vec![1, 16, 1, 1, 0, 0]);
    }
}
