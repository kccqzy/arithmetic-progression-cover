//! Minimum representations of a union of arithmetic progressions (APs).
//!
//! An AP is `(s, d, n) = {s + k*d : 0 <= k < n}`, with `n = None` meaning
//! infinite.  Given input APs whose union is S, we compute:
//!
//!   (a) a minimum-size family of APs, each a subset of S, whose union is S,
//!       with overlap freely allowed;
//!   (b) the same, under the non-overlap rule: for any two distinct chosen
//!       sets A, B, either max(A) < min(B), or max(B) < min(A), or both are
//!       infinite.
//!
//! Method.  S normalizes to a finite prefix F = S ∩ (-inf, T) plus a periodic
//! tail: for x >= T, x ∈ S iff x mod P ∈ R, where P = lcm of the infinite
//! inputs' differences.  R is represented as a byte indicator array Rb of
//! length P, never as a per-residue collection.
//!
//!   (a) becomes exact set cover over ground set F plus tail tokens.
//!   (b) decomposes at a cut c (the earliest infinite start): the prefix is
//!       partitioned into consecutive runs by a greedy pass (provably
//!       optimal), the suffix is covered by infinite APs only; all cuts are
//!       scanned.
//!
//! Scalability.  Large periods are kept tractable by working on whole arrays:
//!   1. Valid residue classes (b mod d whose full lift to Z/P lies in R) are
//!      computed for all divisors d of P by DP down the divisor lattice: the
//!      validity array at level d is the byte-wise AND of the q chunks of
//!      the array at level d*q.
//!   2. Dominated candidates are pruned wholesale.  Candidate (d,b,e) is
//!      dominated by (dp,bp,ep) when dp | d, b ≡ bp (mod dp) and e >= ep.
//!      For classes that do not extend into the prefix (e >= T), e >= ep
//!      holds automatically, so the dominance mask at level d is the OR over
//!      primes q | d of the tiled validity array of level d/q.  The few
//!      classes that do extend into the prefix (at most |F| per divisor) are
//!      checked individually against actual starts.
//!   3. The tail ground set is compressed by coverage signature: residues of
//!      R covered by exactly the same candidates are one token, so the
//!      set-cover universe is the number of distinct signatures, not |R|.
//!
//! Modeling restriction: infinite APs in solutions use differences dividing P.
//! Every input satisfies this and every full residue class is expressible
//! this way; an AP with d not dividing P covers only a strict
//! sub-progression of each class it meets.  We have found no instance where
//! such tilings beat the divisor solutions, but we do not have a proof that
//! they never do.

use std::collections::{BTreeMap, BTreeSet, HashMap};

/// (s, d, n): start, difference, count.  `n = None` means an infinite AP.
pub type AP = (i64, i64, Option<i64>);

// ----------------------------------------------------------------- bitset

/// Fixed-width packed bitset of `n_words * 64` bits, used as the ground-set
/// mask inside `set_cover`.  All bitsets passed through one `set_cover` call
/// must share the same word count (= `n_bits.div_ceil(64)` for that call's
/// `n_univ`); operations below assume that and pair up words by index.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Bitset {
    words: Vec<u64>,
}

impl Bitset {
    pub fn empty(n_bits: usize) -> Self {
        Self { words: vec![0u64; n_bits.div_ceil(64)] }
    }

    /// Bits `0..n_bits` set, the rest zero.
    pub fn full(n_bits: usize) -> Self {
        let mut bs = Self { words: vec![!0u64; n_bits.div_ceil(64)] };
        let tail = n_bits % 64;
        if tail != 0 {
            *bs.words.last_mut().unwrap() = (!0u64) >> (64 - tail);
        }
        bs
    }

    pub fn get(&self, i: usize) -> bool {
        (self.words[i / 64] >> (i % 64)) & 1 != 0
    }

    pub fn insert(&mut self, i: usize) {
        self.words[i / 64] |= 1u64 << (i % 64);
    }

    pub fn count(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// `self | other`.
    pub fn union(&self, other: &Self) -> Self {
        Self {
            words: self.words.iter().zip(&other.words).map(|(&a, &b)| a | b).collect(),
        }
    }

    /// In-place `self |= other`.
    pub fn union_assign(&mut self, other: &Self) {
        for (a, &b) in self.words.iter_mut().zip(&other.words) {
            *a |= b;
        }
    }

    /// `self & !other` (bits in `self` not in `other`).
    pub fn difference(&self, other: &Self) -> Self {
        Self {
            words: self.words.iter().zip(&other.words).map(|(&a, &b)| a & !b).collect(),
        }
    }

    /// In-place `self &= !other`.
    pub fn subtract_assign(&mut self, other: &Self) {
        for (a, &b) in self.words.iter_mut().zip(&other.words) {
            *a &= !b;
        }
    }

    /// True iff `self` is a (possibly improper) subset of `other`.
    pub fn is_subset_of(&self, other: &Self) -> bool {
        self.words.iter().zip(&other.words).all(|(&a, &b)| a & !b == 0)
    }

    /// `(self & other).count()`, without allocating the intersection.
    pub fn intersect_count(&self, other: &Self) -> u32 {
        self.words.iter().zip(&other.words).map(|(&a, &b)| (a & b).count_ones()).sum()
    }

    /// Overwrite `self` with `other`'s bits in place, reusing the existing
    /// allocation.  Both bitsets must share the same word count.
    pub fn copy_from(&mut self, other: &Self) {
        self.words.copy_from_slice(&other.words);
    }
}

// ----------------------------------------------------------- normalization

pub fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

pub fn lcm(a: i64, b: i64) -> i64 {
    a / gcd(a, b) * b
}

/// Return `(F, T, P, Rb)`: sorted prefix, threshold, period, and the tail
/// residue indicator (`Rb[r] == 1` iff residue `r mod P` belongs to S's tail).
pub fn normalize(inputs: &[AP]) -> (BTreeSet<i64>, i64, i64, Vec<u8>) {
    assert!(!inputs.is_empty());
    for &(_, d, n) in inputs {
        assert!(d >= 1 && n.is_none_or(|n| n >= 1));
    }
    let infs: Vec<(i64, i64)> = inputs
        .iter()
        .filter_map(|&(s, d, n)| if n.is_none() { Some((s, d)) } else { None })
        .collect();
    let fins: Vec<(i64, i64, i64)> = inputs
        .iter()
        .filter_map(|&(s, d, n)| n.map(|nn| (s, d, nn)))
        .collect();
    let (p, t, rb): (i64, i64, Vec<u8>) = if !infs.is_empty() {
        let p = infs.iter().map(|&(_, d)| d).fold(1, lcm);
        let mut t = infs.iter().map(|&(s, _)| s).max().unwrap();
        if !fins.is_empty() {
            let max_fin = fins.iter().map(|&(s, d, n)| s + (n - 1) * d).max().unwrap();
            t = t.max(1 + max_fin);
        }
        let mut rb = vec![0u8; p as usize];
        for &(s, d) in &infs {
            let mut r = s.rem_euclid(d) as usize;
            while r < p as usize {
                rb[r] = 1;
                r += d as usize;
            }
        }
        (p, t, rb)
    } else {
        let max_fin = fins.iter().map(|&(s, d, n)| s + (n - 1) * d).max().unwrap();
        (1, 1 + max_fin, vec![0u8; 1])
    };
    let mut f: BTreeSet<i64> = BTreeSet::new();
    for &(s, d, n) in &fins {
        let mut x = s;
        for _ in 0..n {
            if x < t {
                f.insert(x);
            }
            x += d;
        }
    }
    for &(s, d) in &infs {
        let mut x = s;
        while x < t {
            f.insert(x);
            x += d;
        }
    }
    (f, t, p, rb)
}

// -------------------------------------------------------------- candidates

pub fn divisors(p: i64) -> BTreeSet<i64> {
    let mut out: BTreeSet<i64> = BTreeSet::new();
    let mut d: i64 = 1;
    while d * d <= p {
        if p % d == 0 {
            out.insert(d);
            out.insert(p / d);
        }
        d += 1;
    }
    out
}

pub fn prime_factors(mut n: i64) -> Vec<i64> {
    let mut out = Vec::new();
    let mut p: i64 = 2;
    while p * p <= n {
        if n % p == 0 {
            out.push(p);
            while n % p == 0 {
                n /= p;
            }
        }
        p += 1;
    }
    if n > 1 {
        out.push(n);
    }
    out
}

/// `flags[d][b] == 1` iff the full lift of `b mod d` to Z/P lies inside R,
/// for every divisor d of P.  DP down the divisor lattice: the array at d
/// is the byte-wise AND of the q chunks of the array at d*q (any prime q
/// dividing P/d).
pub fn valid_classes(p: i64, rb: &[u8]) -> BTreeMap<i64, Vec<u8>> {
    let mut flags: BTreeMap<i64, Vec<u8>> = BTreeMap::new();
    flags.insert(p, rb.to_vec());
    for d in divisors(p).into_iter().rev().skip(1) {
        let q = prime_factors(p / d)[0];
        let d_usz = d as usize;
        let q_usz = q as usize;
        let par = flags[&(d * q)].clone();
        let mut v = par[..d_usz].to_vec();
        for k in 1..q_usz {
            let chunk = &par[k * d_usz..(k + 1) * d_usz];
            for i in 0..d_usz {
                v[i] &= chunk[i];
            }
        }
        flags.insert(d, v);
    }
    flags
}

/// Leftmost start of the AP with difference `d` in class `b mod d`,
/// extended leftward through F.
fn class_start(
    d: i64,
    b: i64,
    t: i64,
    f: &BTreeSet<i64>,
    e_memo: &mut HashMap<(i64, i64), i64>,
) -> i64 {
    if let Some(&e) = e_memo.get(&(d, b)) {
        return e;
    }
    let mut e = t + (b - t).rem_euclid(d);
    while f.contains(&(e - d)) {
        e -= d;
    }
    e_memo.insert((d, b), e);
    e
}

/// Non-dominated maximal infinite APs inside S with difference dividing P,
/// as `(d, b, e)`: difference, class mod d, leftmost start.
pub fn infinite_candidates(
    f: &BTreeSet<i64>,
    t: i64,
    p: i64,
    rb: &[u8],
) -> Vec<(i64, i64, i64)> {
    let flags = valid_classes(p, rb);
    let mut e_memo: HashMap<(i64, i64), i64> = HashMap::new();
    let mut out: Vec<(i64, i64, i64)> = Vec::new();
    // Descending divisor order matches Python's insertion-order dict iteration.
    for (&d, flags_d) in flags.iter().rev() {
        let d_usz = d as usize;
        // classes whose chain can extend into the prefix (at most |F|)
        let mut ext: BTreeSet<i64> = BTreeSet::new();
        for &y in f.iter() {
            if y >= t - d {
                ext.insert(y.rem_euclid(d));
            }
        }
        // dom[b] = 1 iff some prime q | d has (d/q, b mod (d/q)) valid.
        let mut dom = vec![0u8; d_usz];
        for q in prime_factors(d) {
            let dp = d / q;
            let dp_usz = dp as usize;
            let q_usz = q as usize;
            let src = &flags[&dp];
            for k in 0..q_usz {
                let dst = &mut dom[k * dp_usz..(k + 1) * dp_usz];
                for i in 0..dp_usz {
                    dst[i] |= src[i];
                }
            }
        }
        // Unextendable survivors: valid and not dominated at the byte level.
        for b in 0..d_usz {
            if flags_d[b] & !dom[b] != 0 {
                let b_i = b as i64;
                if !ext.contains(&b_i) {
                    let e = class_start(d, b_i, t, f, &mut e_memo);
                    out.push((d, b_i, e));
                }
            }
        }
        // Extendable classes: exact per-divisor dominance test.
        let divs = divisors(d);
        let proper_divs = divs.iter().rev().skip(1).rev();
        for &b in ext.iter() {
            if flags_d[b as usize] != 0 {
                let e = class_start(d, b, t, f, &mut e_memo);
                let mut dominated = false;
                for &dp in proper_divs.clone() {
                    let bp = b.rem_euclid(dp);
                    if flags[&dp][bp as usize] != 0
                        && e >= class_start(dp, bp, t, f, &mut e_memo)
                    {
                        dominated = true;
                        break;
                    }
                }
                if !dominated {
                    out.push((d, b, e));
                }
            }
        }
    }
    out
}

/// Coverage-signature compression of the tail: residues of R covered by
/// exactly the same candidates are one token.  Returns the sorted list of
/// distinct masks; bit i of a mask marks coverage by candidate i.
pub fn tail_tokens(cands: &[(i64, i64, i64)], p: i64, _rb: &[u8]) -> Vec<u64> {
    if cands.is_empty() {
        return Vec::new();
    }
    assert!(cands.len() <= 64, "tail_tokens: >64 candidates not supported");
    let p_usz = p as usize;
    let mut cover: Vec<u64> = vec![0u64; p_usz];
    for (i, &(d, b, _)) in cands.iter().enumerate() {
        let mut r = b as usize;
        while r < p_usz {
            cover[r] |= 1u64 << i;
            r += d as usize;
        }
    }
    cover.retain(|&c| c != 0);
    cover.sort_unstable();
    cover.dedup();
    cover
}

/// For each candidate, the set of tokens it covers, packed as a Bitset of
/// `toks.len()` bits.
pub fn token_masks(inf_c: &[(i64, i64, i64)], toks: &[u64]) -> Vec<Bitset> {
    let n = inf_c.len();
    let n_toks = toks.len();
    let mut tmask: Vec<Bitset> = (0..n).map(|_| Bitset::empty(n_toks)).collect();
    for (j, &t) in toks.iter().enumerate() {
        for (i, tm) in tmask.iter_mut().enumerate() {
            if (t >> i) & 1 != 0 {
                tm.insert(j);
            }
        }
    }
    tmask
}

/// Maximal APs (including singletons) contained in F, as `(s, d, n)`.
pub fn finite_candidates(f: &BTreeSet<i64>) -> BTreeSet<(i64, i64, i64)> {
    let mut out: BTreeSet<(i64, i64, i64)> = BTreeSet::new();
    for &x in f {
        out.insert((x, 1, 1));
    }
    for (i, s) in f.iter().copied().enumerate() {

        for t in f.iter().skip(i + 1)  {
            let d = t - s;
            if f.contains(&(s - d)) {
                // not left-maximal
                continue;
            }
            let mut n = 2;
            let mut x = t + d;
            while f.contains(&x) {
                n += 1;
                x += d;
            }
            out.insert((s, d, n));
        }
    }
    out
}

// ----------------------------------------------------------- exact cover(s)

/// Exact minimum set cover of `{0..n_univ-1}`; `cands` is `[(mask, payload)]`.
/// Returns chosen payloads, or `None` if infeasible.  Branch and bound over
/// bitmasks with greedy upper bound, dominance pruning, and memoization.
pub fn set_cover(n_univ: usize, cands: Vec<(Bitset, AP)>) -> Option<Vec<AP>> {
    let full = Bitset::full(n_univ);
    if full.is_empty() {
        return Some(Vec::new());
    }

    // First occurrence of each non-zero mask wins, preserving insertion order.
    let mut by_mask: HashMap<&Bitset, &AP> = HashMap::new();
    let mut order: Vec<&Bitset> = Vec::new();
    for (ref m, ref p) in cands.iter() {
        if !m.is_empty() && !by_mask.contains_key(&m) {
            order.push(m);
            by_mask.insert(m, p);
        }
    }

    // Stable sort by descending popcount.
    order.sort_by_cached_key(|m| std::cmp::Reverse(m.count()));

    let mut kept: Vec<&Bitset> = Vec::new();
    for m in order {
        if !kept.iter().any(|k| m.is_subset_of(k)) {
            kept.push(m);
        }
    }

    let mut union = Bitset::empty(n_univ);
    for k in &kept {
        union.union_assign(k);
    }
    if union != full {
        return None;
    }

    let sets = kept;
    let elem_sets: Vec<Vec<usize>> = (0..n_univ)
        .map(|e| {
            sets.iter()
                .enumerate()
                .filter_map(|(i, m)| if m.get(e) { Some(i) } else { None })
                .collect()
        })
        .collect();

    // Greedy upper bound.
    let mut unc = full.clone();
    let mut chosen: Vec<usize> = Vec::new();
    while !unc.is_empty() {
        let i = (0..sets.len())
            .max_by_key(|&i| sets[i].intersect_count(&unc))
            .unwrap();
        chosen.push(i);
        unc.subtract_assign(sets[i]);
    }
    let mut best = chosen;

    let maxsz = sets.iter().map(|m| m.count()).max().unwrap() as usize;
    let mut memo: HashMap<Bitset, usize> = HashMap::new();

    fn dfs(
        unc: &mut Bitset,
        chosen: &mut Vec<usize>,
        best: &mut Vec<usize>,
        sets: &[&Bitset],
        elem_sets: &[Vec<usize>],
        n_univ: usize,
        maxsz: usize,
        memo: &mut HashMap<Bitset, usize>,
    ) {
        if unc.is_empty() {
            if chosen.len() < best.len() {
                *best = chosen.clone();
            }
            return;
        }
        let cnt = unc.count() as usize;
        let lb = cnt.div_ceil(maxsz);
        if chosen.len() + lb >= best.len() {
            return;
        }
        if let Some(&seen) = memo.get(unc) {
            if seen <= chosen.len() {
                return;
            }
        }
        memo.insert(unc.clone(), chosen.len());

        let e = (0..n_univ)
            .filter(|&e| unc.get(e))
            .min_by_key(|&e| elem_sets[e].len())
            .unwrap();

        let mut keyed: Vec<(u32, usize)> = elem_sets[e]
            .iter()
            .map(|&i| (sets[i].intersect_count(unc), i))
            .collect();
        keyed.sort_by_key(|&(k, _)| std::cmp::Reverse(k));

        let snapshot = unc.clone();
        for (_, i) in keyed {
            chosen.push(i);
            unc.subtract_assign(sets[i]);
            dfs(unc, chosen, best, sets, elem_sets, n_univ, maxsz, memo);
            chosen.pop();
            unc.copy_from(&snapshot);
        }
    }

    let mut unc = full.clone();
    let mut chosen2 = Vec::new();
    dfs(
        &mut unc,
        &mut chosen2,
        &mut best,
        &sets,
        &elem_sets,
        n_univ,
        maxsz,
        &mut memo,
    );

    Some(best.iter().map(|&i| by_mask[&sets[i]]).copied().collect())
}

// -------------------------------------------------------- formulation (a)

/// Minimum representation, overlap allowed.
pub fn solve_a(inputs: &[AP]) -> Vec<AP> {
    let (f, t, p, rb) = normalize(inputs);
    let inf_c = infinite_candidates(&f, t, p, &rb);
    let toks = tail_tokens(&inf_c, p, &rb);
    let tmask = token_masks(&inf_c, &toks);
    let n_f = f.len();
    let n_toks = toks.len();
    let n_univ = n_f + n_toks;
    let eidx: HashMap<i64, usize> = f.iter().enumerate().map(|(i, &x)| (x, i)).collect();
    let mut cands: Vec<(Bitset, AP)> = Vec::new();
    for (i, &(d, _b, e)) in inf_c.iter().enumerate() {
        let mut m = Bitset::empty(n_univ);
        for j in 0..n_toks {
            if tmask[i].get(j) {
                m.insert(n_f + j);
            }
        }
        let mut x = e;
        while x < t {
            m.insert(eidx[&x]);
            x += d;
        }
        cands.push((m, (e, d, None)));
    }
    for (s, d, n) in finite_candidates(&f) {
        let mut m = Bitset::empty(n_univ);
        let mut x = s;
        for _ in 0..n {
            m.insert(eidx[&x]);
            x += d;
        }
        cands.push((m, (s, d, Some(n))));
    }
    set_cover(n_univ, cands).expect("solve_a: cover infeasible")
}

// -------------------------------------------------------- formulation (b)

/// Optimal partition of sorted `f` into fewest consecutive AP runs.
pub fn greedy_runs(f: &[i64]) -> Vec<AP> {
    let mut runs: Vec<AP> = Vec::new();
    let mut i = 0;
    while i < f.len() {
        let (d, j) = if i + 1 < f.len() {
            let d = f[i + 1] - f[i];
            let mut j = i + 1;
            while j + 1 < f.len() && f[j + 1] - f[j] == d {
                j += 1;
            }
            (d, j)
        } else {
            (1, i)
        };
        runs.push((f[i], d, Some((j - i + 1) as i64)));
        i = j + 1;
    }
    runs
}

/// Minimum representation under the non-overlap rule.
pub fn solve_b(inputs: &[AP]) -> Vec<AP> {
    let (f, t, p, rb) = normalize(inputs);
    let f_vec: Vec<i64> = f.iter().copied().collect();
    if !rb.iter().any(|&x| x != 0) {
        return greedy_runs(&f_vec);
    }
    let inf_c = infinite_candidates(&f, t, p, &rb);
    let toks = tail_tokens(&inf_c, p, &rb);
    let tmask = token_masks(&inf_c, &toks);
    let n_toks = toks.len();
    let mut best: Option<Vec<AP>> = None;
    for i in 0..=f_vec.len() {
        let c = if i < f_vec.len() { f_vec[i] } else { t };
        let suffix = &f_vec[i..];
        let eidx: HashMap<i64, usize> = suffix.iter().enumerate().map(|(k, &x)| (x, k)).collect();
        let n_s = suffix.len();
        let n_univ = n_s + n_toks;
        let mut cands: Vec<(Bitset, AP)> = Vec::new();
        for (ci, &(d, b, e)) in inf_c.iter().enumerate() {
            let s0 = if e >= c { e } else { c + (b - c).rem_euclid(d) };
            let mut m = Bitset::empty(n_univ);
            for j in 0..n_toks {
                if tmask[ci].get(j) {
                    m.insert(n_s + j);
                }
            }
            let mut x = s0;
            while x < t {
                m.insert(eidx[&x]);
                x += d;
            }
            cands.push((m, (s0, d, None)));
        }
        let tail = set_cover(n_univ, cands);
        if let Some(mut tail) = tail {
            let mut blocks = greedy_runs(&f_vec[..i]);
            if best.as_ref().is_none_or(|b| blocks.len() + tail.len() < b.len()) {
                blocks.append(&mut tail);
                best = Some(blocks);
            }
        }
    }
    best.expect("solve_b: no feasible cut")
}

// ------------------------------------------------------------ verification

pub fn ap_contains(st: &AP, x: i64) -> bool {
    let (s, d, n) = *st;
    if let Some(n) = n {
        s <= x && x < s + n * d && (x - s).rem_euclid(d) == 0
    } else {
        x >= s && (x - s).rem_euclid(d) == 0
    }
}

pub fn verify(inputs: &[AP], sol: &[AP], rule_b: bool) {
    let (f, t, p, rb) = normalize(inputs);
    let p_usz = p as usize;
    let mem = |x: i64| -> bool {
        if x < t {
            f.contains(&x)
        } else {
            rb[x.rem_euclid(p) as usize] != 0
        }
    };
    for st in sol {
        // each chosen set inside S
        let (s, d, n) = *st;
        if let Some(n) = n {
            for k in 0..n {
                let x = s + k * d;
                assert!(mem(x), "{:?}", st);
            }
        } else {
            assert!(p % d == 0, "{:?}", st);
            let mut x = s;
            while x < t {
                assert!(mem(x), "{:?}", st);
                x += d;
            }
            // tail class fully inside R
            let mut r = s.rem_euclid(d) as usize;
            while r < p_usz {
                assert!(rb[r] != 0, "{:?}", st);
                r += d as usize;
            }
        }
    }
    for &x in &f {
        // prefix fully covered
        assert!(sol.iter().any(|st| ap_contains(st, x)), "{}", x);
    }
    // every tail class of R owned by some infinite pick
    let mut cov = vec![0u8; p_usz];
    for &(s, d, n) in sol {
        if n.is_none() {
            let mut r = s.rem_euclid(d) as usize;
            while r < p_usz {
                cov[r] = 1;
                r += d as usize;
            }
        }
    }
    for i in 0..p_usz {
        if rb[i] != 0 {
            assert!(cov[i] != 0, "residue {} uncovered", i);
        }
    }
    // spot-check the seam, capped to 2000 positions
    let w = p.min(1000);
    for x in t..t + 2 * w {
        assert!(
            mem(x) == sol.iter().any(|st| ap_contains(st, x)),
            "{}",
            x
        );
    }
    if rule_b {
        // structural constraints
        let mut spans: Vec<(i64, i64)> = sol
            .iter()
            .filter_map(|&(s, d, n)| n.map(|nn| (s, s + (nn - 1) * d)))
            .collect();
        spans.sort();
        for i in 0..spans.len().saturating_sub(1) {
            assert!(spans[i].1 < spans[i + 1].0, "{:?}", spans);
        }
        let starts: Vec<i64> = sol
            .iter()
            .filter_map(|&(s, _, n)| if n.is_none() { Some(s) } else { None })
            .collect();
        if !starts.is_empty() && !spans.is_empty() {
            let max_hi = spans.iter().map(|&(_, hi)| hi).max().unwrap();
            let min_start = *starts.iter().min().unwrap();
            assert!(max_hi < min_start, "{:?} {:?}", spans, starts);
        }
    }
}

pub fn fmt(st: &AP) -> String {
    let (s, d, n) = *st;
    if let Some(n) = n {
        if n == 1 {
            format!("{{{}}}", s)
        } else {
            format!("{{{}+{}k, 0<=k<{}}}", s, d, n)
        }
    } else {
        format!("{{{}+{}k, k>=0}}", s, d)
    }
}
