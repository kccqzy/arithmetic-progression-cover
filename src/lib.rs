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
//! This file is a faithful port of `ap_cover.py`; see that file's module
//! docstring for the mathematical background.

use std::collections::{BTreeSet, HashMap};

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

    /// `self & !other` (bits in `self` not in `other`).
    pub fn difference(&self, other: &Self) -> Self {
        Self {
            words: self.words.iter().zip(&other.words).map(|(&a, &b)| a & !b).collect(),
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

/// Return `(F, T, P, R)`: sorted prefix, threshold, period, tail residues.
pub fn normalize(inputs: &[AP]) -> (BTreeSet<i64>, i64, i64, BTreeSet<i64>) {
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
    let (p, t, r): (i64, i64, BTreeSet<i64>) = if !infs.is_empty() {
        let p = infs.iter().map(|&(_, d)| d).fold(1, lcm);
        let mut t = infs.iter().map(|&(s, _)| s).max().unwrap();
        if !fins.is_empty() {
            let max_fin = fins.iter().map(|&(s, d, n)| s + (n - 1) * d).max().unwrap();
            t = t.max(1 + max_fin);
        }
        let mut r: BTreeSet<i64> = BTreeSet::new();
        for &(s, d) in &infs {
            for k in 0..p / d {
                r.insert((s + k * d).rem_euclid(p));
            }
        }
        (p, t, r)
    } else {
        let max_fin = fins.iter().map(|&(s, d, n)| s + (n - 1) * d).max().unwrap();
        (1, 1 + max_fin, BTreeSet::new())
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
    (f, t, p, r)
}

// -------------------------------------------------------------- candidates

pub fn divisors(p: i64) -> Vec<i64> {
    let mut out: BTreeSet<i64> = BTreeSet::new();
    let mut d: i64 = 1;
    while d * d <= p {
        if p % d == 0 {
            out.insert(d);
            out.insert(p / d);
        }
        d += 1;
    }
    out.into_iter().collect()
}

/// Maximal infinite APs inside S with difference dividing P, as
/// `(d, b, e, coset)`: difference, class mod d, leftmost start, residues.
pub fn infinite_candidates(
    f: &BTreeSet<i64>,
    t: i64,
    p: i64,
    r: &BTreeSet<i64>,
) -> Vec<(i64, i64, i64, Vec<i64>)> {
    let mut out = Vec::new();
    for d in divisors(p) {
        for b in 0..d {
            let coset: Vec<i64> = (0..p / d).map(|k| b + k * d).collect();
            if coset.iter().all(|x| r.contains(x)) {
                let mut e = t + (b - t).rem_euclid(d); // least element >= T in class
                while f.contains(&(e - d)) {
                    // extend left through prefix
                    e -= d;
                }
                out.push((d, b, e, coset));
            }
        }
    }
    out
}

/// Maximal APs (including singletons) contained in F, as `(s, d, n)`.
pub fn finite_candidates(f: &BTreeSet<i64>) -> Vec<(i64, i64, i64)> {
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
    out.into_iter().collect()
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
    let mut by_mask: HashMap<Bitset, AP> = HashMap::new();
    let mut order: Vec<Bitset> = Vec::new();
    for (m, p) in cands {
        if !m.is_empty() && !by_mask.contains_key(&m) {
            order.push(m.clone());
            by_mask.insert(m, p);
        }
    }

    // Stable sort by descending popcount.
    order.sort_by_cached_key(|m| std::cmp::Reverse(m.count()));

    let mut kept: Vec<Bitset> = Vec::new();
    for m in order {
        if !kept.iter().any(|k| m.is_subset_of(k)) {
            kept.push(m);
        }
    }

    let union: Bitset = kept
        .iter()
        .fold(Bitset::empty(n_univ), |a, b| a.union(b));
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
        unc = unc.difference(&sets[i]);
    }
    let mut best = chosen;

    let maxsz = sets.iter().map(|m| m.count()).max().unwrap() as usize;
    let mut memo: HashMap<Bitset, usize> = HashMap::new();

    fn dfs(
        unc: &Bitset,
        chosen: &mut Vec<usize>,
        best: &mut Vec<usize>,
        sets: &[Bitset],
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

        let mut cand_indices = elem_sets[e].clone();
        cand_indices.sort_by_cached_key(|&i| std::cmp::Reverse(sets[i].intersect_count(unc)));

        for i in cand_indices {
            chosen.push(i);
            let new_unc = unc.difference(&sets[i]);
            dfs(&new_unc, chosen, best, sets, elem_sets, n_univ, maxsz, memo);
            chosen.pop();
        }
    }

    let mut chosen2 = Vec::new();
    dfs(
        &full,
        &mut chosen2,
        &mut best,
        &sets,
        &elem_sets,
        n_univ,
        maxsz,
        &mut memo,
    );

    Some(best.iter().map(|&i| by_mask[&sets[i]]).collect())
}

// -------------------------------------------------------- formulation (a)

/// Minimum representation, overlap allowed.
pub fn solve_a(inputs: &[AP]) -> Vec<AP> {
    let (f, t, p, r) = normalize(inputs);
    let n_univ = f.len() + r.len();
    let eidx: HashMap<i64, usize> = f.iter().enumerate().map(|(i, &x)| (x, i)).collect();
    let tidx: HashMap<i64, usize> = r
        .iter()
        .enumerate()
        .map(|(i, &x)| (x, f.len() + i))
        .collect();
    let mut cands: Vec<(Bitset, AP)> = Vec::new();
    for (d, _b, e, coset) in infinite_candidates(&f, t, p, &r) {
        let mut m = Bitset::empty(n_univ);
        for r_val in &coset {
            m.insert(tidx[r_val]);
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
    let (f, t, p, r) = normalize(inputs);
    if r.is_empty() {
        let f: Vec<i64> = f.iter().copied().collect();
        return greedy_runs(&f);
    }
    let inf_c = infinite_candidates(&f, t, p, &r);
    let f: Vec<i64> = f.iter().copied().collect();
    let mut best: Option<Vec<AP>> = None;
    for i in 0..=f.len() {
        let c = if i < f.len() { f[i] } else { t };
        let suffix = &f[i..];
        let eidx: HashMap<i64, usize> = suffix.iter().enumerate().map(|(k, &x)| (x, k)).collect();
        let tidx: HashMap<i64, usize> = r
            .iter()
            .enumerate()
            .map(|(k, &x)| (x, suffix.len() + k))
            .collect();
        let n_univ = suffix.len() + r.len();
        let mut cands: Vec<(Bitset, AP)> = Vec::new();
        for &(d, b, e, ref coset) in &inf_c {
            let s0 = if e >= c { e } else { c + (b - c).rem_euclid(d) };
            let mut m = Bitset::empty(n_univ);
            for r_val in coset {
                m.insert(tidx[r_val]);
            }
            let mut x = s0;
            while x < t {
                m.insert(eidx[&x]);
                x += d;
            }
            cands.push((m, (s0, d, None)));
        }
        let tail = set_cover(n_univ, cands);
        if let Some(tail) = tail {
            let blocks = greedy_runs(&f[..i]);
            let total: Vec<AP> = blocks.into_iter().chain(tail).collect();
            if best.as_ref().is_none_or(|b| total.len() < b.len()) {
                best = Some(total);
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
    let (f, t, p, r) = normalize(inputs);
    let mem = |x: i64| -> bool {
        if x < t {
            f.contains(&x)
        } else {
            r.contains(&x.rem_euclid(p))
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
            while x < t + p {
                assert!(mem(x), "{:?}", st);
                x += d;
            }
            for k in 0..p / d {
                assert!(r.contains(&(s + k * d).rem_euclid(p)), "{:?}", st);
            }
        }
    }
    for &x in &f {
        // prefix fully covered
        assert!(sol.iter().any(|st| ap_contains(st, x)), "{}", x);
    }
    for &rv in &r {
        // each tail class owned fully
        assert!(
            sol.iter().any(|&(s, d, n)| {
                n.is_none() && rv.rem_euclid(d) == s.rem_euclid(d)
            }),
            "{}",
            rv
        );
    }
    for x in t..t + 2 * p {
        // spot-check the seam
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
