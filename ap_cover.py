#!/usr/bin/env python3
"""Minimum representations of a union of arithmetic progressions (APs).

An AP is (s, d, n) = {s + k*d : 0 <= k < n}, with n = None meaning infinite.
Given input APs whose union is S, we compute:

  (a) a minimum-size family of APs, each a subset of S, whose union is S,
      with overlap freely allowed;
  (b) the same, under the non-overlap rule: for any two distinct chosen sets
      A, B, either max(A) < min(B), or max(B) < min(A), or both are infinite.

Method.  S normalizes to a finite prefix F = S ∩ (-inf, T) plus a periodic
tail: for x >= T, x ∈ S iff x mod P ∈ R, where P = lcm of the infinite
inputs' differences.  R is represented as a byte indicator array Rb of
length P, never as a per-residue collection.

  (a) becomes exact set cover over ground set F plus tail tokens.
  (b) decomposes at a cut c (the earliest infinite start): the prefix is
      partitioned into consecutive runs by a greedy pass (provably optimal),
      the suffix is covered by infinite APs only; all cuts are scanned.

Scalability.  Large periods are kept tractable by working on whole arrays:
  1. Valid residue classes (b mod d whose full lift to Z/P lies in R) are
     computed for all divisors d of P by DP down the divisor lattice: the
     validity array at level d is the byte-wise AND of the q chunks of the
     array at level d*q.  Byte-wise AND/OR run at C speed through integer
     arithmetic on the underlying bytes.
  2. Dominated candidates are pruned wholesale.  Candidate (d,b,e) is
     dominated by (dp,bp,ep) when dp | d, b ≡ bp (mod dp) and e >= ep: its
     residue classes and its prefix coverage are then subsets of the coarser
     candidate's, under every cut.  Dominance is transitive, so it suffices
     that some valid coarser class dominates.  For classes that do not
     extend into the prefix (e >= T), e >= ep holds automatically, so the
     dominance mask at level d is the OR over primes q | d of the tiled
     validity array of level d/q.  The few classes that do extend into the
     prefix (at most |F| per divisor) are checked individually against
     actual starts.
  3. The tail ground set is compressed by coverage signature: residues of R
     covered by exactly the same candidates are one token.  Signatures are
     assembled in candidate-bit byte planes and deduplicated through a
     memoryview cast, so the set-cover universe is the number of distinct
     signatures, not |R|.

Modeling restriction: infinite APs in solutions use differences dividing P.
Every input satisfies this and every full residue class is expressible this
way; an AP with d not dividing P covers only a strict sub-progression of
each class it meets.  We have found no instance where such tilings beat the
divisor solutions, but we do not have a proof that they never do.
"""
import sys
import random
from math import gcd
from functools import reduce

# ----------------------------------------------------------- normalization

def lcm(a, b):
    return a * b // gcd(a, b)

def class_ones(b, P, d):
    """Indicator segment for the class b mod d inside [0, P)."""
    return b"\x01" * len(range(b, P, d))

def normalize(inputs):
    """Return (F, T, P, Rb): sorted prefix, threshold, period, and the tail
    residue indicator (Rb[r] == 1 iff residue r mod P belongs to S's tail)."""
    assert inputs and all(d >= 1 and (n is None or n >= 1) for _, d, n in inputs)
    infs = [(s, d) for s, d, n in inputs if n is None]
    fins = [(s, d, n) for s, d, n in inputs if n is not None]
    if infs:
        P = reduce(lcm, (d for _, d in infs), 1)
        T = max(s for s, _ in infs)
        if fins:
            T = max(T, 1 + max(s + (n - 1) * d for s, d, n in fins))
        Rb = bytearray(P)
        for s, d in infs:
            Rb[s % d::d] = class_ones(s % d, P, d)
    else:
        P, Rb = 1, bytearray(1)
        T = 1 + max(s + (n - 1) * d for s, d, n in fins)
    F = set()
    for s, d, n in fins:
        F.update(x for x in range(s, s + n * d, d) if x < T)
    for s, d in infs:
        F.update(range(s, T, d))
    return sorted(F), T, P, Rb

# -------------------------------------------------------------- candidates

def divisors(P):
    out, d = set(), 1
    while d * d <= P:
        if P % d == 0:
            out |= {d, P // d}
        d += 1
    return sorted(out)

def prime_factors(n):
    out, p = [], 2
    while p * p <= n:
        if n % p == 0:
            out.append(p)
            while n % p == 0:
                n //= p
        p += 1
    if n > 1:
        out.append(n)
    return out

def valid_classes(P, Rb):
    """flags[d][b] == 1 iff the full lift of b mod d to Z/P lies inside R,
    for every divisor d of P.  DP down the divisor lattice: the array at d
    is the byte-wise AND of the q chunks of the array at d*q (any prime q
    dividing P/d), performed as integer arithmetic on the bytes."""
    flags = {P: bytes(Rb)}
    for d in sorted(divisors(P), reverse=True)[1:]:
        q = prime_factors(P // d)[0]
        par = flags[d * q]
        v = int.from_bytes(par[:d], "little")
        for k in range(1, q):
            v &= int.from_bytes(par[k * d:(k + 1) * d], "little")
        flags[d] = v.to_bytes(d, "little")
    return flags

def infinite_candidates(F_set, T, P, Rb):
    """Non-dominated maximal infinite APs inside S with difference dividing
    P, as (d, b, e): difference, class mod d, leftmost start."""
    flags = valid_classes(P, Rb)
    e_memo = {}

    def start(d, b):
        e = e_memo.get((d, b))
        if e is None:
            e = T + (b - T) % d
            while e - d in F_set:
                e -= d
            e_memo[(d, b)] = e
        return e

    out = []
    for d in flags:
        # classes whose chain can extend into the prefix (at most |F|)
        ext = {y % d for y in F_set if y >= T - d}
        dom = 0
        for q in prime_factors(d):
            dom |= int.from_bytes(flags[d // q] * q, "little")
        surv = (int.from_bytes(flags[d], "little") & ~dom).to_bytes(d, "little")
        pos = surv.find(1)
        while pos != -1:                    # unextendable survivors
            if pos not in ext:
                out.append((d, pos, start(d, pos)))
            pos = surv.find(1, pos + 1)
        for b in sorted(ext):               # extendable classes: exact test
            if flags[d][b]:
                e = start(d, b)
                if not any(flags[dp][b % dp] and e >= start(dp, b % dp)
                           for dp in divisors(d)[:-1]):
                    out.append((d, b, e))
    return out

def tail_tokens(cands, P, Rb):
    """Coverage-signature compression of the tail: residues of R covered by
    exactly the same candidates are one token.  Returns the sorted list of
    distinct masks; bit i of a mask marks coverage by candidate i."""
    if not cands:
        return []
    nplanes = (len(cands) + 7) // 8
    rec = next((r for r in (1, 2, 4, 8) if r >= nplanes), None)
    if rec is None:                         # more than 64 candidates
        cover = {}
        for i, (d, b, e) in enumerate(cands):
            bit = 1 << i
            for r in range(b, P, d):
                cover[r] = cover.get(r, 0) | bit
        return sorted(set(cover.values()))
    planes = []
    for j in range(rec):
        pi = 0
        for i in range(8 * j, min(8 * j + 8, len(cands))):
            d, b, _ = cands[i]
            ind = bytearray(P)
            ind[b::d] = bytes([1 << (i - 8 * j)]) * len(range(b, P, d))
            pi |= int.from_bytes(ind, "little")
        planes.append(pi.to_bytes(P, "little"))
    if rec == 1:
        return sorted(set(planes[0]) - {0})
    comb = bytearray(rec * P)
    for j in range(rec):
        comb[j::rec] = planes[j]
    code = {2: "H", 4: "I", 8: "Q"}[rec]
    vals = set(memoryview(bytes(comb)).cast(code)) - {0}
    return sorted(int.from_bytes(v.to_bytes(rec, sys.byteorder), "little")
                  for v in vals)

def finite_candidates(F, F_set):
    """Maximal APs (including singletons) contained in F, as (s, d, n)."""
    out = {(x, 1, 1) for x in F}
    for i, s in enumerate(F):
        for t in F[i + 1:]:
            d = t - s
            if s - d in F_set:             # not left-maximal
                continue
            n, x = 2, t + d
            while x in F_set:
                n, x = n + 1, x + d
            out.add((s, d, n))
    return sorted(out)

# ----------------------------------------------------------- exact cover(s)

def set_cover(n_univ, cands):
    """Exact minimum set cover of {0..n_univ-1}; cands is [(mask, payload)].
    Returns chosen payloads, or None if infeasible.  Branch and bound over
    bitmasks with greedy upper bound, dominance pruning, and memoization."""
    full = (1 << n_univ) - 1
    if full == 0:
        return []
    by_mask = {}
    for m, p in cands:
        if m:
            by_mask.setdefault(m, p)
    kept = []
    for m in sorted(by_mask, key=lambda m: -m.bit_count()):
        if not any(m | k == k for k in kept):   # drop dominated sets
            kept.append(m)
    if reduce(lambda a, b: a | b, kept, 0) != full:
        return None
    sets = kept
    elem_sets = [[i for i, m in enumerate(sets) if m >> e & 1]
                 for e in range(n_univ)]
    unc, chosen = full, []                      # greedy upper bound
    while unc:
        i = max(range(len(sets)), key=lambda i: (sets[i] & unc).bit_count())
        chosen.append(i)
        unc &= ~sets[i]
    best = chosen
    maxsz = max(m.bit_count() for m in sets)
    memo = {}

    def dfs(unc, chosen):
        nonlocal best
        if not unc:
            if len(chosen) < len(best):
                best = chosen[:]
            return
        lb = (unc.bit_count() + maxsz - 1) // maxsz
        if len(chosen) + lb >= len(best):
            return
        seen = memo.get(unc)
        if seen is not None and seen <= len(chosen):
            return
        memo[unc] = len(chosen)
        e = min((e for e in range(n_univ) if unc >> e & 1),
                key=lambda e: len(elem_sets[e]))
        for i in sorted(elem_sets[e],
                        key=lambda i: -(sets[i] & unc).bit_count()):
            chosen.append(i)
            dfs(unc & ~sets[i], chosen)
            chosen.pop()

    dfs(full, [])
    return [by_mask[sets[i]] for i in best]

# -------------------------------------------------------- formulation (a)

def token_masks(inf_c, toks):
    """For each candidate, the bitmask of tokens it covers."""
    tmask = [0] * len(inf_c)
    for j, t in enumerate(toks):
        for i in range(len(inf_c)):
            if t >> i & 1:
                tmask[i] |= 1 << j
    return tmask

def solve_a(inputs):
    """Minimum representation, overlap allowed.
    Returns a list of ('fin', s, d, n) and ('inf', s, d) descriptors."""
    F, T, P, Rb = normalize(inputs)
    F_set = set(F)
    inf_c = infinite_candidates(F_set, T, P, Rb)
    toks = tail_tokens(inf_c, P, Rb)
    tmask = token_masks(inf_c, toks)
    nF = len(F)
    eidx = {x: i for i, x in enumerate(F)}
    cands = []
    for i, (d, b, e) in enumerate(inf_c):
        m = tmask[i] << nF
        for x in range(e, T, d):
            m |= 1 << eidx[x]
        cands.append((m, (e, d, None)))
    for s, d, n in finite_candidates(F, F_set):
        m = 0
        for x in range(s, s + n * d, d):
            m |= 1 << eidx[x]
        cands.append((m, (s, d, n)))
    return set_cover(nF + len(toks), cands)

# -------------------------------------------------------- formulation (b)

def greedy_runs(F):
    """Optimal partition of sorted F into fewest consecutive AP runs."""
    runs, i = [], 0
    while i < len(F):
        if i + 1 < len(F):
            d, j = F[i + 1] - F[i], i + 1
            while j + 1 < len(F) and F[j + 1] - F[j] == d:
                j += 1
        else:
            d, j = 1, i
        runs.append((F[i], d, j - i + 1))
        i = j + 1
    return runs

def solve_b(inputs):
    """Minimum representation under the non-overlap rule: finite blocks have
    strictly ordered hulls and end before the earliest infinite start;
    infinite APs may overlap each other."""
    F, T, P, Rb = normalize(inputs)
    F_set = set(F)
    if 1 not in Rb:                             # S finite: blocks only
        return [(s, d, n) for s, d, n in greedy_runs(F)]
    inf_c = infinite_candidates(F_set, T, P, Rb)
    toks = tail_tokens(inf_c, P, Rb)
    tmask = token_masks(inf_c, toks)
    best = None
    for i in range(len(F) + 1):                 # cut c = F[i], or c = T
        c = F[i] if i < len(F) else T
        suffix = F[i:]
        eidx = {x: k for k, x in enumerate(suffix)}
        nS = len(suffix)
        cands = []
        for ci, (d, b, e) in enumerate(inf_c):
            s0 = e if e >= c else c + (b - c) % d   # first AP element >= c
            m = tmask[ci] << nS
            for x in range(s0, T, d):
                m |= 1 << eidx[x]
            cands.append((m, (s0, d, None)))
        tail = set_cover(nS + len(toks), cands)
        if tail is None:
            continue
        blocks = [(s, d, n) for s, d, n in greedy_runs(F[:i])]
        if best is None or len(blocks) + len(tail) < len(best):
            best = blocks + tail
    return best

# ------------------------------------------------------------ verification

def ap_contains(st, x):
    if st[2] is not None:
        s, d, n = st
        return s <= x < s + n * d and (x - s) % d == 0
    s, d, _ = st
    return x >= s and (x - s) % d == 0

def verify(inputs, sol, rule_b):
    F, T, P, Rb = normalize(inputs)
    F_set = set(F)
    mem = lambda x: (x in F_set) if x < T else bool(Rb[x % P])
    for st in sol:                              # each chosen set inside S
        if st[2] is not None:
            s, d, n = st
            assert all(mem(x) for x in range(s, s + n * d, d)), st
        else:
            s, d, _ = st
            assert P % d == 0, st
            assert all(mem(x) for x in range(s, T, d)), st
            assert Rb[s % d::d].count(0) == 0, st   # tail classes inside R
    for x in F:                                 # prefix fully covered
        assert any(ap_contains(st, x) for st in sol), x
    cov = bytearray(P)                          # every tail class owned
    for s, d, n in sol:
        if n is None:
            cov[s % d::d] = class_ones(s % d, P, d)
    Ri, Ci = int.from_bytes(Rb, "little"), int.from_bytes(cov, "little")
    assert Ri & ~Ci == 0
    W = min(P, 1000)                            # redundant seam spot-check
    for x in range(T, T + 2 * W):
        assert mem(x) == any(ap_contains(st, x) for st in sol), x
    if rule_b:                                  # structural constraints
        spans = sorted((s, s + (n - 1) * d)
                       for s, d, n in sol if n is not None)
        assert all(spans[i][1] < spans[i + 1][0]
                   for i in range(len(spans) - 1)), spans
        starts = [s for s, d, n in sol if n is None]
        if starts and spans:
            assert max(hi for _, hi in spans) < min(starts), (spans, starts)

# ------------------------------------------------------------------- demo

def fmt(st):
    if st[2] is not None:
        s, d, n = st
        return f"{{{s}}}" if n == 1 else f"{{{s}+{d}k, 0<=k<{n}}}"
    s, d, _ = st
    return f"{{{s}+{d}k, k>=0}}"

def report(name, inputs):
    sa, sb = solve_a(inputs), solve_b(inputs)
    verify(inputs, sa, False)
    verify(inputs, sb, True)
    desc = "  ".join(fmt(st)
                     for st in inputs)
    print(f"{name}\n  inputs ({len(inputs)}): {desc}")
    print(f"  (a) {len(sa)} sets:  "
          + "  ".join(fmt(s) for s in sorted(sa, key=lambda t: t[1])))
    print(f"  (b) {len(sb)} sets:  "
          + "  ".join(fmt(s) for s in sorted(sb, key=lambda t: t[1])) + "\n")

EXAMPLES = [
    ("original case 1", [(22, 4, None), (26, 4, None)]),
    ("original case 2", [(22, 6, None), (24, 6, None), (26, 6, None)]),
    ("original case 3", [(22, 6, None), (26, 6, None), (28, 6, None)]),
    ("original case 4", [(23, 2, None), (23, 4, None)]),
    ("original case 5", [(23, 2, None), (21, 2, 1)]),
    ("original case 6", [(23, 2, None), (19, 2, 3)]),
    ("odds k>=30 with evens k>=30", [(61, 2, None), (60, 2, None)]),
    ("evens >= 0 with {1, 5}", [(0, 2, None), (1, 4, 2)]),
    ("{0,7,14,21,28} with {3,7,11,15,19}", [(0, 7, 5), (3, 4, 5)]),
    ("multiples of 2 and of 3", [(0, 2, None), (0, 3, None)]),
    ("all six classes mod 6", [(k, 6, None) for k in range(6)]),
    ("staggered classes plus a finite stray",
     [(0, 4, None), (2, 4, None), (1, 2, None), (5, 8, 3)]),
    ("class with an interior gap", [(0, 4, None), (2, 4, 2)]),
    ("complicated 1", [(5, 1, 11), (5, 6, 12), (7, 5, 19), (-9, 3, 13), (-7, 4, 20), (-4, 5, 13), (-5, 6, 12), (-1, 6, 11), (-2, 6, 5)]),
    ("all primes 6", [(2, 5, None), (2, 7, None), (2, 11, None), (2, 13, None), (2, 17, None), (2, 19, None)]),
    ("all primes 7", [(2, 5, None), (2, 7, None), (2, 11, None), (2, 13, None), (2, 17, None), (2, 19, None), (2, 23, None)]),
]

def random_instance(rng):
    aps = []
    for _ in range(rng.randint(1, 30)):
        d = rng.randint(1, 6)
        s = rng.randint(-10, 10)
        n = None if rng.random() < 0.5 else rng.randint(1, 20)
        aps.append((s, d, n))
    return aps

def stress(trials, seed=1):
    rng = random.Random(seed)
    gaps = {}
    for trial in range(trials):
        inputs = random_instance(rng)
        sa, sb = solve_a(inputs), solve_b(inputs)
        verify(inputs, sa, False)
        verify(inputs, sb, True)
        assert len(sa) <= len(inputs), (inputs, sa)
        assert len(sa) <= len(sb), (inputs, sa, sb)
        g = len(sb) - len(sa)
        gaps[g] = gaps.get(g, 0) + 1
        if trial > 0 and (1 + trial) % 100 == 0:
            hist = ", ".join(f"gap {k}: {v}" for k, v in sorted(gaps.items()))
            print(f"{trial + 1} random instances verified.  (b) minus (a): {hist}")
    if trials % 100 > 0:
            hist = ", ".join(f"gap {k}: {v}" for k, v in sorted(gaps.items()))
            print(f"{trials} random instances verified.  (b) minus (a): {hist}")

if __name__ == "__main__":
    for name, inputs in EXAMPLES:
        report(name, inputs)
    stress(trials=1000)
