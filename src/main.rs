//! Demo + stress tester for `ap_cover`.

use ap_cover::{fmt, solve_a, solve_b, verify, AP};
use std::collections::BTreeMap;

fn examples() -> Vec<(&'static str, Vec<AP>)> {
    vec![
        (
            "original case 1",
            vec![(22, 4, None), (26, 4, None)],
        ),
        (
            "original case 2",
            vec![(22, 6, None), (24, 6, None), (26, 6, None)],
        ),
        (
            "original case 3",
            vec![(22, 6, None), (26, 6, None), (28, 6, None)],
        ),
        (
            "original case 4",
            vec![(23, 2, None), (23, 4, None)],
        ),
        (
            "original case 5",
            vec![(23, 2, None), (21, 2, Some(1))],
        ),
        (
            "original case 6",
            vec![(23, 2, None), (19, 2, Some(3))],
        ),
        (
            "odds k>=30 with evens k>=30",
            vec![(61, 2, None), (60, 2, None)],
        ),
        (
            "evens >= 0 with {1, 5}",
            vec![(0, 2, None), (1, 4, Some(2))],
        ),
        (
            "{0,7,14,21,28} with {3,7,11,15,19}",
            vec![(0, 7, Some(5)), (3, 4, Some(5))],
        ),
        (
            "multiples of 2 and of 3",
            vec![(0, 2, None), (0, 3, None)],
        ),
        (
            "all six classes mod 6",
            (0..6).map(|k| (k, 6, None)).collect(),
        ),
        (
            "staggered classes plus a finite stray",
            vec![
                (0, 4, None),
                (2, 4, None),
                (1, 2, None),
                (5, 8, Some(3)),
            ],
        ),
        (
            "class with an interior gap",
            vec![(0, 4, None), (2, 4, Some(2))],
        ),
        (
            "complicated 1",
            vec![(5, 1, Some(11)), (5, 6, Some(12)), (7, 5, Some(19)), (-9, 3, Some(13)), (-7, 4, Some(20)), (-4, 5, Some(13)), (-5, 6, Some(12)), (-1, 6, Some(11)), (-2, 6, Some(5))]
        ),
        (
            "all primes",
            vec![(2, 5, None), (2, 7, None), (2, 11, None), (2, 13, None), (2, 17, None), (2, 19, None)],
        )
    ]
}

fn report(name: &str, inputs: &[AP]) {
    let sa = solve_a(inputs);
    let sb = solve_b(inputs);
    verify(inputs, &sa, false);
    verify(inputs, &sb, true);
    let desc = inputs
        .iter()
        .map(fmt)
        .collect::<Vec<_>>()
        .join("  ");
    println!("{}\n  inputs ({}): {}", name, inputs.len(), desc);
    let mut sa_sorted = sa.clone();
    sa_sorted.sort_by_key(|t| t.1);
    let mut sb_sorted = sb.clone();
    sb_sorted.sort_by_key(|t| t.1);
    println!(
        "  (a) {} sets:  {}",
        sa.len(),
        sa_sorted.iter().map(fmt).collect::<Vec<_>>().join("  ")
    );
    println!(
        "  (b) {} sets:  {}\n",
        sb.len(),
        sb_sorted.iter().map(fmt).collect::<Vec<_>>().join("  ")
    );
}

// Simple xorshift64* RNG — deterministic, no external dependencies.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Inclusive on both ends, matching Python's `random.randint`.
    pub fn randint(&mut self, lo: i64, hi: i64) -> i64 {
        let range = (hi - lo + 1) as u64;
        lo + (self.next_u64() % range) as i64
    }

    /// Uniform in [0, 1), matching Python's `random.random`.
    pub fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn random_instance(rng: &mut Rng) -> Vec<AP> {
    let mut aps = Vec::new();
    let n_ap = rng.randint(1, 30);
    for _ in 0..n_ap {
        let d = rng.randint(1, 6);
        let s = rng.randint(-10, 10);
        let n = if rng.random() < 0.5 {
            None
        } else {
            Some(rng.randint(1, 20))
        };
        aps.push((s, d, n));
    }
    aps
}

fn stress(trials: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    let mut gaps: BTreeMap<i64, i64> = BTreeMap::new();
    for trial in 0..trials {
        let inputs = random_instance(&mut rng);
        let sa = solve_a(&inputs);
        let sb = solve_b(&inputs);
        verify(&inputs, &sa, false);
        verify(&inputs, &sb, true);
        assert!(sa.len() <= inputs.len(), "{:?} {:?}", inputs, sa);
        assert!(sa.len() <= sb.len(), "{:?} {:?} {:?}", inputs, sa, sb);
        let g = (sb.len() - sa.len()) as i64;
        *gaps.entry(g).or_insert(0) += 1;
        if trial > 0 && (1 + trial) % 100 == 0 {
            let hist = gaps
                .iter()
                .map(|(k, v)| format!("gap {}: {}", k, v))
                .collect::<Vec<_>>()
                .join(", ");
            println!(
                "{} random instances verified.  (b) minus (a): {}",
                trial + 1,
                hist
            );
        }
    }
    if trials % 100 > 0 {
        let hist = gaps
            .iter()
            .map(|(k, v)| format!("gap {}: {}", k, v))
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "{} random instances verified.  (b) minus (a): {}",
            trials, hist
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let trials: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);

    for (name, inputs) in examples() {
        report(name, &inputs);
    }
    stress(trials, seed);
}
