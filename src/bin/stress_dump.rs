//! Per-instance cardinality dump for the stress test.
//!
//! Usage: `cargo run --release --bin stress_dump -- [trials] [seed]`
//!
//! Prints one line per random instance, in the form `i |sa| |sb|`, where
//! `i` is the 0-based trial index.  Useful for verifying that an
//! optimization preserves the per-instance cardinality of `solve_a` and
//! `solve_b` outputs: dump before the change and after, then `diff` should
//! be empty.

use ap_cover::{solve_a, solve_b, verify, AP};

// Mirror of the xorshift64* RNG in src/main.rs so dumped instances align
// exactly with what the stress test would generate.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed } }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn randint(&mut self, lo: i64, hi: i64) -> i64 {
        let range = (hi - lo + 1) as u64;
        lo + (self.next_u64() % range) as i64
    }
    fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn random_instance(rng: &mut Rng) -> Vec<AP> {
    let mut aps = Vec::new();
    let n_ap = rng.randint(1, 30);
    for _ in 0..n_ap {
        let d = rng.randint(1, 6);
        let s = rng.randint(-10, 10);
        let n = if rng.random() < 0.5 { None } else { Some(rng.randint(1, 20)) };
        aps.push((s, d, n));
    }
    aps
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let trials: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);

    let mut rng = Rng::new(seed);
    for i in 0..trials {
        let inputs = random_instance(&mut rng);
        let sa = solve_a(&inputs);
        let sb = solve_b(&inputs);
        verify(&inputs, &sa, false);
        verify(&inputs, &sb, true);
        println!("{} {} {}", i, sa.len(), sb.len());
    }
}
