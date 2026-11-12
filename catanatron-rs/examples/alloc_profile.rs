//! How much self-play allocates: counts heap allocations (and bytes) through
//! a wrapping global allocator while playing a few self-play games with the
//! training settings, single-threaded.
//!   cargo run --release --example alloc_profile -- <model.ctnn> [games]
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use catanatron::map::{MapType, NumberPlacement};
use catanatron::mcts::{load_net, Utility};
use catanatron::selfplay::{self_play, SelfPlayConfig};
use catanatron::state::DiceMode;
use catanatron::*;

struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
/// Allocations (count, bytes) by size class: <=64, <=256, <=1K, <=4K, <=16K, <=64K, more.
static HIST: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];
fn bucket(n: usize) -> usize {
    match n { 0..=64 => 0, 65..=256 => 1, 257..=1024 => 2, 1025..=4096 => 3, 4097..=16384 => 4, 16385..=65536 => 5, _ => 6 }
}
fn count(n: usize) {
    ALLOCS.fetch_add(1, Ordering::Relaxed);
    BYTES.fetch_add(n as u64, Ordering::Relaxed);
    let b = bucket(n);
    HIST[2 * b].fetch_add(1, Ordering::Relaxed);
    HIST[2 * b + 1].fetch_add(n as u64, Ordering::Relaxed);
}

/// With a third argument, every SAMPLE-th allocation's backtrace is kept
/// (outside the sampler's own allocations) and the most common call sites
/// printed at the end.
const SAMPLE: u64 = 2000;
static TRACE: AtomicU64 = AtomicU64::new(0);
thread_local! { static IN_TRACE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
static TRACES: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
fn sample() {
    if TRACE.load(Ordering::Relaxed) == 0 || ALLOCS.load(Ordering::Relaxed) % SAMPLE != 0 {
        return;
    }
    IN_TRACE.with(|f| {
        if f.replace(true) {
            return;
        }
        let bt = std::backtrace::Backtrace::force_capture().to_string();
        // Keep the crate's own frames: the first few catanatron:: functions.
        let frames: Vec<&str> = bt
            .lines()
            .filter(|l| l.contains("catanatron::") && !l.contains("alloc_profile"))
            .map(|l| l.trim().trim_start_matches(|c: char| c.is_ascii_digit() || c == ':' || c == ' '))
            .take(3)
            .collect();
        TRACES.lock().unwrap().push(frames.join(" <- "));
        f.set(false);
    });
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        count(l.size());
        sample();
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        count(n);
        System.realloc(p, l, n)
    }
}

#[global_allocator]
static A: Counting = Counting;

fn main() {
    let model = std::env::args().nth(1).expect("model");
    let games: u64 = std::env::args().nth(2).map_or(4, |g| g.parse().expect("games"));
    TRACE.store(std::env::args().nth(3).is_some() as u64, Ordering::Relaxed);
    let net = load_net(&model).expect("model");
    let cfg = SelfPlayConfig {
        sims_full: 400, sims_fast: 50, sims_placement: 2400, p_full: 0.25,
        entropy_budget: false, kl_escalate: false, admit_kl: 0.0, focus_sims: 0, temp_moves: 15, root_noise: 0.25,
        dirichlet_alpha: 0.3, c_puct: 1.0, fpu_reduction: 0.25, escalate_gap: 0.0, endgame_vp: 13, endgame_branches: 1,
        opening_branches: 1, placement_forks: 1, placement_temp: 2.0, placement_uniform: 0.25, move_forks: 0, move_fork_temp: 1.0, move_fork_uniform: 0.0, sibling_labels: false, pool_board_share: 1.0, pool_seeds: Vec::new(), random_share: 0.0,
        random_eps: 1.0, utility: Utility { margin: 0.1, speed: 0.3, speed_turns: 150.0 }, gumbel_m: 0,
        game: GameConfig {
            map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
            dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, ..Default::default()
        },
    };
    let (a0, b0) = (ALLOCS.load(Ordering::Relaxed), BYTES.load(Ordering::Relaxed));
    let t0 = Instant::now();
    let d = self_play(net, games, 99, 1, &cfg).expect("self-play");
    let n = d.z.len();
    let secs = t0.elapsed().as_secs_f64();
    let (a, b) = (ALLOCS.load(Ordering::Relaxed) - a0, BYTES.load(Ordering::Relaxed) - b0);
    println!("{games} games, {} examples, {secs:.1}s single-threaded ({:.2} ms/example)", n, 1000.0 * secs / n as f64);
    println!("allocations: {a} ({:.0} per example, {:.1} M/s), {:.1} GB allocated ({:.0} KB per example, avg {:.0} B each)",
             a as f64 / n as f64, a as f64 / secs / 1e6, b as f64 / 1e9, b as f64 / n as f64 / 1e3, b as f64 / a as f64);
    let names = ["<=64B", "<=256B", "<=1KB", "<=4KB", "<=16KB", "<=64KB", ">64KB"];
    for (i, name) in names.iter().enumerate() {
        let (c, b) = (HIST[2 * i].load(Ordering::Relaxed), HIST[2 * i + 1].load(Ordering::Relaxed));
        println!("  {name:7} {:6.1}% of allocations, {:5.1}% of bytes", 100.0 * c as f64 / a as f64, 100.0 * b as f64 / BYTES.load(Ordering::Relaxed) as f64);
    }
    let traces = TRACES.lock().unwrap();
    if !traces.is_empty() {
        let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for t in traces.iter() {
            *counts.entry(t.as_str()).or_default() += 1;
        }
        let mut top: Vec<(&str, usize)> = counts.into_iter().collect();
        top.sort_by(|a, b| b.1.cmp(&a.1));
        println!("allocation sites (1 in {SAMPLE} sampled, {} samples):", traces.len());
        for (site, c) in top.iter().take(25) {
            println!("  {:5.1}%  {site}", 100.0 * *c as f64 / traces.len() as f64);
        }
    }
}
