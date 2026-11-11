//! Prints in-memory sizes of the core state types and the cost of a copy.
use std::time::Instant;
use catanatron::board::Board;
use catanatron::state::{PlayerState, State};
use catanatron::*;

fn main() {
    println!("State       {:>6} bytes", std::mem::size_of::<State>());
    println!("  Board     {:>6} bytes", std::mem::size_of::<Board>());
    println!("  Player    {:>6} bytes (x4)", std::mem::size_of::<PlayerState>());
    let g = Game::new(&COLORS, &GameConfig { seed: Some(1), ..Default::default() });
    let n = 1_000_000;
    let t = Instant::now();
    let mut acc = 0u32;
    for _ in 0..n {
        let s = std::hint::black_box(&g.state).clone();
        acc = acc.wrapping_add(std::hint::black_box(s).num_turns);
    }
    println!("State clone {:>6.1} ns  ({acc})", t.elapsed().as_nanos() as f64 / n as f64);
}
