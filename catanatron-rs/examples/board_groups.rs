//! Shares of random OpenHex boards in each group (`map::scarcest_resource`),
//! the weights group experts get in distillation.
//!
//!   cargo run --release --example board_groups -- [boards]
use catanatron::map::{build_map, scarcest_resource, MapType, NumberPlacement};
use catanatron::rng::Rng;
use catanatron::types::RESOURCES;

fn main() {
    let n: u64 = std::env::args().nth(1).map_or(100_000, |s| s.parse().expect("boards"));
    let mut counts = [0u64; 5];
    for seed in 0..n {
        // The draw `Game::new` makes for a game with this seed.
        let mut rng = Rng::new(seed);
        let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut rng);
        counts[scarcest_resource(&map).idx()] += 1;
    }
    for r in RESOURCES {
        println!("{:?}\t{:.4}", r, counts[r.idx()] as f64 / n as f64);
    }
}
