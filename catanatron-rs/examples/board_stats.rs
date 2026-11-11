//! How often real boards have each `BoardStyle` property (the rejection rate
//! of a board mix).
use catanatron::map::{build_map, BoardStyle, MapType, NumberPlacement};
use catanatron::rng::Rng;

fn main() {
    let n = 20000u64;
    for style in [BoardStyle::ScarceOws, BoardStyle::PortStart] {
        let hits = (0..n)
            .filter(|&seed| style.matches(&build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(seed))))
            .count();
        println!("{style:?}: {:.1}% of real boards", 100.0 * hits as f64 / n as f64);
    }
}
