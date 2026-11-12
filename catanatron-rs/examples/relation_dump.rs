//! The opening-relation block's board constants per land node, for the
//! spot-net experiments (runs/spot_fit.py --relation): for each board seed read
//! from stdin prints "#board <seed>" then "<node> v0 .. v8" (plan, greedy
//! opponent plan, greedy loss, plan cover, greedy cover, pips x scarcity, hit,
//! pip variance, 5-6-8-9 cover), as the network sees them at the first pick.
//!
//!   cargo run --release --example relation_dump < seeds.txt > relation.txt
use catanatron::encoding::{relation_offset_of, ObsFeatures, ObservationSpace, RELATION_CONST, RELATION_PER_NODE};
use catanatron::map::{build_map, MapType, NumberPlacement};
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::*;
use std::io::{self, BufRead, Write};

fn main() {
    let stdin = io::stdin();
    let out = io::stdout();
    let mut out = io::BufWriter::new(out.lock());
    for line in stdin.lock().lines() {
        let seed: u64 = match line.unwrap().trim().parse() { Ok(s) => s, Err(_) => continue };
        let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(seed));
        let rules = GameConfig {
            map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
            dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(1),
            catan_map: Some(map.clone()), record_history: false, fixed_seats: true, ..Default::default()
        };
        let game = Game::new(&COLORS[..2], &rules);
        let obs_space = ObservationSpace::with_features(&map, ObsFeatures { relation: true, ..Default::default() });
        let off = relation_offset_of(&obs_space).expect("relation block");
        let mut obs = vec![0.0f32; obs_space.len()];
        obs_space.encode_into(&game.state, game.state.current_color(), &mut obs);
        let n_land = map.land_nodes.iter().filter(|&&x| x).count();
        writeln!(out, "#board {seed}").unwrap();
        for nd in 0..n_land {
            let v: Vec<String> = (0..RELATION_CONST).map(|j| format!("{:.4}", obs[off + nd * RELATION_PER_NODE + j])).collect();
            writeln!(out, "{nd} {}", v.join(" ")).unwrap();
        }
    }
}
