//! Does scarcity predict expert first picks? For each (board, specialist) pair:
//! the specialist's first settlement (network top move), a reference network's,
//! and for every land node its raw pips and its scarcity-weighted pips (pips in
//! each resource divided by the board's total pips of that resource, summed),
//! so a script can rank the picks under both scores. One TSV row per board.
//!
//!   cargo run --release --example pick_features -- <reference.ctnn> <board>=<specialist.ctnn> [...]
use catanatron::map::{build_map, MapType, NumberPlacement};
use catanatron::players::from_key;
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::*;

fn first_pick(path: &str, map: &std::sync::Arc<catanatron::map::CatanMap>) -> u8 {
    let rules = GameConfig {
        map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
        dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(1),
        catan_map: Some(map.clone()), record_history: false, fixed_seats: true, ..Default::default()
    };
    let game = Game::new(&COLORS[..2], &rules);
    let mut bot = from_key(&format!("NN:{path}"), COLORS[0], 1).expect("net");
    let actions = game.playable_actions.clone();
    match bot.decide(&game, &actions).kind {
        ActionKind::BuildSettlement(n) => n,
        other => panic!("first action is not a settlement: {other:?}"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reference = &args[1];
    println!("board\tspec_pick\tref_pick\tnode:raw:scarce...");
    for spec in &args[2..] {
        let (board, path) = spec.split_once('=').expect("board=net");
        let board: u64 = board.parse().expect("board seed");
        let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(board));
        let pips = |n: u8| (6 - (7 - n as i32).abs()).max(0) as f32;
        let mut board_pips = [0.0f32; 5];
        for &ti in &map.land_tiles {
            let t = &map.tiles[ti];
            if let (Some(r), Some(num)) = (t.resource(), t.number()) {
                board_pips[r.idx()] += pips(num);
            }
        }
        let mut cols = Vec::new();
        for n in 0..map.num_nodes {
            if !map.land_nodes[n] {
                continue;
            }
            let (mut raw, mut scarce) = (0.0f32, 0.0f32);
            for &ti in &map.adjacent_tiles[n] {
                let t = &map.tiles[ti];
                if let (Some(r), Some(num)) = (t.resource(), t.number()) {
                    raw += pips(num);
                    scarce += pips(num) / board_pips[r.idx()].max(1.0);
                }
            }
            cols.push(format!("{n}:{raw:.0}:{scarce:.3}"));
        }
        let sp = first_pick(path, &map);
        let rp = first_pick(reference, &map);
        println!("{board}\t{sp}\t{rp}\t{}", cols.join(" "));
    }
}
