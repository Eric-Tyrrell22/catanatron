//! Observations after each legal first settlement, as the network that would
//! value them sees them: for a board and a "main.ctnn@opening.ctnn" key, one
//! "node <n> <obs floats>" line per legal first settlement, encoded for the
//! opening network (the main network without one) from the first seat's view
//! right after the placement. Feeds runs/value_fit_test.py, which fits the
//! value head to forced-play win rates.
//!
//!   cargo run --release --example forced_obs -- <board seed> <net key>
use catanatron::encoding::{ActionSpace, ObservationSpace};
use catanatron::map::{build_map, MapType, NumberPlacement};
use catanatron::mcts::load_net;
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::*;
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let board: u64 = args[1].parse().expect("board seed");
    let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(board));
    let rules = GameConfig {
        map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
        dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(1),
        catan_map: Some(map.clone()), record_history: false, fixed_seats: true, ..Default::default()
    };
    let game = Game::new(&COLORS[..2], &rules);
    let net = load_net(&args[2]).expect("net");
    let net = net.opening.as_deref().unwrap_or(&net);
    let obs_space = ObservationSpace::for_network(&map, net.obs_size);
    let mut obs = vec![0.0f32; net.obs_size];
    let out = std::io::stdout();
    let mut out = std::io::BufWriter::new(out.lock());
    writeln!(out, "#obs_size {}", net.obs_size).unwrap();
    // The pre-pick view (the first seat deciding) and each legal first pick's policy index, for policy targets.
    let act = ActionSpace::new(&map);
    obs_space.encode_into(&game.state, game.state.current_color(), &mut obs);
    write!(out, "#pre").unwrap();
    for v in &obs {
        write!(out, " {v}").unwrap();
    }
    writeln!(out).unwrap();
    for a in &game.playable_actions {
        if let ActionKind::BuildSettlement(node) = a.kind {
            writeln!(out, "#slot {node} {}", act.encode(&game.state, a).expect("encodable")).unwrap();
        }
    }
    for a in &game.playable_actions {
        if let ActionKind::BuildSettlement(node) = a.kind {
            let mut g = game.clone();
            g.execute(*a).expect("legal");
            let s = &g.state;
            obs_space.encode_into(s, s.current_color(), &mut obs);
            write!(out, "node {node}").unwrap();
            for v in &obs {
                write!(out, " {v}").unwrap();
            }
            writeln!(out).unwrap();
        }
    }
}
