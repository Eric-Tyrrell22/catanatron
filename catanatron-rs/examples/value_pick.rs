//! One-ply value view of the first placement: for a board and each network,
//! the network's value (for the first seat, read right after placing) of every
//! legal first settlement. A "main.ctnn@opening.ctnn" key evaluates with its
//! opening network. One "#value <label> node:value ..." line per network.
//!
//!   cargo run --release --example value_pick -- <board seed> <label=net.ctnn> [...]
use catanatron::encoding::ObservationSpace;
use catanatron::map::{build_map, MapType, NumberPlacement};
use catanatron::mcts::load_net;
use catanatron::nn::Scratch;
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::*;

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
    for spec in &args[2..] {
        let (label, path) = spec.split_once('=').expect("label=net");
        let net = load_net(path).expect("net");
        let net = net.opening.as_deref().unwrap_or(&net);
        let obs_space = ObservationSpace::for_network(&map, net.obs_size);
        let mut obs = vec![0.0f32; net.obs_size];
        let (mut scratch, mut logits) = (Scratch::default(), Vec::new());
        let mut items = Vec::new();
        for a in &game.playable_actions {
            if let ActionKind::BuildSettlement(node) = a.kind {
                let mut g = game.clone();
                g.execute(*a).expect("legal");
                // Still the first seat's decision (its road), so the value is for the first seat.
                let s = &g.state;
                obs_space.encode_into(s, s.current_color(), &mut obs);
                let v = net.forward(&obs, &mut scratch, &mut logits);
                items.push((node, v));
            }
        }
        items.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!("#value {label} {}", items.iter().map(|(n, v)| format!("{n}:{v:.4}")).collect::<Vec<_>>().join(" "));
    }
}
