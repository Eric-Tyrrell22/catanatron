//! First placements by several networks on one board: the first seat's first
//! settlement, and the second seat's reply to it (each network replying to its
//! own first pick). Network only (top move). A key may be "main.ctnn@opening.ctnn".
//!
//!   cargo run --release --example first_pick -- <board seed> <label=net.ctnn> [<label=net.ctnn> ...]
use catanatron::map::{build_map, MapType, NumberPlacement};
use catanatron::players::from_key;
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let board: u64 = args[1].parse().expect("board seed");
    for spec in &args[2..] {
        let (label, path) = spec.split_once('=').expect("label=net");
        let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(board));
        let rules = GameConfig {
            map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
            dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(1),
            catan_map: Some(map), record_history: false, fixed_seats: true, ..Default::default()
        };
        let mut game = Game::new(&COLORS[..2], &rules);
        let mut bots: Vec<Box<dyn Player>> = COLORS[..2].iter().enumerate().map(|(i, &c)| from_key(&format!("NN:{path}"), c, 1 + i as u64).expect("net")).collect();
        let mut picks = Vec::new();
        while game.state.is_initial_build_phase && picks.len() < 4 {
            let actions = game.playable_actions.clone();
            let seat = if game.state.current_color() == COLORS[0] { 0 } else { 1 };
            let action = if actions.len() == 1 { actions[0] } else { bots[seat].decide(&game, &actions) };
            picks.push(format!("{action:?}"));
            game.execute(action).expect("legal");
        }
        println!("{board} {label:<14} {}", picks.join(" | "));
    }
}
