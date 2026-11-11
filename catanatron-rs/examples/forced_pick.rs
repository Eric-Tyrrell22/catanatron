//! Win rate by forced first pick: on one board, A (always the first seat) has
//! its first settlement forced to each listed node and plays the rest of the
//! game by its own key; B (second seat) plays by its key throughout. Network
//! players are deterministic, the dice vary per game. Prints A's win rate per
//! forced node, so the picks of two policies can be compared under the same
//! play. Keys as for players::from_key, e.g. "NN:spec.ctnn".
//!
//!   cargo run --release --example forced_pick -- <A key> <B key> <board seed> <node,node,...> [games] [threads]
use catanatron::map::{build_map, MapType, NumberPlacement};
use catanatron::parallel::{default_threads, par_map};
use catanatron::players::from_key;
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (a_key, b_key) = (args[1].clone(), args[2].clone());
    let board: u64 = args[3].parse().expect("board seed");
    let nodes: Vec<u8> = args[4].split(',').map(|s| s.parse().expect("node id")).collect();
    let games: u64 = args.get(5).map_or(400, |s| s.parse().expect("games"));
    let threads: usize = args.get(6).map_or(default_threads(), |s| s.parse().expect("threads"));
    for &forced in &nodes {
        let wins: Vec<u32> = par_map(games, threads, |g| {
            let seed = 960_000 + g;
            let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(board));
            let rules = GameConfig {
                map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
                dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(seed),
                catan_map: Some(map), record_history: false, fixed_seats: true, ..Default::default()
            };
            let mut game = Game::new(&COLORS[..2], &rules);
            let mut a = from_key(&a_key, COLORS[0], seed).expect("A");
            let mut b = from_key(&b_key, COLORS[1], seed + 1).expect("B");
            let mut first = true;
            while !game.is_over() && game.state.num_turns < 400 {
                let actions = game.playable_actions.clone();
                let mover = game.state.current_color();
                let action = if first && mover == COLORS[0] {
                    first = false;
                    *actions.iter().find(|a| matches!(a.kind, ActionKind::BuildSettlement(n) if n == forced)).expect("forced node is a legal first settlement")
                } else if actions.len() == 1 {
                    actions[0]
                } else if mover == COLORS[0] {
                    a.decide(&game, &actions)
                } else {
                    b.decide(&game, &actions)
                };
                game.execute(action).expect("legal");
            }
            (game.winning_color() == Some(COLORS[0])) as u32
        });
        let w: u32 = wins.iter().sum();
        println!("board {board} forced node {forced:3}: A (first seat) wins {:.1}% of {games} (±{:.1})", 100.0 * w as f32 / games as f32, 100.0 * (0.25 / games as f32).sqrt());
    }
}
