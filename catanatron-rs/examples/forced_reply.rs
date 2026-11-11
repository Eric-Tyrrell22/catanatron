//! Win rate by forced reply: on one board, A (first seat) has its first
//! settlement forced to a given node (its road and everything after by its own
//! key), B (second seat) has its first settlement forced to each listed node
//! and plays the rest by its key. Network players are deterministic, the dice
//! vary per game. Prints B's win rate per forced reply, so the replies of two
//! policies to the same first pick can be compared under the same play.
//!
//!   cargo run --release --example forced_reply -- <A key> <B key> <board seed> <A's first node> <reply,reply,...> [games] [threads]
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
    let first: u8 = args[4].parse().expect("A's first node");
    let replies: Vec<u8> = args[5].split(',').map(|s| s.parse().expect("node id")).collect();
    let games: u64 = args.get(6).map_or(400, |s| s.parse().expect("games"));
    let threads: usize = args.get(7).map_or(default_threads(), |s| s.parse().expect("threads"));
    for &reply in &replies {
        let wins: Vec<u32> = par_map(games, threads, |g| {
            let seed = 970_000 + g;
            let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(board));
            let rules = GameConfig {
                map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
                dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(seed),
                catan_map: Some(map), record_history: false, fixed_seats: true, ..Default::default()
            };
            let mut game = Game::new(&COLORS[..2], &rules);
            let mut a = from_key(&a_key, COLORS[0], seed).expect("A");
            let mut b = from_key(&b_key, COLORS[1], seed + 1).expect("B");
            let (mut a_first, mut b_first) = (true, true);
            while !game.is_over() && game.state.num_turns < 400 {
                let actions = game.playable_actions.clone();
                let mover = game.state.current_color();
                let forced = if a_first && mover == COLORS[0] {
                    a_first = false;
                    Some(first)
                } else if b_first && mover == COLORS[1] {
                    b_first = false;
                    Some(reply)
                } else {
                    None
                };
                let action = if let Some(n) = forced {
                    *actions.iter().find(|a| matches!(a.kind, ActionKind::BuildSettlement(k) if k == n)).expect("forced node is a legal settlement")
                } else if actions.len() == 1 {
                    actions[0]
                } else if mover == COLORS[0] {
                    a.decide(&game, &actions)
                } else {
                    b.decide(&game, &actions)
                };
                game.execute(action).expect("legal");
            }
            (game.winning_color() == Some(COLORS[1])) as u32
        });
        let w: u32 = wins.iter().sum();
        println!("board {board} first {first:3} reply {reply:3}: B (second seat) wins {:.1}% of {games} (±{:.1})", 100.0 * w as f32 / games as f32, 100.0 * (0.25 / games as f32).sqrt());
    }
}
