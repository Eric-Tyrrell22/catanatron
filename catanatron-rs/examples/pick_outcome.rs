//! Win rate by first pick: A (any player key, e.g. a search) against B on one
//! board, seats alternating per game; for each game A's first settlement node
//! and the result. Prints A's win rate per (seat, first pick).
//!
//!   cargo run --release --example pick_outcome -- <A key> <B key> <board seed> [games] [threads]
//! Keys as for players::from_key, e.g. "MCTS:400,c1,f0.25:main.ctnn@opening.ctnn" or "NN:spec.ctnn".
use catanatron::map::{build_map, MapType, NumberPlacement};
use catanatron::parallel::{default_threads, par_map};
use catanatron::players::from_key;
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::*;
use std::collections::BTreeMap;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (a_key, b_key) = (args[1].clone(), args[2].clone());
    let board: u64 = args[3].parse().expect("board seed");
    let games: u64 = args.get(4).map_or(400, |s| s.parse().expect("games"));
    let threads: usize = args.get(5).map_or(default_threads(), |s| s.parse().expect("threads"));
    let rows: Vec<(usize, u8, bool)> = par_map(games, threads, |g| {
        let seed = 950_000 + g;
        let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(board));
        let rules = GameConfig {
            map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
            dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(seed),
            catan_map: Some(map), record_history: false, fixed_seats: true, ..Default::default()
        };
        let mut game = Game::new(&COLORS[..2], &rules);
        let a_seat = (g % 2) as usize;
        let a_color = COLORS[a_seat];
        let mut a = from_key(&a_key, a_color, seed).expect("A");
        let mut b = from_key(&b_key, COLORS[1 - a_seat], seed + 1).expect("B");
        let mut first: Option<u8> = None;
        while !game.is_over() && game.state.num_turns < 400 {
            let actions = game.playable_actions.clone();
            let mover = game.state.current_color();
            let action = if actions.len() == 1 { actions[0] } else if mover == a_color { a.decide(&game, &actions) } else { b.decide(&game, &actions) };
            if first.is_none() && mover == a_color {
                if let ActionKind::BuildSettlement(n) = action.kind {
                    first = Some(n);
                }
            }
            game.execute(action).expect("legal");
        }
        (a_seat, first.unwrap_or(255), game.winning_color() == Some(a_color))
    });
    let mut by: BTreeMap<(usize, u8), (u32, u32)> = BTreeMap::new();
    for (seat, node, won) in &rows {
        let e = by.entry((*seat, *node)).or_default();
        e.0 += 1;
        e.1 += *won as u32;
    }
    let total: u32 = rows.iter().map(|r| r.2 as u32).sum();
    println!("board {board}: A wins {:.1}% of {games}", 100.0 * total as f32 / games as f32);
    for ((seat, node), (n, w)) in &by {
        println!("  A as {} seat, first pick node {node:3}: {n:4} games, wins {:.1}% (±{:.1})",
                 if *seat == 0 { "first " } else { "second" }, 100.0 * *w as f32 / *n as f32, 100.0 * (0.25 / *n as f32).sqrt());
    }
}
