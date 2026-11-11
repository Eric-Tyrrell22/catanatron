//! Seat bias: network-only games with A always in the first seat and B in the
//! second (A may equal B for self-play), on one board or on random boards.
//! Prints the first seat's win rate. A key may be "main.ctnn@opening.ctnn".
//!
//!   cargo run --release --example seat_bias -- <A.ctnn> <B.ctnn> <board seed | random> [games] [threads]
use catanatron::map::{build_map, MapType, NumberPlacement};
use catanatron::parallel::{default_threads, par_map};
use catanatron::players::from_key;
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (a_key, b_key) = (format!("NN:{}", args[1]), format!("NN:{}", args[2]));
    let board: Option<u64> = args[3].parse().ok();
    let games: u64 = args.get(4).map_or(2000, |s| s.parse().expect("games"));
    let threads: usize = args.get(5).map_or(default_threads(), |s| s.parse().expect("threads"));
    let wins: Vec<u32> = par_map(games, threads, |g| {
        let seed = 940_000 + g;
        let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(board.unwrap_or(seed)));
        let rules = GameConfig {
            map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
            dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(seed),
            catan_map: Some(map), record_history: false, fixed_seats: true, ..Default::default()
        };
        let mut game = Game::new(&COLORS[..2], &rules);
        let mut a = from_key(&a_key, COLORS[0], seed).expect("A");
        let mut b = from_key(&b_key, COLORS[1], seed + 1).expect("B");
        while !game.is_over() && game.state.num_turns < 400 {
            let actions = game.playable_actions.clone();
            let action = if actions.len() == 1 {
                actions[0]
            } else if game.state.current_color() == COLORS[0] {
                a.decide(&game, &actions)
            } else {
                b.decide(&game, &actions)
            };
            game.execute(action).expect("legal");
        }
        (game.winning_color() == Some(COLORS[0])) as u32
    });
    let w: u32 = wins.iter().sum();
    println!("{}: first seat (A) wins {:.1}% of {games} (±{:.1})", args[3], 100.0 * w as f32 / games as f32, 100.0 * (0.25 / games as f32).sqrt());
}
