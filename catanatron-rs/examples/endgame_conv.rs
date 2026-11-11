//! Endgame conversion: network-only games A vs B on one board (seats alternate
//! per game). For each game: who first reached `vp` points and who won. Prints
//! each side's conversion rate (wins / games it was first to `vp`), the mean
//! number of turns from the threshold to the end, and how often the other side
//! came back. A key may be "main.ctnn@opening.ctnn".
//!
//!   cargo run --release --example endgame_conv -- <A.ctnn> <B.ctnn> <board seed> [games] [threads] [vp]
use catanatron::map::{build_map, MapType, NumberPlacement};
use catanatron::parallel::{default_threads, par_map};
use catanatron::players::from_key;
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (a_key, b_key) = (format!("NN:{}", args[1]), format!("NN:{}", args[2]));
    let board: u64 = args[3].parse().expect("board seed");
    let games: u64 = args.get(4).map_or(2000, |s| s.parse().expect("games"));
    let threads: usize = args.get(5).map_or(default_threads(), |s| s.parse().expect("threads"));
    let vp: u32 = args.get(6).map_or(13, |s| s.parse().expect("vp"));
    // Per game: (A was first to vp, turns from the threshold to the end) if anyone reached it; A won; game finished.
    let rows: Vec<(Option<(bool, u32)>, bool, bool)> = par_map(games, threads, |g| {
        let seed = 930_000 + g;
        let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(board));
        let rules = GameConfig {
            map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
            dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(seed),
            catan_map: Some(map.clone()), record_history: false, fixed_seats: true, ..Default::default()
        };
        let mut game = Game::new(&COLORS[..2], &rules);
        let a_color = COLORS[(g % 2) as usize];
        let b_color = COLORS[((g + 1) % 2) as usize];
        let mut a = from_key(&a_key, a_color, seed).expect("A");
        let mut b = from_key(&b_key, b_color, seed + 1).expect("B");
        let mut first: Option<(Color, u32)> = None;
        while !game.is_over() && game.state.num_turns < 400 {
            let mover = game.state.current_color();
            let actions = game.playable_actions.clone();
            let action = if actions.len() == 1 {
                actions[0]
            } else if mover == a_color {
                a.decide(&game, &actions)
            } else {
                b.decide(&game, &actions)
            };
            game.execute(action).expect("legal");
            if first.is_none() {
                for &c in &COLORS[..2] {
                    if game.state.player(c).actual_victory_points >= vp {
                        first = Some((c, game.state.num_turns));
                        break;
                    }
                }
            }
        }
        let winner = game.winning_color();
        (first.map(|(c, t)| (c == a_color, game.state.num_turns - t)), winner == Some(a_color), winner.is_some())
    });
    let finished = rows.iter().filter(|r| r.2).count();
    let a_wins = rows.iter().filter(|r| r.1).count();
    println!("board {board}: {games} games, {finished} finished; A wins {:.1}%", 100.0 * a_wins as f32 / games as f32);
    for (label, side_a) in [("A", true), ("B", false)] {
        let firsts: Vec<&(Option<(bool, u32)>, bool, bool)> = rows.iter().filter(|r| r.0.map_or(false, |(fa, _)| fa == side_a)).collect();
        let n = firsts.len();
        let converted = firsts.iter().filter(|r| r.1 == side_a && r.2).count();
        let turns: f32 = firsts.iter().map(|r| r.0.unwrap().1 as f32).sum::<f32>() / n.max(1) as f32;
        println!("  {label} first to {vp}: {n:5} games -> converted {:.1}% (±{:.1}), {:.1} turns from {vp} to the end",
                 100.0 * converted as f32 / n.max(1) as f32, 100.0 * (0.25 / n.max(1) as f32).sqrt(), turns);
    }
}
