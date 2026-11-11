//! Game length distribution (turns) for a bot key against itself under
//! OpenHex 1v1 rules.
//!   cargo run --release --example game_lengths -- <bot key> [games]
use catanatron::map::{MapType, NumberPlacement};
use catanatron::players::{from_key, Player};
use catanatron::state::DiceMode;
use catanatron::*;

fn main() {
    let key = std::env::args().nth(1).expect("bot key");
    let games: u64 = std::env::args().nth(2).map_or(1000, |g| g.parse().unwrap());
    let mut lens: Vec<u32> = (0..games)
        .map(|seed| {
            let config = GameConfig {
                map_type: MapType::Base, vps_to_win: 15, discard_limit: 9,
                friendly_robber: FriendlyRobber::PublicPoints, dice: DiceMode::OpenHex,
                number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(seed), ..Default::default()
            };
            let mut g = Game::new(&COLORS[..2], &config);
            let mut bots: Vec<Box<dyn Player>> = COLORS[..2].iter().map(|&c| from_key(&key, c, seed).unwrap()).collect();
            g.play(&mut bots).unwrap();
            if g.winning_color().is_some() { g.state.num_turns } else { u32::MAX }
        })
        .collect();
    lens.sort();
    let n = lens.len();
    let pct = |p: f64| lens[((n as f64 - 1.0) * p) as usize];
    let show = |v: u32| if v == u32::MAX { "capped".to_string() } else { v.to_string() };
    let over = |t: u32| 100.0 * lens.iter().filter(|&&l| l > t).count() as f64 / n as f64;
    println!("{n} games: median {} | p90 {} | p99 {} | p99.9 {} | max {}", show(pct(0.5)), show(pct(0.9)), show(pct(0.99)), show(pct(0.999)), show(lens[n - 1]));
    println!("  longer than 200 turns: {:.2}% | 300: {:.2}% | 400: {:.2}% | 1000 (no winner): {:.2}%", over(200), over(300), over(400), over(999));
}
