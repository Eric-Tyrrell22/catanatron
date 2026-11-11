//! Where does search go at the first placement? For a board and a network,
//! searches the empty-board first settlement with each simulation count and
//! prints the top settlements by visits (share of visits), so one can see how
//! deep a search must be before its choice changes.
//!
//!   cargo run --release --example placement_search -- <board seed> <net key path> <sims,sims,...> [threads]
use catanatron::map::{build_map, MapType, NumberPlacement};
use catanatron::mcts::{load_net, MctsConfig, MctsPlayer, Utility};
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let board: u64 = args[1].parse().expect("board seed");
    let net = load_net(&args[2]).expect("net");
    let sims: Vec<u32> = args[3].split(',').map(|s| s.parse().expect("sims")).collect();
    let threads: usize = args.get(4).map_or(1, |s| s.parse().expect("threads"));
    let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(board));
    let rules = GameConfig {
        map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
        dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(1),
        catan_map: Some(map), record_history: false, fixed_seats: true, ..Default::default()
    };
    let game = Game::new(&COLORS[..2], &rules);
    for &n in &sims {
        let config = MctsConfig { simulations: n, c_puct: 1.0, fpu_reduction: 0.25, utility: Utility { margin: 0.1, speed: 0.3, speed_turns: 150.0 },
                                  ..Default::default() };
        let _ = threads;
        let mut player = MctsPlayer::new(COLORS[0], net.clone(), config, 7);
        let t0 = std::time::Instant::now();
        let visits = player.search(&game).expect("search");
        let total: u32 = visits.iter().sum();
        let mut items: Vec<(u8, u32)> = game.playable_actions.iter().zip(&visits).filter_map(|(a, &v)| match a.kind {
            ActionKind::BuildSettlement(node) => Some((node, v)),
            _ => None,
        }).collect();
        items.sort_by_key(|&(_, v)| std::cmp::Reverse(v));
        println!("{n:>7} sims ({:.1}s): {}", t0.elapsed().as_secs_f32(),
                 items.iter().take(5).map(|(node, v)| format!("node {node} {:.1}%", 100.0 * *v as f32 / total.max(1) as f32)).collect::<Vec<_>>().join("  "));
    }
}
