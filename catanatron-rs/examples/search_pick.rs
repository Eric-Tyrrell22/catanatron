//! Does more search improve the first pick? For each board seed, searches the
//! first settlement of the first seat with a network at several simulation
//! budgets (PUCT, no root noise, the self-play utility) and prints the root
//! visit counts per node, so runs/search_curve.py can score each budget's
//! argmax against the forced win rates of the expert boards.
//!
//!   cargo run --release --example search_pick -- <main.ctnn[@opening.ctnn]> <sims,sims,...> <board seed> [board seed ...] [--threads N] [--noise]
//! `--noise` adds the self-play root noise (Dirichlet 0.3, weight 0.25), as in training.
//! Output: "board B sims S: node:visits node:visits ..." (nodes sorted by visits)
use catanatron::map::{build_map, MapType, NumberPlacement};
use catanatron::mcts::{load_net, MctsConfig, MctsPlayer};
use catanatron::parallel::{default_threads, par_map};
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::*;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let threads = match args.iter().position(|a| a == "--threads") {
        Some(i) => { let t = args[i + 1].parse().expect("threads"); args.drain(i..i + 2); t }
        None => default_threads(),
    };
    let noise = match args.iter().position(|a| a == "--noise") {
        Some(i) => { args.remove(i); 0.25 }
        None => 0.0,
    };
    let net = load_net(&args[0]).expect("network");
    let sims: Vec<u32> = args[1].split(',').map(|s| s.parse().expect("sims")).collect();
    let boards: Vec<u64> = args[2..].iter().map(|s| s.parse().expect("board seed")).collect();
    let jobs: Vec<(u64, u32)> = boards.iter().flat_map(|&b| sims.iter().map(move |&s| (b, s))).collect();
    let lines = par_map(jobs.len() as u64, threads, |j| {
        let (board, s) = jobs[j as usize];
        let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(board));
        let rules = GameConfig {
            map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
            dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(1),
            catan_map: Some(map.clone()), record_history: false, fixed_seats: true, ..Default::default()
        };
        let game = Game::new(&COLORS[..2], &rules);
        let mut config = MctsConfig { simulations: s, c_puct: 1.0, fpu_reduction: 0.25, root_noise: noise, dirichlet_alpha: 0.3, ..Default::default() };
        config.utility.margin = 0.1;
        config.utility.speed = 0.3;
        config.utility.speed_turns = 150.0;
        let mut player = MctsPlayer::new(COLORS[0], net.clone(), config, board);
        let visits = player.search(&game).expect("search");
        let mut rows: Vec<(u32, u8)> = game.playable_actions.iter().zip(&visits)
            .filter_map(|(a, &v)| match a.kind { ActionKind::BuildSettlement(n) => Some((v, n)), _ => None }).collect();
        rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        format!("board {board} sims {s}: {}", rows.iter().map(|(v, n)| format!("{n}:{v}")).collect::<Vec<_>>().join(" "))
    });
    for l in lines {
        println!("{l}");
    }
}
