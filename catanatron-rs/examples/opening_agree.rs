//! First placements across many boards: does the network's top first
//! settlement agree with its own deep search, and what does the
//! disagreement cost? One line per (board, seat): the seat, agree,
//! regret (search Q of its best spot minus of the network's top spot),
//! the network's prior on its top spot, and its normalized entropy.
//!
//!   cargo run --release --example opening_agree -- <model.ctnn> [boards] [sims] [threads] > rows.tsv
use catanatron::map::{MapType, NumberPlacement};
use catanatron::mcts::{load_net, MctsConfig, MctsPlayer, Utility};
use catanatron::parallel::{default_threads, par_map};
use catanatron::state::DiceMode;
use catanatron::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let net = load_net(&args[1]).expect("model");
    let boards: u64 = args.get(2).map_or(500, |s| s.parse().expect("boards"));
    let sims: u32 = args.get(3).map_or(2400, |s| s.parse().expect("sims"));
    let threads: usize = args.get(4).map_or(default_threads(), |s| s.parse().expect("threads"));
    let cfg = MctsConfig { simulations: sims, c_puct: 1.0, fpu_reduction: 0.25, utility: Utility { margin: 0.1, speed: 0.3, speed_turns: 150.0 }, ..Default::default() };
    println!("board\tseat\tn_legal\tagree\tregret\tprior_top\tprior_search_best\tentropy");
    let rows = par_map(boards, threads, |bseed| {
        let rules = GameConfig {
            map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
            dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(700_000 + bseed),
            record_history: false, ..Default::default()
        };
        let mut game = Game::new(&COLORS[..2], &rules);
        let mut out = String::new();
        let mut k = 0u64;
        // Seat 0's first settlement and road (search picks), then seat 1's first settlement.
        while game.state.is_initial_build_phase && k < 3 {
            let actions = game.playable_actions.clone();
            let c = game.state.current_color();
            let mut m = MctsPlayer::new(c, net.clone(), cfg.clone(), bseed * 10 + k);
            let visits = m.search(&game).expect("search");
            let (stats, _) = m.root_stats();
            let best = (0..visits.len()).max_by_key(|&j| visits[j]).unwrap();
            if matches!(actions[0].kind, ActionKind::BuildSettlement(_)) {
                let top = (0..stats.len()).max_by(|&a, &b| stats[a].0.total_cmp(&stats[b].0)).unwrap();
                let q_best = stats[best].2.unwrap_or(0.0);
                let q_top = stats[top].2.unwrap_or_else(|| stats.iter().filter_map(|s| s.2).fold(f32::INFINITY, f32::min));
                let sum: f32 = stats.iter().map(|s| s.0).sum();
                let ent = -stats.iter().map(|s| s.0 / sum).filter(|&p| p > 0.0).map(|p| p * p.ln()).sum::<f32>() / (actions.len() as f32).ln();
                out.push_str(&format!("{bseed}\t{}\t{}\t{}\t{:.4}\t{:.3}\t{:.3}\t{:.3}\n", game.state.seat(c), actions.len(), (top == best) as u8, (q_best - q_top).max(0.0), stats[top].0 / sum, stats[best].0 / sum, ent));
            }
            game.execute(actions[best]).expect("legal");
            k += 1;
        }
        out
    });
    for r in rows {
        print!("{r}");
    }
}
