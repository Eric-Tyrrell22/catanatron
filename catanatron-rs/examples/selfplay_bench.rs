//! Self-play throughput with the training settings (the actors' `az.py`
//! flags: OpenHex 1v1, 400/50 sims, 2400 placement sims, endgame and
//! opening branches, a placement fork). Game `i` depends only on the seed,
//! so a run is reproducible whatever the thread count.
//!   cargo run --release --example selfplay_bench -- <model.ctnn> [games] [threads] [seed]
use std::time::Instant;

use catanatron::map::{MapType, NumberPlacement};
use catanatron::mcts::{load_net, Utility};
use catanatron::selfplay::{self_play, SelfPlayConfig};
use catanatron::state::DiceMode;
use catanatron::*;

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args.next().expect("model path");
    let games: u64 = args.next().map_or(16, |g| g.parse().expect("games"));
    let threads: usize = args.next().map_or(4, |t| t.parse().expect("threads"));
    let seed: u64 = args.next().map_or(99, |s| s.parse().expect("seed"));
    let net = load_net(&model).expect("model");
    let cfg = SelfPlayConfig {
        sims_full: 400, sims_fast: 50, sims_placement: 2400, p_full: 0.25,
        entropy_budget: false, kl_escalate: false, admit_kl: 0.0, focus_sims: 0, temp_moves: 15, root_noise: 0.25,
        dirichlet_alpha: 0.3, c_puct: 1.0, fpu_reduction: 0.25, escalate_gap: 0.0, endgame_vp: 13, endgame_branches: 1,
        opening_branches: 1, placement_forks: 1, placement_temp: 2.0, placement_uniform: 0.25, fork_settlements_only: false, placement_margin: 0.0, placement_fork_weights: [1.0; 4], move_forks: 0, move_fork_temp: 1.0,
        move_fork_uniform: 0.0, sibling_labels: false, pool_board_share: 1.0, pool_seeds: Vec::new(), random_share: 0.0,
        random_eps: 1.0, utility: Utility { margin: 0.1, speed: 0.3, speed_turns: 150.0 }, gumbel_m: 0,
        game: GameConfig {
            map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
            dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, ..Default::default()
        },
    };
    let t0 = Instant::now();
    let d = self_play(net, games, seed, threads, &cfg).expect("self-play");
    let secs = t0.elapsed().as_secs_f64();
    let n = d.len();
    // Checksum of the recorded targets: the same code must give the same games.
    let sum: f64 = d.z.iter().map(|&z| z as f64).sum::<f64>() + d.q.iter().map(|&q| q as f64).sum::<f64>();
    println!(
        "{games} games ({} branches, {} forks) on {threads} threads: {secs:.2}s = {:.3} games/s ({:.3} per thread)",
        d.branches, d.forks, games as f64 / secs, games as f64 / secs / threads as f64
    );
    println!(
        "  {n} examples ({:.1} ms each per thread), {} decisions, {} sims ({:.0}/s, {:.0}/s per thread), {} evals ({:.0}/s per thread)",
        1000.0 * secs * threads as f64 / n as f64, d.decisions, d.sims,
        d.sims as f64 / secs, d.sims as f64 / secs / threads as f64, d.evals, d.evals as f64 / secs / threads as f64
    );
    println!("  results {:?}, checksum {sum:.4}", d.results);
}
