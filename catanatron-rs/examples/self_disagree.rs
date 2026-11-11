//! Where does a network disagree with its own deep search? Plays the network
//! against itself (moves chosen by a modest search, as in play) and at every
//! decision with a real choice also runs a deep search, recording whether
//! the network's top move is the deep search's, and the regret (deep
//! search's value of its best move minus of the network's choice). Tagged by
//! phase (by the mover's points) and move type, so the losses can be
//! attributed.
//!
//!   cargo run --release --example self_disagree -- <model.ctnn> [games] [deep sims] [threads] > disagree.tsv
//!
//! One line per decision: game, turn, mover_vps, opp_vps, phase, kind,
//! n_legal, agree, regret, net_prior_of_best, prompt.
use catanatron::map::{MapType, NumberPlacement};
use catanatron::mcts::{load_net, MctsConfig, MctsPlayer, Utility};
use catanatron::parallel::{default_threads, par_map};
use catanatron::state::DiceMode;
use catanatron::*;

fn kind(a: &Action) -> &'static str {
    match a.kind {
        ActionKind::Roll => "roll",
        ActionKind::MoveRobber { .. } => "robber",
        ActionKind::DiscardResource(_) => "discard",
        ActionKind::BuildRoad(_) => "road",
        ActionKind::BuildSettlement(_) => "settlement",
        ActionKind::BuildCity(_) => "city",
        ActionKind::BuyDevelopmentCard => "buy_dev",
        ActionKind::PlayKnightCard => "knight",
        ActionKind::PlayYearOfPlenty(..) => "yop",
        ActionKind::PlayMonopoly(_) => "monopoly",
        ActionKind::PlayRoadBuilding => "road_building",
        ActionKind::MaritimeTrade { .. } => "trade",
        ActionKind::EndTurn => "end_turn",
        _ => "other",
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = args.get(1).expect("model path");
    let games: u64 = args.get(2).map_or(100, |s| s.parse().expect("games"));
    let deep: u32 = args.get(3).map_or(4000, |s| s.parse().expect("deep sims"));
    let threads: usize = args.get(4).map_or(default_threads(), |s| s.parse().expect("threads"));
    let net = load_net(model).expect("model");
    let live = MctsConfig { simulations: 400, c_puct: 1.0, fpu_reduction: 0.25, utility: Utility { margin: 0.1, speed: 0.3, speed_turns: 150.0 }, ..Default::default() };
    let deep_cfg = MctsConfig { simulations: deep, ..live.clone() };
    println!("game\tturn\tmover_vps\topp_vps\tphase\tkind\tnet_kind\tn_legal\tagree\tregret\tprior_best\tentropy\tprompt");
    let rows = par_map(games, threads, |g| {
        let seed = 500_000 + g;
        let rules = GameConfig {
            map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
            dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(seed),
            record_history: false, ..Default::default()
        };
        let mut game = Game::new(&COLORS[..2], &rules);
        let mut out = String::new();
        let mut k = 0u64;
        while !game.is_over() && game.state.num_turns < 400 {
            let actions = game.playable_actions.clone();
            let c = game.state.current_color();
            if actions.len() == 1 {
                game.execute(actions[0]).expect("legal");
                continue;
            }
            // The deep search: what the network thinks (prior) vs what search finds.
            let mut m = MctsPlayer::new(c, net.clone(), deep_cfg.clone(), seed * 1000 + k);
            k += 1;
            let visits = m.search(&game).expect("search");
            let (stats, _) = m.root_stats();
            let best = (0..visits.len()).max_by_key(|&j| visits[j]).unwrap();
            let top_prior = (0..stats.len()).max_by(|&a, &b| stats[a].0.total_cmp(&stats[b].0)).unwrap();
            let q_best = stats[best].2.unwrap_or(0.0);
            let q_net = stats[top_prior].2.unwrap_or_else(|| stats.iter().filter_map(|s| s.2).fold(f32::INFINITY, f32::min));
            let regret = (q_best - q_net).max(0.0);
            // Normalized entropy of the network's policy: the cheap uncertainty signal.
            let prior_sum: f32 = stats.iter().map(|s| s.0).sum();
            let entropy = -stats.iter().map(|s| s.0 / prior_sum).filter(|&p| p > 0.0).map(|p| p * p.ln()).sum::<f32>()
                / (actions.len() as f32).ln().max(1e-6);
            let s = &game.state;
            let opp = s.colors.iter().find(|&&x| x != c).unwrap();
            let (mv, ov) = (s.player(c).victory_points, s.player(*opp).victory_points);
            let phase = if s.is_initial_build_phase { "opening" } else if mv.max(ov) < 6 { "early" } else if mv.max(ov) < 11 { "mid" } else { "late" };
            out.push_str(&format!(
                "{g}\t{}\t{mv}\t{ov}\t{phase}\t{}\t{}\t{}\t{}\t{regret:.4}\t{:.3}\t{entropy:.3}\t{:?}\n",
                s.num_turns, kind(&actions[best]), kind(&actions[top_prior]), actions.len(), (top_prior == best) as u8, stats[best].0, s.current_prompt
            ));
            // Play on with the live-strength search (a fresh, cheaper searcher).
            let mut p = MctsPlayer::new(c, net.clone(), live.clone(), seed * 1000 + 500 + k);
            let pv = p.search(&game).expect("search");
            let play = (0..pv.len()).max_by_key(|&j| pv[j]).unwrap();
            game.execute(actions[play]).expect("legal");
        }
        out
    });
    for r in rows {
        print!("{r}");
    }
}
