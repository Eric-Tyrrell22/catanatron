//! Are search's opening settlement picks stable? For opening settlement
//! positions from real games, compares the network's top pick with search
//! picks: two independent searches at N sims, and one at 4N.
//!   cargo run --release --example opening_consistency -- <model.ctnn> [games] [sims]
use catanatron::map::{MapType, NumberPlacement};
use catanatron::mcts::{load_net, MctsConfig, MctsPlayer, Utility};
use catanatron::players::{from_key, Player};
use catanatron::state::DiceMode;
use catanatron::*;

fn top(v: &[f32]) -> usize {
    (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap()
}

fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let games: u64 = std::env::args().nth(2).map_or(12, |g| g.parse().expect("games"));
    let sims: u32 = std::env::args().nth(3).map_or(2400, |s| s.parse().expect("sims"));
    let net = load_net(&path).unwrap();
    let config = |n: u32| MctsConfig {
        simulations: n,
        placement_sims: n,
        c_puct: 1.0,
        fpu_reduction: 0.25,
        root_noise: 0.0,
        utility: Utility { margin: 0.1, speed: 0.1, speed_turns: 400.0 },
        ..Default::default()
    };
    let key = format!("NN:{path}");
    // Counters: [a==b, a==big, b==big, net==a, net==big]; top visit shares.
    let mut agree = [0u32; 5];
    let (mut n, mut share_a, mut share_big) = (0u32, 0.0f64, 0.0f64);
    for seed in 0..games {
        let gc = GameConfig {
            map_type: MapType::Base,
            vps_to_win: 15,
            discard_limit: 9,
            friendly_robber: FriendlyRobber::PublicPoints,
            dice: DiceMode::OpenHex,
            number_placement: NumberPlacement::SpiralRandomCorner,
            seed: Some(1000 + seed),
            ..Default::default()
        };
        let mut g = Game::new(&COLORS[..2], &gc);
        let mut bots: Vec<Box<dyn Player>> =
            COLORS[..2].iter().map(|&c| from_key(&key, c, seed).expect("NN player")).collect();
        while g.state.is_initial_build_phase {
            let opening = g.playable_actions.len() > 1
                && g.playable_actions.iter().all(|a| matches!(a.kind, ActionKind::BuildSettlement(_)));
            if opening {
                let color = g.state.current_color();
                let mut picks = Vec::new();
                let mut prior = Vec::new();
                for (k, sims_k) in [(1u64, sims), (2, sims), (3, sims * 4)] {
                    let mut m = MctsPlayer::new(color, net.clone(), config(sims_k), seed * 10 + k);
                    let visits = m.search(&g).unwrap();
                    let total: u32 = visits.iter().sum();
                    let v: Vec<f32> = visits.iter().map(|&x| x as f32).collect();
                    let t = top(&v);
                    if k == 1 {
                        share_a += visits[t] as f64 / total as f64;
                        prior = m.root_stats().0.iter().map(|s| s.0).collect();
                    }
                    if k == 3 {
                        share_big += visits[t] as f64 / total as f64;
                    }
                    picks.push(t);
                }
                let net_pick = top(&prior);
                let (a, b, big) = (picks[0], picks[1], picks[2]);
                for (i, same) in [a == b, a == big, b == big, net_pick == a, net_pick == big].into_iter().enumerate() {
                    agree[i] += same as u32;
                }
                n += 1;
            }
            g.play_tick(&mut bots).expect("legal play");
        }
    }
    let pct = |x: u32| 100.0 * x as f64 / n as f64;
    println!("{n} opening settlement positions, {sims} sims (and {} for the big search)", sims * 4);
    println!("two {sims}-sim searches pick the same spot:        {:.0}%", pct(agree[0]));
    println!("{sims}-sim search agrees with the {}-sim search:   {:.0}% / {:.0}%", sims * 4, pct(agree[1]), pct(agree[2]));
    println!("network's top pick = {sims}-sim search pick:        {:.0}%", pct(agree[3]));
    println!("network's top pick = {}-sim search pick:          {:.0}%", sims * 4, pct(agree[4]));
    println!("visit share of the top spot: {sims} sims {:.0}%, {} sims {:.0}%", 100.0 * share_a / n as f64, sims * 4, 100.0 * share_big / n as f64);
}
