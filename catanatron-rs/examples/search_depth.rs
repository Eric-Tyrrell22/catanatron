//! How far ahead search looks, for a given network and simulation counts.
//!   cargo run --release --example search_depth -- <model.ctnn>
use catanatron::mcts::{load_net, MctsConfig, MctsPlayer};
use catanatron::players::{from_key, Player};
use catanatron::*;

fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let net = load_net(&path).unwrap();
    println!("{:>6} {:>12} {:>12} {:>12} {:>12} {:>8} {:>10}", "sims", "max moves", "mean moves", "max turns", "mean turns", "nodes", "ms/search");
    for sims in [100u32, 400, 1600] {
        let mut m = MctsPlayer::new(Color::Red, net.clone(), MctsConfig { simulations: sims, ..Default::default() }, 1);
        let (mut n, mut acc, mut ms) = (0.0, [0.0f64; 5], 0.0);
        for seed in 0..6 {
            let mut g = Game::new(&COLORS[..2], &GameConfig { seed: Some(seed), ..Default::default() });
            let mut bots: Vec<Box<dyn Player>> = vec![from_key("F", Color::Red, seed).unwrap(), from_key("F", Color::Blue, seed).unwrap()];
            let mut k = 0;
            while !g.is_over() {
                // Sample every 5th multi-option decision of Red in normal play.
                if g.state.current_color() == Color::Red && g.playable_actions.len() > 1 && !g.state.is_initial_build_phase {
                    k += 1;
                    if k % 5 == 0 {
                        let t = std::time::Instant::now();
                        m.search(&g).unwrap();
                        ms += t.elapsed().as_secs_f64() * 1e3;
                        let (a, b, c, d, e) = m.tree_depth_stats();
                        for (x, v) in acc.iter_mut().zip([a as f64, b, c as f64, d, e as f64]) { *x += v; }
                        n += 1.0;
                    }
                }
                g.play_tick(&mut bots).unwrap();
            }
        }
        println!("{:>6} {:>12.1} {:>12.1} {:>12.1} {:>12.2} {:>8.0} {:>10.1}", sims, acc[0]/n, acc[1]/n, acc[2]/n, acc[3]/n, acc[4]/n, ms/n);
    }
}
