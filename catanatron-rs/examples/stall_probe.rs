//! Replays a stalled MCTS-vs-random game (see stall_diagnosis) to a turn,
//! then asks searches of several sizes what to do at our next main-phase
//! decision after rolling, and follows each search's line for the rest of
//! that turn.
//!   cargo run --release --example stall_probe -- <model.ctnn> <seed> <turn>
use catanatron::map::{MapType, NumberPlacement};
use catanatron::mcts::{load_net, MctsConfig, MctsPlayer, Utility};
use catanatron::players::{from_key, Player};
use catanatron::state::DiceMode;
use catanatron::*;

fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let seed: u64 = std::env::args().nth(2).expect("seed").parse().unwrap();
    let turn: u32 = std::env::args().nth(3).expect("turn").parse().unwrap();
    let key = format!("MCTS:100,c1,f0.25,m0.1,s0.1/400:{path}");
    let config = GameConfig {
        map_type: MapType::Base, vps_to_win: 15, discard_limit: 9,
        friendly_robber: FriendlyRobber::PublicPoints, dice: DiceMode::OpenHex,
        number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(seed), ..Default::default()
    };
    let mut g = Game::new(&COLORS[..2], &config);
    let us = COLORS[0];
    let mut bots: Vec<Box<dyn Player>> = vec![from_key(&key, COLORS[0], seed).unwrap(), from_key("R", COLORS[1], seed).unwrap()];
    // Replay to the turn, then to our first post-roll decision with real options.
    while !g.is_over() && !(g.state.num_turns >= turn && g.state.current_color() == us
        && g.state.player(us).has_rolled && g.state.current_prompt == ActionPrompt::PlayTurn && g.playable_actions.len() > 1) {
        g.play_tick(&mut bots).unwrap();
    }
    let p = g.state.player(us);
    println!("seed {seed}, turn {}: VP {} hand {:?} bank {:?}, {} legal actions", g.state.num_turns, p.actual_victory_points, p.resources, g.state.bank, g.playable_actions.len());
    let kinds: Vec<String> = g.playable_actions.iter().map(|a| format!("{:?}", a.kind).split(['(', ' ', '{']).next().unwrap().to_string()).collect();
    println!("  legal: {:?}", kinds.iter().fold(std::collections::BTreeMap::new(), |mut m, k| { *m.entry(k.clone()).or_insert(0) += 1; m }));
    let speed = 0.1f32 * (1.0 - g.state.num_turns as f32 / 400.0).max(0.0);
    println!("  speed bonus for winning now: {speed:.3} (zero once past turn 400)");
    let net = load_net(&path).unwrap();
    {
        let cfg = MctsConfig { simulations: 1600, c_puct: 1.0, fpu_reduction: 0.25,
            utility: Utility { margin: 0.1, speed: 0.1, speed_turns: 400.0 }, ..Default::default() };
        let mut m = MctsPlayer::new(us, net.clone(), cfg, seed);
        m.search(&g).unwrap();
        let (stats, root) = m.root_stats();
        println!("  1600-sim root value {root:.3} (bound {:.2}); per action (prior, visits, q):", cfg.utility.bound());
        for (a, (p, n, q)) in g.playable_actions.iter().zip(stats) {
            println!("    {:<60} prior {p:.3} visits {n:5} q {}", format!("{:?}", a.kind), q.map_or("-".into(), |q| format!("{q:.3}")));
        }
    }
    for sims in [100u32, 400, 1600, 6400] {
        let cfg = MctsConfig { simulations: sims, c_puct: 1.0, fpu_reduction: 0.25,
            utility: Utility { margin: 0.1, speed: 0.1, speed_turns: 400.0 }, ..Default::default() };
        // Follow this search's choices until it ends the turn (or 12 actions).
        let mut h = g.copy_without_history();
        let mut line = Vec::new();
        for _ in 0..12 {
            if h.is_over() || h.state.current_color() != us { break; }
            let mut m = MctsPlayer::new(us, net.clone(), cfg, seed);
            let actions = h.playable_actions.clone();
            let a = m.decide(&h, &actions);
            line.push(format!("{:?}", a.kind));
            if matches!(a.kind, ActionKind::EndTurn) { break; }
            h.execute(a).unwrap();
        }
        let q = h.state.player(us);
        println!("  {sims:>5} sims: {} | after: VP {} hand {:?}", line.join(" -> "), q.actual_victory_points, q.resources);
    }
}
