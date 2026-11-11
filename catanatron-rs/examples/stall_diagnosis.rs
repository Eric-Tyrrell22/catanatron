//! Why does a searching bot fail to finish against a random one? Plays
//! MCTS(<model>) vs R under OpenHex rules until a game reaches the turn cap,
//! then prints the final position and the searcher's recent actions.
//!   cargo run --release --example stall_diagnosis -- <model.ctnn> [sims] [max games]
use std::collections::BTreeMap;

use catanatron::map::{MapType, NumberPlacement};
use catanatron::players::{from_key, Player};
use catanatron::state::DiceMode;
use catanatron::*;

fn kind_name(k: &ActionKind) -> String {
    let s = format!("{k:?}");
    s.split(['(', ' ', '{']).next().unwrap_or("").to_string()
}

fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let sims: u32 = std::env::args().nth(2).map_or(100, |s| s.parse().unwrap());
    let max_games: u64 = std::env::args().nth(3).map_or(60, |s| s.parse().unwrap());
    let key = format!("MCTS:{sims},c1,f0.25,m0.1,s0.1/400:{path}");
    let mut stalls = 0;
    for seed in 0..max_games {
        let config = GameConfig {
            map_type: MapType::Base,
            vps_to_win: 15,
            discard_limit: 9,
            friendly_robber: FriendlyRobber::PublicPoints,
            dice: DiceMode::OpenHex,
            number_placement: NumberPlacement::SpiralRandomCorner,
            seed: Some(seed),
            ..Default::default()
        };
        let mut g = Game::new(&COLORS[..2], &config);
        let us = COLORS[0];
        let mut bots: Vec<Box<dyn Player>> =
            vec![from_key(&key, COLORS[0], seed).unwrap(), from_key("R", COLORS[1], seed).unwrap()];
        // Our actions per turn window, and VP over time.
        let mut recent: Vec<(u32, String)> = Vec::new();
        let mut vp_at = BTreeMap::new();
        while !g.is_over() {
            let rec = g.play_tick(&mut bots).expect("legal play");
            if rec.action.color == us {
                recent.push((g.state.num_turns, kind_name(&rec.action.kind)));
            }
            vp_at.insert(g.state.num_turns / 100, (g.state.player(us).actual_victory_points, g.state.player(COLORS[1]).actual_victory_points));
        }
        if g.winning_color().is_some() {
            continue;
        }
        stalls += 1;
        let s = &g.state;
        println!("=== seed {seed}: no winner after {} turns", s.num_turns);
        println!("points by turn (us, random): {:?}", vp_at.iter().map(|(k, v)| (k * 100, *v)).collect::<Vec<_>>());
        for p in &s.players {
            println!(
                "  {:?}{}: VP {} (public {}), hand {:?}, devs {:?}, pieces left: settlements {} cities {} roads {}, longest road {}",
                p.color, if p.color == us { " (search)" } else { " (random)" },
                p.actual_victory_points, p.victory_points, p.resources, p.dev_cards,
                p.settlements_available, p.cities_available, p.roads_available, p.longest_road_length
            );
        }
        println!("  bank {:?}, dev deck {} left, robber on {:?}", s.bank, s.dev_deck.len(), s.board.robber_coordinate);
        let legal_settle = s.board.buildable_node_ids(us, false).len();
        let edges = s.board.buildable_edges(us).len();
        println!("  our legal settlement spots now: {legal_settle}, buildable road edges: {edges}");
        let last: Vec<&(u32, String)> = recent.iter().filter(|(t, _)| *t + 100 >= s.num_turns).collect();
        let mut hist: BTreeMap<&str, usize> = BTreeMap::new();
        for (_, k) in &last {
            *hist.entry(k.as_str()).or_default() += 1;
        }
        println!("  our actions in the last 100 turns: {hist:?}");
        if stalls >= 3 {
            break;
        }
    }
}
