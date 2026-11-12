//! How often does a network settle on a port, and when? Plays network-only
//! games (both seats the same key) on random Base boards and prints one line
//! per settlement built: the game turn, the phase (open = initial placement,
//! mid = bought), the port on the node (none, 3:1, or the resource), and
//! whether that resource is the builder's most-produced one at the time.
//! Also prints the share of buildable nodes that carry a port, as the chance
//! baseline. Pipe into runs/port_picks.py for the terminal chart.
//!
//!   cargo run --release --example port_picks -- <net key> [games=400] [threads] [first board seed=5000000] [fixed]
//! With `fixed`, every game is played on the first board (different dice).
use catanatron::map::{build_map, MapType, NumberPlacement};
use catanatron::parallel::{default_threads, par_map};
use catanatron::players::from_key;
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let key = args[1].clone();
    let games: u64 = args.get(2).map_or(400, |s| s.parse().expect("games"));
    let threads: usize = args.get(3).map_or(default_threads(), |s| s.parse().expect("threads"));
    let first: u64 = args.get(4).map_or(5_000_000, |s| s.parse().expect("seed"));
    let fixed = args.get(5).is_some_and(|s| s == "fixed");
    let rows: Vec<(f64, Vec<String>)> = par_map(games, threads, |g| {
        let seed = first + g;
        let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(if fixed { first } else { seed }));
        let rules = GameConfig {
            map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
            dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(seed),
            catan_map: Some(map.clone()), record_history: false, fixed_seats: true, ..Default::default()
        };
        let n_land = map.land_nodes.iter().filter(|&&x| x).count();
        let port_of = |nd: usize| -> Option<Option<Resource>> {
            map.port_nodes.iter().find(|(_, nodes)| nodes.contains(&(nd as NodeId))).map(|(r, _)| *r)
        };
        let port_share = (0..n_land).filter(|&nd| port_of(nd).is_some()).count() as f64 / n_land as f64;
        let mut game = Game::new(&COLORS[..2], &rules);
        let mut a = from_key(&key, COLORS[0], seed).expect("A");
        let mut b = from_key(&key, COLORS[1], seed + 1).expect("B");
        let mut out = Vec::new();
        while !game.is_over() && game.state.num_turns < 400 {
            let actions = game.playable_actions.clone();
            let mover = game.state.current_color();
            let action = if actions.len() == 1 { actions[0] } else if mover == COLORS[0] { a.decide(&game, &actions) } else { b.decide(&game, &actions) };
            if let ActionKind::BuildSettlement(nd) = action.kind {
                let s = &game.state;
                // The builder's production before this settlement, in pips per resource.
                let mut prod = [0.0f64; 5];
                for &x in s.player(mover).settlements.iter() {
                    for (r, v) in map.node_production[x as usize].iter().enumerate() { prod[r] += v; }
                }
                for &x in s.player(mover).cities.iter() {
                    for (r, v) in map.node_production[x as usize].iter().enumerate() { prod[r] += 2.0 * v; }
                }
                let top = (0..5).max_by(|&x, &y| prod[x].total_cmp(&prod[y])).unwrap();
                let (port, matches) = match port_of(nd as usize) {
                    None => ("none".to_string(), 0),
                    Some(None) => ("3:1".to_string(), 0),
                    Some(Some(r)) => (format!("{r:?}"), (r.idx() == top && prod[top] > 0.0) as u8),
                };
                let phase = if s.is_initial_build_phase { "open" } else { "mid" };
                out.push(format!("{} {phase} {port} {matches} {:.1}", s.num_turns, prod[top] * 36.0));
            }
            game.execute(action).expect("legal");
        }
        (port_share, out)
    });
    let share: f64 = rows.iter().map(|r| r.0).sum::<f64>() / rows.len() as f64;
    println!("#port_share {share:.3}");
    for (_, lines) in rows {
        for l in lines {
            println!("{l}");
        }
    }
}
