//! Two networks head to head (network only, seats alternating), OpenHex 1v1
//! rules: how each one plays, from the same games.
//!   cargo run --release --example style_vs -- <a.ctnn> <b.ctnn> <board> [games]
//! board: `standard` (a new random board each game), `seed:<n>` (the board a
//! game with seed n draws, as examples/board_screen numbers them), or
//! `spec:<file>` (a file holding an OpenHex board in catanatron_env.board_spec
//! form).
use std::collections::HashMap;
use std::sync::Arc;

use catanatron::map::{build_map, CatanMap, MapType, NumberPlacement};
use catanatron::openhex::{turn_hex, turn_vertex, Geometry};
use catanatron::players::{from_key, Player};
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::*;

fn pips(n: Option<u8>) -> u32 {
    n.map_or(0, |n| 6 - (7 - n as i32).unsigned_abs())
}

fn resource(name: &str) -> Option<Resource> {
    match name {
        "wood" | "forest" => Some(Resource::Wood),
        "brick" | "hills" => Some(Resource::Brick),
        "sheep" | "pasture" => Some(Resource::Sheep),
        "wheat" | "fields" => Some(Resource::Wheat),
        "ore" | "mountains" => Some(Resource::Ore),
        "desert" | "-" => None,
        other => panic!("unknown resource or terrain {other}"),
    }
}

/// An OpenHex board from its compact text form (as `mirror::openhex_map_spec`).
fn spec_map(spec: &str) -> Arc<CatanMap> {
    let base = build_map(MapType::Base, NumberPlacement::OfficialSpiral, &mut Rng::new(0));
    let geo = Geometry::new(&base).expect("geometry");
    let (tiles, ports) = spec.trim().split_once('|').expect("tiles|ports");
    let int = |x: &str| x.parse::<i8>().expect("coordinate");
    let land: Vec<_> = tiles
        .split(';')
        .map(|t| {
            let f: Vec<&str> = t.split(',').collect();
            let number = if f[4] == "-" { None } else { Some(f[4].parse::<u8>().expect("number")) };
            (turn_hex((int(f[0]), int(f[1]), int(f[2]))), resource(f[3]), number)
        })
        .collect();
    let port_list: Vec<_> = ports
        .split(';')
        .map(|p| {
            let f: Vec<&str> = p.split(',').collect();
            let a = geo.node(turn_vertex((int(f[0]), int(f[1]), int(f[2])))).expect("port vertex");
            let b = geo.node(turn_vertex((int(f[3]), int(f[4]), int(f[5])))).expect("port vertex");
            ([a, b], resource(f[6]))
        })
        .collect();
    Arc::new(base.with_layout(&land, &port_list).expect("board"))
}

#[derive(Default)]
struct Stats {
    games: u64,
    wins: u64,
    settle: u64,
    city: u64,
    road: u64,
    dev: u64,
    knight: u64,
    port_settle: u64,
    trades: [u64; 3],
    open_pips: [u64; 5],
    open_port: u64,
    longest_road: u64,
    largest_army: u64,
    vp: u64,
    spots: HashMap<String, u64>,
}

fn spot(m: &CatanMap, n: NodeId) -> String {
    let mut parts: Vec<String> = m.adjacent_tiles[n as usize]
        .iter()
        .filter_map(|&ti| m.tiles[ti].resource().map(|r| format!("{}{}", &format!("{r:?}")[..2], pips(m.tiles[ti].number()))))
        .collect();
    parts.sort();
    let port = m.port_nodes.iter().find(|(_, ns)| ns.contains(&n)).map(|(r, _)| match r {
        Some(r) => format!(" [2:1 {r:?}]"),
        None => " [3:1]".to_string(),
    });
    format!("#{n} {}{}", parts.join("+"), port.unwrap_or_default())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let paths = [args.get(1).expect("model a"), args.get(2).expect("model b")];
    let board = args.get(3).expect("board");
    let games: u64 = args.get(4).map_or(1000, |g| g.parse().expect("games"));
    let fixed: Option<Arc<CatanMap>> = if board == "standard" {
        None
    } else if let Some(n) = board.strip_prefix("seed:") {
        Some(build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(n.parse().expect("seed"))))
    } else if let Some(f) = board.strip_prefix("spec:") {
        Some(spec_map(&std::fs::read_to_string(f).expect("spec file")))
    } else {
        panic!("board must be standard, seed:<n> or spec:<file>");
    };
    let mut stats = [Stats::default(), Stats::default()];
    let mut turns = 0u64;
    for seed in 0..games {
        let config = GameConfig {
            map_type: MapType::Base,
            vps_to_win: 15,
            discard_limit: 9,
            friendly_robber: FriendlyRobber::PublicPoints,
            dice: DiceMode::OpenHex,
            number_placement: NumberPlacement::SpiralRandomCorner,
            catan_map: fixed.clone(),
            seed: Some(seed),
            record_history: false,
            ..Default::default()
        };
        let mut g = Game::new(&COLORS[..2], &config);
        // Model of each color; alternated by seed (the game shuffles seats too).
        let model_of = |c: Color| (c.idx() + seed as usize) % 2;
        let mut bots: Vec<Box<dyn Player>> = COLORS[..2]
            .iter()
            .map(|&c| from_key(&format!("NN:{}", paths[model_of(c)]), c, seed).expect("NN player"))
            .collect();
        let map = g.state.board.map.clone();
        while !g.is_over() {
            let setup = g.state.is_initial_build_phase;
            let rec = g.play_tick(&mut bots).expect("legal play");
            let s = &mut stats[model_of(rec.action.color)];
            match rec.action.kind {
                ActionKind::BuildSettlement(n) => {
                    let on_port = map.port_nodes.iter().any(|(_, ns)| ns.contains(&n));
                    if setup {
                        *s.spots.entry(spot(&map, n)).or_default() += 1;
                        s.open_port += on_port as u64;
                        for &ti in &map.adjacent_tiles[n as usize] {
                            if let Some(r) = map.tiles[ti].resource() {
                                s.open_pips[r.idx()] += pips(map.tiles[ti].number()) as u64;
                            }
                        }
                    } else {
                        s.settle += 1;
                        s.port_settle += on_port as u64;
                    }
                }
                ActionKind::BuildCity(_) => s.city += 1,
                ActionKind::BuildRoad(_) if !setup => s.road += 1,
                ActionKind::BuyDevelopmentCard => s.dev += 1,
                ActionKind::PlayKnightCard => s.knight += 1,
                ActionKind::MaritimeTrade { rate, .. } => s.trades[(rate - 2) as usize] += 1,
                _ => {}
            }
        }
        turns += g.state.num_turns as u64;
        let winner = g.winning_color();
        let lr = g.state.longest_road_color();
        let la = g.state.largest_army().map(|(c, _)| c);
        for &c in &COLORS[..2] {
            let s = &mut stats[model_of(c)];
            s.games += 1;
            s.wins += (winner == Some(c)) as u64;
            s.longest_road += (lr == Some(c)) as u64;
            s.largest_army += (la == Some(c)) as u64;
            s.vp += g.state.player(c).actual_victory_points as u64;
        }
    }
    println!("{games} games on {board}, {:.1} turns per game", turns as f64 / games as f64);
    for (i, s) in stats.iter().enumerate() {
        let n = s.games as f64;
        let name = std::path::Path::new(paths[i]).file_name().map_or(String::new(), |f| f.to_string_lossy().into_owned());
        println!("\n{name}: wins {:.1}%, final VP {:.1}", 100.0 * s.wins as f64 / n, s.vp as f64 / n);
        println!(
            "  per game: settlements {:.2} (on a port {:.2}), cities {:.2}, roads {:.2}, dev cards {:.2}, knights {:.2}",
            s.settle as f64 / n, s.port_settle as f64 / n, s.city as f64 / n, s.road as f64 / n, s.dev as f64 / n, s.knight as f64 / n
        );
        println!(
            "  trades 2:1 {:.2} 3:1 {:.2} 4:1 {:.2}; longest road {:.0}%, largest army {:.0}%",
            s.trades[0] as f64 / n, s.trades[1] as f64 / n, s.trades[2] as f64 / n,
            100.0 * s.longest_road as f64 / n, 100.0 * s.largest_army as f64 / n
        );
        println!(
            "  opening pips: wood {:.1} brick {:.1} sheep {:.1} wheat {:.1} ore {:.1}; opening on a port {:.0}%",
            s.open_pips[0] as f64 / n, s.open_pips[1] as f64 / n, s.open_pips[2] as f64 / n, s.open_pips[3] as f64 / n,
            s.open_pips[4] as f64 / n, 100.0 * s.open_port as f64 / (2.0 * n)
        );
        if fixed.is_some() {
            let mut spots: Vec<_> = s.spots.iter().collect();
            spots.sort_by(|a, b| b.1.cmp(a.1));
            let line: Vec<String> = spots.iter().take(5).map(|(k, v)| format!("{k} {:.0}%", 100.0 * **v as f64 / (2.0 * n))).collect();
            println!("  opening spots (share of its settlements): {}", line.join(", "));
        }
    }
}
