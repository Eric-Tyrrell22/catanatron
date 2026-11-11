//! A network's play style on one kind of board: self-play (network only, both
//! seats), OpenHex 1v1 rules, per-player averages.
//!   cargo run --release --example style -- <model.ctnn> <standard|scarce|port> [games]
use catanatron::map::{BoardMix, MapType, NumberPlacement};
use catanatron::players::{from_key, Player};
use catanatron::state::DiceMode;
use catanatron::*;

fn pips(n: Option<u8>) -> u32 {
    n.map_or(0, |n| 6 - (7 - n as i32).unsigned_abs())
}

fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let board = std::env::args().nth(2).expect("board kind");
    let games: u64 = std::env::args().nth(3).map_or(1000, |g| g.parse().expect("games"));
    let board_mix = match board.as_str() {
        "standard" => BoardMix::default(),
        "scarce" => BoardMix { scarce_ows: 1.0, port_start: 0.0 },
        "port" => BoardMix { scarce_ows: 0.0, port_start: 1.0 },
        other => panic!("unknown board kind {other}"),
    };
    let key = format!("NN:{path}");
    // Per-player totals.
    let (mut settle, mut city, mut road, mut dev, mut port_settle, mut trades) = (0u64, 0u64, 0u64, 0u64, 0u64, [0u64; 3]);
    let mut open_pips = [0f64; 5];
    let (mut setup_port, mut setup_strong, mut strong_any) = (0u64, 0u64, 0u64);
    let mut turns = 0u64;
    for seed in 0..games {
        let config = GameConfig {
            map_type: MapType::Base,
            vps_to_win: 15,
            discard_limit: 9,
            friendly_robber: FriendlyRobber::PublicPoints,
            dice: DiceMode::OpenHex,
            number_placement: NumberPlacement::SpiralRandomCorner,
            seed: Some(seed),
            board_mix,
            ..Default::default()
        };
        let mut g = Game::new(&COLORS[..2], &config);
        let map = g.state.board.map.clone();
        // Strong port nodes: a 2:1 port node touching its own resource on a 6/8, >= 8 pips.
        let strong: Vec<NodeId> = map
            .port_nodes
            .iter()
            .filter_map(|(r, ns)| r.map(|r| (r, ns)))
            .flat_map(|(r, ns)| ns.iter().map(move |&n| (r, n)))
            .filter(|&(r, n)| {
                let adj = &map.adjacent_tiles[n as usize];
                adj.iter().any(|&i| map.tiles[i].resource() == Some(r) && pips(map.tiles[i].number()) == 5)
                    && adj.iter().map(|&i| pips(map.tiles[i].number())).sum::<u32>() >= 8
            })
            .map(|(_, n)| n)
            .collect();
        strong_any += !strong.is_empty() as u64;
        let mut bots: Vec<Box<dyn Player>> =
            COLORS[..2].iter().map(|&c| from_key(&key, c, seed).expect("NN player")).collect();
        while !g.is_over() {
            let setup = g.state.is_initial_build_phase;
            let rec = g.play_tick(&mut bots).expect("legal play");
            match rec.action.kind {
                ActionKind::MaritimeTrade { rate, .. } => trades[(rate - 2) as usize] += 1,
                ActionKind::BuildSettlement(n) => {
                    let on_port = map.port_nodes.iter().any(|(_, ns)| ns.contains(&n));
                    if setup {
                        setup_port += on_port as u64;
                        setup_strong += strong.contains(&n) as u64;
                        for &ti in &map.adjacent_tiles[n as usize] {
                            if let Some(r) = map.tiles[ti].resource() {
                                open_pips[r.idx()] += pips(map.tiles[ti].number()) as f64;
                            }
                        }
                    } else {
                        settle += 1;
                        port_settle += on_port as u64;
                    }
                }
                ActionKind::BuildCity(_) => city += 1,
                ActionKind::BuildRoad(_) if !setup => road += 1,
                ActionKind::BuyDevelopmentCard => dev += 1,
                _ => {}
            }
        }
        turns += g.state.num_turns as u64;
    }
    let p = 2.0 * games as f64; // player-games
    let n = games as f64;
    println!(
        "{board:8} turns {:5.1} | per player: settlements {:.2} (on port {:.2}) cities {:.2} roads {:.2} devs {:.2} | trades 2:1 {:.2} 3:1 {:.2} 4:1 {:.2} | opening on port {:.0}%, strong port taken in setup {:.0}% of games that have one | opening pips W{:.1} B{:.1} S{:.1} Wh{:.1} O{:.1}",
        turns as f64 / n, settle as f64 / p, port_settle as f64 / p, city as f64 / p, road as f64 / p, dev as f64 / p,
        trades[0] as f64 / p, trades[1] as f64 / p, trades[2] as f64 / p,
        100.0 * setup_port as f64 / (2.0 * p), 100.0 * setup_strong as f64 / strong_any.max(1) as f64,
        open_pips[0] / p, open_pips[1] / p, open_pips[2] / p, open_pips[3] / p, open_pips[4] / p
    );
}
