//! How a network uses ports: bank trades by rate, and buildings on port nodes.
//! Plays OpenHex 1v1 games, the network (no search) in both seats.
//!   cargo run --release --example port_usage -- <model.ctnn> [games]
use catanatron::map::{MapType, NumberPlacement};
use catanatron::players::{from_key, Player};
use catanatron::state::DiceMode;
use catanatron::*;

fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let games: u64 = std::env::args().nth(2).map_or(1000, |g| g.parse().expect("games"));
    let key = format!("NN:{path}");
    // [rate 2, 3, 4] trades; settlements on a port node (setup / later, 3:1 / 2:1); cities there.
    let mut trades = [0u64; 3];
    let (mut setup_port, mut setup_all, mut later_port, mut later_all) = (0u64, 0u64, 0u64, 0u64);
    let (mut generic, mut specific) = (0u64, 0u64);
    let (mut winner_ports, mut loser_ports, mut decided) = (0u64, 0u64, 0u64);
    let (mut turns, mut trading_games) = (0u64, 0u64);
    // Opening production (pips by resource over both opening settlements), per player.
    let mut open_pips = [0f64; 5];
    let (mut openings, mut no_wood_brick, mut no_ore_wheat) = (0u64, 0u64, 0u64);
    // Trades at rate 2 that gave the port's own resource (sanity: should be all of them).
    for seed in 0..games {
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
        let mut bots: Vec<Box<dyn Player>> =
            COLORS[..2].iter().map(|&c| from_key(&key, c, seed).expect("NN player")).collect();
        let port_of = |g: &Game, n: NodeId| g.state.board.map.port_nodes.iter().find(|(_, ns)| ns.contains(&n)).map(|(r, _)| *r);
        let mut owned_ports = [0u64; 2];
        let mut opening_res = [[0f64; 5]; 2];
        let mut traded = false;
        while !g.is_over() {
            let setup = g.state.is_initial_build_phase;
            let rec = g.play_tick(&mut bots).expect("legal play");
            let who = rec.action.color.idx();
            match rec.action.kind {
                ActionKind::MaritimeTrade { rate, .. } => {
                    trades[(rate - 2) as usize] += 1;
                    traded = true;
                }
                ActionKind::BuildSettlement(n) => {
                    let port = port_of(&g, n);
                    if setup {
                        setup_all += 1;
                        setup_port += port.is_some() as u64;
                        for &ti in &g.state.board.map.adjacent_tiles[n as usize] {
                            let t = &g.state.board.map.tiles[ti];
                            if let (Some(r), Some(num)) = (t.resource(), t.number()) {
                                opening_res[who][r.idx()] += 36.0 * catanatron::map::number_probability(num);
                            }
                        }
                    } else {
                        later_all += 1;
                        later_port += port.is_some() as u64;
                    }
                    match port {
                        Some(None) => generic += 1,
                        Some(Some(_)) => specific += 1,
                        None => {}
                    }
                    owned_ports[who] += port.is_some() as u64;
                }
                _ => {}
            }
        }
        turns += g.state.num_turns as u64;
        for res in &opening_res {
            for k in 0..5 {
                open_pips[k] += res[k];
            }
            openings += 1;
            no_wood_brick += (res[0] == 0.0 || res[1] == 0.0) as u64;
            no_ore_wheat += (res[3] == 0.0 || res[4] == 0.0) as u64;
        }
        trading_games += traded as u64;
        if let Some(w) = g.winning_color() {
            decided += 1;
            winner_ports += owned_ports[w.idx()];
            loser_ports += owned_ports[1 - w.idx()];
        }
    }
    let n = games as f64;
    let pct = |a: u64, b: u64| 100.0 * a as f64 / b.max(1) as f64;
    println!("{games} games, {:.0} turns/game; games with any bank trade: {:.0}%", turns as f64 / n, pct(trading_games, games));
    println!("bank trades per game: 2:1 {:.2}  3:1 {:.2}  4:1 {:.2}  (port-rate share {:.0}%)",
             trades[0] as f64 / n, trades[1] as f64 / n, trades[2] as f64 / n, pct(trades[0] + trades[1], trades.iter().sum()));
    println!("settlements on a port: setup {:.0}% ({setup_port}/{setup_all}), later {:.0}% ({later_port}/{later_all}); 3:1 {generic}, 2:1 {specific}",
             pct(setup_port, setup_all), pct(later_port, later_all));
    let o = openings.max(1) as f64;
    println!("opening pips per player: wood {:.1} brick {:.1} sheep {:.1} wheat {:.1} ore {:.1} (total {:.1}); missing wood or brick {:.0}%, missing wheat or ore {:.0}%",
             open_pips[0] / o, open_pips[1] / o, open_pips[2] / o, open_pips[3] / o, open_pips[4] / o, open_pips.iter().sum::<f64>() / o,
             pct(no_wood_brick, openings), pct(no_ore_wheat, openings));
    println!("ports per player per game: {:.2}; winner {:.2} vs loser {:.2}",
             (winner_ports + loser_ports) as f64 / (2.0 * decided.max(1) as f64),
             winner_ports as f64 / decided.max(1) as f64, loser_ports as f64 / decided.max(1) as f64);
}
