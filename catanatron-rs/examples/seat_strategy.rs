//! How a network plays by seat (first vs second player): openings,
//! building, dev cards, ports, win rate. OpenHex 1v1 games, the network (no
//! search) in both seats.
//!   cargo run --release --example seat_strategy -- <model.ctnn> [games]
use catanatron::map::{MapType, NumberPlacement};
use catanatron::players::{from_key, Player};
use catanatron::state::DiceMode;
use catanatron::*;

#[derive(Default)]
struct Seat {
    wins: u64,
    pips: [f64; 5],
    no_wood_brick: u64,
    no_wheat_ore: u64,
    port_opening: u64,
    settlements: u64,
    cities: u64,
    roads: u64,
    devs: u64,
    trades: [u64; 3],
    first_settle_pips: f64,
    second_settle_pips: f64,
    longest_road: u64,
    largest_army: u64,
    final_vp: u64,
}

fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let games: u64 = std::env::args().nth(2).map_or(1000, |g| g.parse().expect("games"));
    let key = format!("NN:{path}");
    let mut seats = [Seat::default(), Seat::default()];
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
            ..Default::default()
        };
        let mut g = Game::new(&COLORS[..2], &config);
        let mut bots: Vec<Box<dyn Player>> =
            COLORS[..2].iter().map(|&c| from_key(&key, c, seed).expect("NN player")).collect();
        let first = g.state.colors[0];
        let seat_of = |c: Color| (c != first) as usize;
        let mut opening = [[0f64; 5]; 2];
        let mut opening_count = [0usize; 2];
        while !g.is_over() {
            let setup = g.state.is_initial_build_phase;
            let rec = g.play_tick(&mut bots).expect("legal play");
            let k = seat_of(rec.action.color);
            let st = &mut seats[k];
            match rec.action.kind {
                ActionKind::BuildSettlement(n) if setup => {
                    let map = &g.state.board.map;
                    let mut pips = 0.0;
                    for &ti in &map.adjacent_tiles[n as usize] {
                        let t = &map.tiles[ti];
                        if let (Some(r), Some(num)) = (t.resource(), t.number()) {
                            let p = 36.0 * catanatron::map::number_probability(num);
                            opening[k][r.idx()] += p;
                            pips += p;
                        }
                    }
                    if opening_count[k] == 0 { st.first_settle_pips += pips } else { st.second_settle_pips += pips }
                    opening_count[k] += 1;
                    st.port_opening += map.port_nodes.iter().any(|(_, ns)| ns.contains(&n)) as u64;
                }
                ActionKind::BuildSettlement(_) => st.settlements += 1,
                ActionKind::BuildCity(_) => st.cities += 1,
                ActionKind::BuildRoad(_) if !setup => st.roads += 1,
                ActionKind::BuyDevelopmentCard => st.devs += 1,
                ActionKind::MaritimeTrade { rate, .. } => st.trades[(rate - 2) as usize] += 1,
                _ => {}
            }
        }
        turns += g.state.num_turns as u64;
        for (k, res) in opening.iter().enumerate() {
            let st = &mut seats[k];
            for r in 0..5 {
                st.pips[r] += res[r];
            }
            st.no_wood_brick += (res[0] == 0.0 || res[1] == 0.0) as u64;
            st.no_wheat_ore += (res[3] == 0.0 || res[4] == 0.0) as u64;
        }
        for p in &g.state.players {
            let st = &mut seats[seat_of(p.color)];
            st.longest_road += p.has_road as u64;
            st.largest_army += p.has_army as u64;
            st.final_vp += p.actual_victory_points as u64;
        }
        if let Some(w) = g.winning_color() {
            seats[seat_of(w)].wins += 1;
        }
    }
    let n = games as f64;
    let pct = |a: u64| 100.0 * a as f64 / n;
    println!("{games} games, {:.0} turns/game", turns as f64 / n);
    println!("{:<34} {:>10} {:>10}", "", "first", "second");
    let row = |label: &str, a: String, b: String| println!("{label:<34} {a:>10} {b:>10}");
    let [f, s] = &seats;
    row("win rate", format!("{:.1}%", pct(f.wins)), format!("{:.1}%", pct(s.wins)));
    row("1st opening settlement pips", format!("{:.1}", f.first_settle_pips / n), format!("{:.1}", s.first_settle_pips / n));
    row("2nd opening settlement pips", format!("{:.1}", f.second_settle_pips / n), format!("{:.1}", s.second_settle_pips / n));
    for (r, name) in ["wood", "brick", "sheep", "wheat", "ore"].iter().enumerate() {
        row(&format!("opening {name} pips"), format!("{:.1}", f.pips[r] / n), format!("{:.1}", s.pips[r] / n));
    }
    row("opening missing wood or brick", format!("{:.0}%", pct(f.no_wood_brick)), format!("{:.0}%", pct(s.no_wood_brick)));
    row("opening missing wheat or ore", format!("{:.0}%", pct(f.no_wheat_ore)), format!("{:.0}%", pct(s.no_wheat_ore)));
    row("port settlements in opening", format!("{:.2}", f.port_opening as f64 / n), format!("{:.2}", s.port_opening as f64 / n));
    row("settlements built later", format!("{:.2}", f.settlements as f64 / n), format!("{:.2}", s.settlements as f64 / n));
    row("cities built", format!("{:.2}", f.cities as f64 / n), format!("{:.2}", s.cities as f64 / n));
    row("roads built later", format!("{:.1}", f.roads as f64 / n), format!("{:.1}", s.roads as f64 / n));
    row("dev cards bought", format!("{:.1}", f.devs as f64 / n), format!("{:.1}", s.devs as f64 / n));
    row("bank trades 4:1 / 3:1 / 2:1", format!("{:.1}/{:.1}/{:.1}", f.trades[2] as f64 / n, f.trades[1] as f64 / n, f.trades[0] as f64 / n),
        format!("{:.1}/{:.1}/{:.1}", s.trades[2] as f64 / n, s.trades[1] as f64 / n, s.trades[0] as f64 / n));
    row("ends with longest road", format!("{:.0}%", pct(f.longest_road)), format!("{:.0}%", pct(s.longest_road)));
    row("ends with largest army", format!("{:.0}%", pct(f.largest_army)), format!("{:.0}%", pct(s.largest_army)));
    row("final points (incl. hidden)", format!("{:.1}", f.final_vp as f64 / n), format!("{:.1}", s.final_vp as f64 / n));
}
