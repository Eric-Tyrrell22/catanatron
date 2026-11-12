//! Dead roads and monopoly timing in network-only games (both seats the same
//! key) on random Base boards. One line per fact, for runs/behaviour_probe.py:
//!
//!   roads <seat won 0/1> <roads built> <dead> <has longest road 0/1> <vp> <game turns>
//!     dead = roads at game end on no path between two of the owner's buildings
//!     (leaf roads pruned repeatedly: a road to nowhere, however long the chain).
//!   mono <hold turns> <haul> <best haul now> <max haul seen while held> <game turn> <own vp>
//!     for every Monopoly played: how long it was held (game turns, both seats),
//!     cards taken, the most any resource would have taken at that moment, and
//!     the most any resource would have taken at any of the holder's decisions
//!     since the card was bought.
//!   mono_unplayed <hold turns> <max haul seen> <won 0/1>
//!     for every Monopoly still in hand when the game ended.
//!
//!   cargo run --release --example behaviour_probe -- <net key> [games=400] [threads] [first board seed=6000000]
use catanatron::map::{build_map, MapType, NumberPlacement};
use catanatron::parallel::{default_threads, par_map};
use catanatron::players::from_key;
use catanatron::rng::Rng;
use catanatron::state::{DiceMode, State};
use catanatron::*;

/// Roads of `color` on no path between two of its buildings: leaves (an endpoint
/// with no own building and no other own road) are pruned until none remain.
fn dead_roads(game: &Game, color: Color) -> (usize, usize) {
    let map = &game.state.board.map;
    let b = &game.state.board;
    let mut edges: Vec<(usize, usize)> = map.edges.iter().enumerate()
        .filter(|&(e, _)| b.roads[e] == Some(color))
        .map(|(_, &(x, y))| (x as usize, y as usize)).collect();
    let total = edges.len();
    let owned = |nd: usize| b.buildings[nd].is_some_and(|(c, _)| c == color);
    loop {
        let before = edges.len();
        let degree = |nd: usize, edges: &[(usize, usize)]| edges.iter().filter(|&&(x, y)| x == nd || y == nd).count();
        let snapshot = edges.clone();
        edges.retain(|&(x, y)| !((!owned(x) && degree(x, &snapshot) == 1) || (!owned(y) && degree(y, &snapshot) == 1)));
        if edges.len() == before {
            break;
        }
    }
    (total, total - edges.len())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let key = args[1].clone();
    let games: u64 = args.get(2).map_or(400, |s| s.parse().expect("games"));
    let threads: usize = args.get(3).map_or(default_threads(), |s| s.parse().expect("threads"));
    let first: u64 = args.get(4).map_or(6_000_000, |s| s.parse().expect("seed"));
    let mono = DevCard::Monopoly.idx();
    let rows: Vec<Vec<String>> = par_map(games, threads, |g| {
        let seed = first + g;
        let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(seed));
        let rules = GameConfig {
            map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
            dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(seed),
            catan_map: Some(map.clone()), record_history: false, fixed_seats: true, ..Default::default()
        };
        let mut game = Game::new(&COLORS[..2], &rules);
        let mut a = from_key(&key, COLORS[0], seed).expect("A");
        let mut b = from_key(&key, COLORS[1], seed + 1).expect("B");
        let mut out = Vec::new();
        // Per seat: monopoly cards in hand as (turn bought, max haul seen since), oldest first.
        let mut held: [Vec<(u32, u32)>; 2] = [Vec::new(), Vec::new()];
        let opp_haul = |s: &State, me: Color| -> [u32; 5] {
            let mut h = [0u32; 5];
            for p in s.players.iter().filter(|p| p.color != me) {
                for r in 0..5 { h[r] += p.resources[r]; }
            }
            h
        };
        while !game.is_over() && game.state.num_turns < 400 {
            let actions = game.playable_actions.clone();
            let mover = game.state.current_color();
            let seat = (mover != COLORS[0]) as usize;
            let s = &game.state;
            if !held[seat].is_empty() {
                let m = *opp_haul(s, mover).iter().max().unwrap();
                for h in held[seat].iter_mut() { h.1 = h.1.max(m); }
            }
            let action = if actions.len() == 1 { actions[0] } else if mover == COLORS[0] { a.decide(&game, &actions) } else { b.decide(&game, &actions) };
            let before = s.player(mover).dev_cards[mono];
            if let ActionKind::PlayMonopoly(r) = action.kind {
                let hauls = opp_haul(s, mover);
                let (bought, seen) = held[seat].remove(0);
                out.push(format!("mono {} {} {} {} {} {}", s.num_turns - bought, hauls[r.idx()], hauls.iter().max().unwrap(), seen.max(hauls[r.idx()]), s.num_turns, s.player(mover).actual_victory_points));
            }
            let turn = s.num_turns;
            game.execute(action).expect("legal");
            if matches!(action.kind, ActionKind::BuyDevelopmentCard) && game.state.player(mover).dev_cards[mono] > before {
                held[seat].push((turn, 0));
            }
        }
        let winner = game.winning_color();
        for (seat, &c) in COLORS[..2].iter().enumerate() {
            let (total, dead) = dead_roads(&game, c);
            let p = game.state.player(c);
            out.push(format!("roads {} {total} {dead} {} {} {}", (winner == Some(c)) as u8, p.has_road as u8, p.actual_victory_points, game.state.num_turns));
            for &(bought, seen) in &held[seat] {
                out.push(format!("mono_unplayed {} {seen} {}", game.state.num_turns - bought, (winner == Some(c)) as u8));
            }
        }
        out
    });
    for lines in rows {
        for l in lines {
            println!("{l}");
        }
    }
}
