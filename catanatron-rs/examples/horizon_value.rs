//! How far ahead must one look to rank first picks? For a board, A's first
//! settlement is forced to each candidate node and the game is played on by
//! both sides' keys (network only, dice vary). Along each game the main
//! network's value for A is read at horizons: right after the pick (one ply),
//! at the end of the initial build phase, and when the turn counter first
//! reaches each listed turn; the final outcome closes the list. Means over the
//! games give, per node, the value a search with that horizon would see.
//!
//!   cargo run --release --example horizon_value -- <A key> <B key> <board seed> <node,...|top:K> [games] [threads] [turns=10,20,40,80]
//! Event horizons come after the turn horizons: A's first road, settlement, city and dev-card purchase after the opening.
//! Output: "board B node N: wr P% | ply1 v | open v | t10 v ... | road1 v | settle1 v | city1 v | end v"
use catanatron::encoding::ObservationSpace;
use catanatron::map::{build_map, MapType, NumberPlacement};
use catanatron::mcts::load_net;
use catanatron::nn::{PolicyValueNet, Scratch};
use catanatron::parallel::{default_threads, par_map};
use catanatron::players::from_key;
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::*;

/// The value of `game` for `me`, by the network that would evaluate it (the opening network in the initial build phase).
fn value_for(net: &PolicyValueNet, spaces: &(ObservationSpace, Option<ObservationSpace>), game: &Game, me: Color, scratch: &mut Scratch, logits: &mut Vec<f32>) -> f32 {
    let s = &game.state;
    let (n, space) = match (&net.opening, &spaces.1, s.is_initial_build_phase) {
        (Some(o), Some(os), true) => (&**o, os),
        _ => (net, &spaces.0),
    };
    let mut obs = vec![0.0f32; n.obs_size];
    space.encode_into(s, s.current_color(), &mut obs);
    let v = n.forward(&obs, scratch, logits);
    if s.current_color() == me { v } else { -v }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (a_key, b_key) = (args[1].clone(), args[2].clone());
    let board: u64 = args[3].parse().expect("board seed");
    let games: u64 = args.get(5).map_or(48, |s| s.parse().expect("games"));
    let threads: usize = args.get(6).map_or(default_threads(), |s| s.parse().expect("threads"));
    let turns: Vec<u32> = args.get(7).and_then(|s| s.strip_prefix("turns=")).map_or(vec![10, 20, 40, 80], |s| s.split(',').map(|t| t.parse().expect("turn")).collect());
    let net = load_net(a_key.strip_prefix("NN:").unwrap_or(&a_key)).expect("A's network");
    let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(board));
    let rules = GameConfig {
        map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
        dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(1),
        catan_map: Some(map.clone()), record_history: false, fixed_seats: true, ..Default::default()
    };
    let spaces = (ObservationSpace::for_network(&map, net.obs_size), net.opening.as_ref().map(|o| ObservationSpace::for_network(&map, o.obs_size)));
    let nodes: Vec<u8> = match args[4].strip_prefix("top:") {
        Some(k) => {
            let k: usize = k.parse().expect("top:K");
            let game = Game::new(&COLORS[..2], &rules);
            let mut spots: Vec<(f64, u8)> = game.playable_actions.iter().filter_map(|a| match a.kind {
                ActionKind::BuildSettlement(n) => Some((map.node_production[n as usize].iter().sum::<f64>(), n)),
                _ => None,
            }).collect();
            spots.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            spots.iter().take(k).map(|&(_, n)| n).collect()
        }
        None => args[4].split(',').map(|s| s.parse().expect("node id")).collect(),
    };
    let cols = 2 + turns.len() + 4 + 1; // ply1, open, turns..., road1, settle1, city1, dev1, end
    for &forced in &nodes {
        let rows: Vec<(u32, Vec<f32>)> = par_map(games, threads, |g| {
            let seed = 960_000 + g;
            let mut game = Game::new(&COLORS[..2], &GameConfig { seed: Some(seed), ..rules.clone() });
            let mut a = from_key(&a_key, COLORS[0], seed).expect("A");
            let mut b = from_key(&b_key, COLORS[1], seed + 1).expect("B");
            let (mut scratch, mut logits) = (Scratch::default(), Vec::new());
            let mut vals = vec![f32::NAN; cols];
            let mut first = true;
            let mut was_opening = true;
            let (mut roads, mut settles, mut cities, mut devs) = (0u32, 0u32, 0u32, 0u32);
            while !game.is_over() && game.state.num_turns < 400 {
                let actions = game.playable_actions.clone();
                let mover = game.state.current_color();
                let action = if first && mover == COLORS[0] {
                    first = false;
                    *actions.iter().find(|x| matches!(x.kind, ActionKind::BuildSettlement(n) if n == forced)).expect("forced node is a legal first settlement")
                } else if actions.len() == 1 {
                    actions[0]
                } else if mover == COLORS[0] {
                    a.decide(&game, &actions)
                } else {
                    b.decide(&game, &actions)
                };
                let built = if mover == COLORS[0] && !game.state.is_initial_build_phase {
                    match action.kind {
                        ActionKind::BuildRoad(_) => { roads += 1; Some((0, roads)) }
                        ActionKind::BuildSettlement(_) => { settles += 1; Some((1, settles)) }
                        ActionKind::BuildCity(_) => { cities += 1; Some((2, cities)) }
                        ActionKind::BuyDevelopmentCard => { devs += 1; Some((3, devs)) }
                        _ => None,
                    }
                } else {
                    None
                };
                game.execute(action).expect("legal");
                if let Some((kind, count)) = built {
                    let col = 2 + turns.len() + kind;
                    if count == 1 && vals[col].is_nan() {
                        vals[col] = value_for(&net, &spaces, &game, COLORS[0], &mut scratch, &mut logits);
                    }
                }
                if vals[0].is_nan() {
                    vals[0] = value_for(&net, &spaces, &game, COLORS[0], &mut scratch, &mut logits); // one ply: right after the pick
                }
                if was_opening && !game.state.is_initial_build_phase {
                    was_opening = false;
                    vals[1] = value_for(&net, &spaces, &game, COLORS[0], &mut scratch, &mut logits);
                }
                for (i, &t) in turns.iter().enumerate() {
                    if vals[2 + i].is_nan() && game.state.num_turns >= t {
                        vals[2 + i] = value_for(&net, &spaces, &game, COLORS[0], &mut scratch, &mut logits);
                    }
                }
            }
            let won = (game.winning_color() == Some(COLORS[0])) as u32;
            vals[cols - 1] = if won == 1 { 1.0 } else { -1.0 };
            // A game that ended before a horizon: carry the outcome (the position was decided by then).
            for v in vals.iter_mut() {
                if v.is_nan() {
                    *v = if won == 1 { 1.0 } else { -1.0 };
                }
            }
            (won, vals)
        });
        let w: u32 = rows.iter().map(|r| r.0).sum();
        let mean: Vec<f32> = (0..cols).map(|c| rows.iter().map(|r| r.1[c]).sum::<f32>() / rows.len() as f32).collect();
        let labels: Vec<String> = ["ply1".to_string(), "open".to_string()].into_iter().chain(turns.iter().map(|t| format!("t{t}")))
            .chain(["road1".to_string(), "settle1".to_string(), "city1".to_string(), "dev1".to_string(), "end".to_string()]).collect();
        println!("board {board} node {forced:3}: wr {:.1}% | {}", 100.0 * w as f32 / games as f32,
                 labels.iter().zip(&mean).map(|(l, v)| format!("{l} {v:+.3}")).collect::<Vec<_>>().join(" | "));
    }
}
