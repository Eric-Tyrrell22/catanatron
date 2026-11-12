//! Which cheap label best tracks the true value of a mid-game position? Plays
//! network-only games with a key, samples positions every S turns after the
//! opening and, for each, computes:
//!   truth  win rate for the mover over T network-only playouts to the end (dice vary)
//!   z      the source game's actual outcome for the mover (+1/-1)
//!   v1     the value head on the position itself
//!   q      the root value of a PUCT search (SIMS sims, no noise)
//!   h<K>_<P> the mean value-head reading K turns later over the first P of the horizon playouts
//! One line per position: "pos <game> <turn> <mover vp> <opp vp> truth z v1 q h3_4 h3_12 h5_4 h5_12 h10_4 h10_12 h20_4 h20_12",
//! plus "econ <game> <turn> <FEATURE_NAMES means> | 54 x P(mover builds node) | 54 x P(opponent builds node)" at +5 turns, "econ_k <game> <turn> <playout> <features>" per playout
//! (the first line printed is "features <names>")
//! and "obs <game> <turn> <observation>" for probing the trunk (runs/econ_probe.py); then "time ..." lines. runs/midgame_horizon.py summarises.
//!
//!   cargo run --release --example midgame_horizon -- <main.ctnn[@opening.ctnn]> [games=1] [threads] [every=5 turns] [truth playouts=64] [horizon playouts=12] [sims=400]
use catanatron::encoding::ObservationSpace;
use catanatron::map::{build_map, number_probability, MapType, NumberPlacement};
use catanatron::state::{PlayerState, State};
use catanatron::types::DevCard;
use catanatron::board::BuildingType;
use catanatron::mcts::{load_net, MctsConfig, MctsPlayer};
use catanatron::nn::{PolicyValueNet, Scratch};
use catanatron::parallel::{default_threads, par_map};
use catanatron::players::from_key;
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::*;
use std::time::Instant;

const HORIZONS: [u32; 4] = [3, 5, 10, 20];

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

/// Everything we can say about the position at the horizon, for feature mining: a vector of named scalars
/// (`FEATURE_NAMES`, each a change over the horizon or a state at the horizon, from the mover's seat) plus
/// per land node 0 empty / 1 mover / 2 opponent.
struct Outcome { feats: Vec<f32>, nodes: Vec<u8> }

const FEATURE_NAMES: &[&str] = &[
    "vp_gain_me", "vp_gain_opp", "vp_swing", "settles_me", "settles_opp", "cities_me", "cities_opp", "roads_me", "roads_opp",
    "knights_me", "knights_opp", "monopoly_me", "monopoly_opp", "yop_me", "yop_opp", "roadbuild_me", "roadbuild_opp", "devbuy_me", "devbuy_opp",
    "vp_cards_me", "vp_cards_opp", "dev_in_hand_me", "dev_in_hand_opp", "longest_len_me", "longest_len_opp", "has_road_me", "has_road_opp", "has_army_me", "has_army_opp",
    "cards_me", "cards_opp", "wood_me", "brick_me", "sheep_me", "wheat_me", "ore_me", "wood_opp", "brick_opp", "sheep_opp", "wheat_opp", "ore_opp",
    "robber_on_me", "robber_on_opp", "robber_pips", "prod_me", "prod_opp", "game_over", "won",
];

fn production(s: &State, c: Color) -> f32 {
    // expected cards per roll for `c` (robbed tile excluded), settlements 1, cities 2
    let mut p = 0.0f64;
    for t in s.board.map.land_tiles.iter().map(|&i| &s.board.map.tiles[i]) {
        if t.coordinate == s.board.robber_coordinate { continue; }
        let Some(n) = t.number() else { continue };
        let w: f64 = t.nodes.iter().map(|&nd| match s.board.buildings[nd as usize] { Some((col, BuildingType::Settlement)) if col == c => 1.0, Some((col, BuildingType::City)) if col == c => 2.0, _ => 0.0 }).sum();
        p += w * number_probability(n);
    }
    p as f32
}

fn outcome_of(game: &Game, me: Color, before: &Game) -> Outcome {
    let (s, b) = (&game.state, &before.state);
    let opp = s.colors.iter().copied().find(|&c| c != me).unwrap();
    let nodes = (0..54).map(|n| match s.board.buildings[n] { Some((c, _)) if c == me => 1, Some(_) => 2, None => 0 }).collect();
    let (pm, po, bm, bo) = (s.player(me), s.player(opp), b.player(me), b.player(opp));
    let d = |a: u32, b: u32| a as f32 - b as f32;
    let dev = |p: &PlayerState, q: &PlayerState, c: DevCard| d(p.played_dev_cards[c.idx()], q.played_dev_cards[c.idx()]);
    let robber = s.board.map.tile_at(s.board.robber_coordinate);
    let on = |c: Color| robber.map_or(0.0, |t| t.nodes.iter().filter(|&&n| matches!(s.board.buildings[n as usize], Some((col, _)) if col == c)).count() as f32);
    let won = game.winning_color() == Some(me);
    let feats = vec![
        d(pm.actual_victory_points, bm.actual_victory_points), d(po.actual_victory_points, bo.actual_victory_points),
        d(pm.actual_victory_points, bm.actual_victory_points) - d(po.actual_victory_points, bo.actual_victory_points),
        d(pm.settlements.len() as u32, bm.settlements.len() as u32), d(po.settlements.len() as u32, bo.settlements.len() as u32),
        d(pm.cities.len() as u32, bm.cities.len() as u32), d(po.cities.len() as u32, bo.cities.len() as u32),
        d(pm.roads.len() as u32, bm.roads.len() as u32), d(po.roads.len() as u32, bo.roads.len() as u32),
        dev(pm, bm, DevCard::Knight), dev(po, bo, DevCard::Knight), dev(pm, bm, DevCard::Monopoly), dev(po, bo, DevCard::Monopoly),
        dev(pm, bm, DevCard::YearOfPlenty), dev(po, bo, DevCard::YearOfPlenty), dev(pm, bm, DevCard::RoadBuilding), dev(po, bo, DevCard::RoadBuilding),
        d(pm.dev_cards.iter().sum::<u32>() + pm.played_dev_cards.iter().sum::<u32>(), bm.dev_cards.iter().sum::<u32>() + bm.played_dev_cards.iter().sum::<u32>()),
        d(po.dev_cards.iter().sum::<u32>() + po.played_dev_cards.iter().sum::<u32>(), bo.dev_cards.iter().sum::<u32>() + bo.played_dev_cards.iter().sum::<u32>()),
        pm.dev_cards[DevCard::VictoryPoint.idx()] as f32, po.dev_cards[DevCard::VictoryPoint.idx()] as f32,
        pm.dev_cards.iter().sum::<u32>() as f32, po.dev_cards.iter().sum::<u32>() as f32,
        pm.longest_road_length as f32, po.longest_road_length as f32,
        pm.has_road as u8 as f32, po.has_road as u8 as f32, pm.has_army as u8 as f32, po.has_army as u8 as f32,
        pm.resources.iter().sum::<u32>() as f32, po.resources.iter().sum::<u32>() as f32,
        pm.resources[0] as f32, pm.resources[1] as f32, pm.resources[2] as f32, pm.resources[3] as f32, pm.resources[4] as f32,
        po.resources[0] as f32, po.resources[1] as f32, po.resources[2] as f32, po.resources[3] as f32, po.resources[4] as f32,
        on(me), on(opp), robber.and_then(|t| t.number()).map_or(0.0, |n| (6 - (n as i32 - 7).abs()) as f32),
        production(s, me), production(s, opp), game.is_over() as u8 as f32, won as u8 as f32,
    ];
    debug_assert_eq!(feats.len(), FEATURE_NAMES.len());
    Outcome { feats, nodes }
}

/// Plays `game` on with both seats by `key` (network only); returns (mover won, values of `me` at the horizons).
fn play_on(mut game: Game, key: &str, me: Color, seed: u64, net: &PolicyValueNet, spaces: &(ObservationSpace, Option<ObservationSpace>), horizons: Option<&[u32]>) -> (bool, Vec<f32>) {
    play_on_full(game, key, me, seed, net, spaces, horizons).0
}

fn play_on_full(mut game: Game, key: &str, me: Color, seed: u64, net: &PolicyValueNet, spaces: &(ObservationSpace, Option<ObservationSpace>), horizons: Option<&[u32]>) -> ((bool, Vec<f32>), Option<Outcome>) {
    let before = game.copy_without_history();
    let mut at_h5: Option<Outcome> = None;
    game.state.rng = Rng::new(seed); // fresh dice from here on (a copied position carries the source game's generator)
    let mut a = from_key(key, COLORS[0], seed).expect("A");
    let mut b = from_key(key, COLORS[1], seed + 1).expect("B");
    let (mut scratch, mut logits) = (Scratch::default(), Vec::new());
    let start = game.state.num_turns;
    let hs = horizons.unwrap_or(&[]);
    let mut vals = vec![f32::NAN; hs.len()];
    while !game.is_over() && game.state.num_turns < 400 {
        if at_h5.is_none() && horizons.is_some() && game.state.num_turns >= start + 5 {
            at_h5 = Some(outcome_of(&game, me, &before));
        }
        for (i, &h) in hs.iter().enumerate() {
            if vals[i].is_nan() && game.state.num_turns >= start + h {
                vals[i] = value_for(net, spaces, &game, me, &mut scratch, &mut logits);
            }
        }
        if horizons.is_some() && vals.iter().all(|v| !v.is_nan()) {
            break; // horizon-only playout: done once every horizon is read
        }
        let actions = game.playable_actions.clone();
        let mover = game.state.current_color();
        let action = if actions.len() == 1 { actions[0] } else if mover == COLORS[0] { a.decide(&game, &actions) } else { b.decide(&game, &actions) };
        game.execute(action).expect("legal");
    }
    let won = game.winning_color() == Some(me);
    let end = if won { 1.0 } else if game.is_over() { -1.0 } else { 0.0 };
    for v in vals.iter_mut() {
        if v.is_nan() { *v = end; } // the game ended before the horizon: the outcome stands in
    }
    if at_h5.is_none() && horizons.is_some() {
        at_h5 = Some(outcome_of(&game, me, &before)); // ended inside the horizon: the final position stands in
    }
    ((won, vals), at_h5)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    println!("features {}", FEATURE_NAMES.join(" "));
    let key = format!("NN:{}", args[1].strip_prefix("NN:").unwrap_or(&args[1]));
    let games: u64 = args.get(2).map_or(1, |s| s.parse().expect("games"));
    let threads: usize = args.get(3).map_or(default_threads(), |s| s.parse().expect("threads"));
    let every: u32 = args.get(4).map_or(5, |s| s.parse().expect("every"));
    let truth_n: u64 = args.get(5).map_or(64, |s| s.parse().expect("truth playouts"));
    let hor_n: u64 = args.get(6).map_or(12, |s| s.parse().expect("horizon playouts"));
    let sims: u32 = args.get(7).map_or(400, |s| s.parse().expect("sims"));
    let net = load_net(&args[1]).expect("network");
    let t_all = Instant::now();
    let out: Vec<(Vec<String>, [f64; 4])> = par_map(games, threads, |g| {
        let seed = 8_000_000 + g;
        let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(seed));
        let rules = GameConfig {
            map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
            dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(seed),
            catan_map: Some(map.clone()), record_history: false, fixed_seats: true, ..Default::default()
        };
        let spaces = (ObservationSpace::for_network(&map, net.obs_size), net.opening.as_ref().map(|o| ObservationSpace::for_network(&map, o.obs_size)));
        // The source game, keeping a copy at every sampled position.
        let mut game = Game::new(&COLORS[..2], &rules);
        let mut a = from_key(&key, COLORS[0], seed).expect("A");
        let mut b = from_key(&key, COLORS[1], seed + 1).expect("B");
        let mut samples: Vec<Game> = Vec::new();
        let mut last = u32::MAX;
        while !game.is_over() && game.state.num_turns < 400 {
            let s = &game.state;
            if !s.is_initial_build_phase && s.num_turns % every == 0 && s.num_turns != last && game.playable_actions.len() > 1 {
                last = s.num_turns;
                samples.push(game.copy_without_history());
            }
            let actions = game.playable_actions.clone();
            let mover = game.state.current_color();
            let action = if actions.len() == 1 { actions[0] } else if mover == COLORS[0] { a.decide(&game, &actions) } else { b.decide(&game, &actions) };
            game.execute(action).expect("legal");
        }
        let winner = game.winning_color();
        let mut lines = Vec::new();
        let mut times = [0.0f64; 4]; // truth, v1, q, horizons
        let (mut scratch, mut logits) = (Scratch::default(), Vec::new());
        for pos in &samples {
            let me = pos.state.current_color();
            let z = if winner == Some(me) { 1.0 } else if winner.is_some() { -1.0 } else { 0.0 };
            let t = Instant::now();
            let wins = (0..truth_n).filter(|&k| play_on(pos.copy_without_history(), &key, me, 9_000_000 + k, &net, &spaces, None).0).count();
            let truth = wins as f32 / truth_n as f32;
            times[0] += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let v1 = value_for(&net, &spaces, pos, me, &mut scratch, &mut logits);
            times[1] += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let mut config = MctsConfig { simulations: sims, c_puct: 1.0, fpu_reduction: 0.25, root_noise: 0.0, ..Default::default() };
            config.utility.margin = 0.1; config.utility.speed = 0.3; config.utility.speed_turns = 150.0;
            let mut player = MctsPlayer::new(me, net.clone(), config, seed);
            player.search(pos).expect("search");
            let (stats, root_value) = player.root_stats();
            let visited: u32 = stats.iter().map(|s| s.1).sum();
            let q_sum: f32 = stats.iter().filter_map(|&(_, n, q)| q.map(|q| q * n as f32)).sum();
            let q = if visited > 0 { q_sum / visited as f32 } else { root_value };
            times[2] += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let full: Vec<((bool, Vec<f32>), Option<Outcome>)> = (0..hor_n).map(|k| play_on_full(pos.copy_without_history(), &key, me, 9_500_000 + k, &net, &spaces, Some(&HORIZONS))).collect();
            let runs: Vec<Vec<f32>> = full.iter().map(|r| r.0 .1.clone()).collect();
            let outs: Vec<&Outcome> = full.iter().filter_map(|r| r.1.as_ref()).collect();
            times[3] += t.elapsed().as_secs_f64();
            // Future state at +5 turns: mean over the playouts (econ), and each playout's own vector (econ_k) for reliability.
            let k = outs.len().max(1) as f32;
            let means: Vec<String> = (0..FEATURE_NAMES.len()).map(|i| format!("{:.3}", outs.iter().map(|o| o.feats[i]).sum::<f32>() / k)).collect();
            let node_p: Vec<String> = (0..54).map(|n| format!("{:.2}", outs.iter().filter(|o| o.nodes[n] == 1).count() as f32 / k)).collect();
            let node_o: Vec<String> = (0..54).map(|n| format!("{:.2}", outs.iter().filter(|o| o.nodes[n] == 2).count() as f32 / k)).collect();
            let mut obs = vec![0.0f32; net.obs_size];
            spaces.0.encode_into(&pos.state, me, &mut obs);
            lines.push(format!("econ {g} {} {} | {} | {}", pos.state.num_turns, means.join(" "), node_p.join(" "), node_o.join(" ")));
            for (j, o) in outs.iter().enumerate() {
                lines.push(format!("econ_k {g} {} {j} {}", pos.state.num_turns, o.feats.iter().map(|v| format!("{v:.1}")).collect::<Vec<_>>().join(" ")));
            }
            lines.push(format!("obs {g} {} {}", pos.state.num_turns, obs.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(" ")));
            let mut h = Vec::new();
            for i in 0..HORIZONS.len() {
                for p in [4usize, hor_n as usize] {
                    h.push(format!("{:+.3}", runs[..p.min(runs.len())].iter().map(|r| r[i]).sum::<f32>() / p.min(runs.len()) as f32));
                }
            }
            let (mvp, ovp) = (pos.state.player(me).actual_victory_points, pos.state.players.iter().find(|p| p.color != me).map_or(0, |p| p.victory_points));
            lines.push(format!("pos {g} {} {mvp} {ovp} {truth:.3} {z:+.0} {v1:+.3} {q:+.3} {}", pos.state.num_turns, h.join(" ")));
        }
        (lines, times)
    });
    let mut times = [0.0f64; 4];
    let mut n = 0;
    for (lines, t) in &out {
        for l in lines { println!("{l}"); }
        n += lines.len();
        for i in 0..4 { times[i] += t[i]; }
    }
    println!("time positions {n} games {games}: truth {:.1}s ({truth_n} playouts each), v1 {:.3}s, search {:.1}s ({sims} sims), horizons {:.1}s ({hor_n} playouts to +20 turns); wall {:.1}s on {threads} threads",
             times[0], times[1], times[2], times[3], t_all.elapsed().as_secs_f64());
}
