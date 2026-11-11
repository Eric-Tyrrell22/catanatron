//! Prints the opening-relation block (ObsFeatures::relation) for a board:
//! per land node, the board constants on the empty board, then the
//! state-dependent entries for the second seat after the first seat has
//! placed a settlement on <node> and a road, so the reply features can be
//! checked by eye (plan damage, distance, road block, shared tiles, denial).
//!
//!   cargo run --release --example relation_probe -- <board seed> <first-seat node>
use catanatron::encoding::{relation_offset_of, ObsFeatures, ObservationSpace, RELATION_CONST, RELATION_PER_NODE, RELATION_STATE};
use catanatron::map::{build_map, MapType, NumberPlacement};
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::*;

const CONST_NAMES: [&str; RELATION_CONST] = ["plan", "greedy", "g_loss", "plan_cov", "g_cov", "pips_x_sc", "hit", "pip_var", "cov5689"];
const STATE_NAMES: [&str; RELATION_STATE] =
    ["opp_hurt", "opp_d2", "opp_near", "opp_block", "opp_tiles", "opp_deny", "adj_max", "loc_rank", "own_score", "own_cover", "own_compl"];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let board: u64 = args[1].parse().expect("board seed");
    let first: u8 = args[2].parse().expect("first-seat node");
    let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(board));
    let rules = GameConfig {
        map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
        dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(1),
        catan_map: Some(map.clone()), record_history: false, fixed_seats: true, ..Default::default()
    };
    let mut game = Game::new(&COLORS[..2], &rules);
    let obs_space = ObservationSpace::with_features(&map, ObsFeatures { relation: true, ..Default::default() });
    let off = relation_offset_of(&obs_space).expect("relation block");
    let n_land = map.land_nodes.iter().filter(|&&x| x).count();
    let mut obs = vec![0.0f32; obs_space.len()];
    let pips = |n: usize| map.node_production[n].iter().sum::<f64>() * 36.0;
    println!("board {board}: constants on the empty board (top 12 nodes by pips)");
    obs_space.encode_into(&game.state, game.state.current_color(), &mut obs);
    let mut order: Vec<usize> = (0..n_land).collect();
    order.sort_by(|&a, &b| pips(b).total_cmp(&pips(a)));
    println!("{:>5} {:>5} {}", "node", "pips", CONST_NAMES.iter().map(|s| format!("{s:>9}")).collect::<String>());
    for &n in order.iter().take(12) {
        let row = &obs[off + n * RELATION_PER_NODE..off + n * RELATION_PER_NODE + RELATION_CONST];
        println!("{n:>5} {:>5.1} {}", pips(n), row.iter().map(|v| format!("{v:>9.3}")).collect::<String>());
    }
    // The first seat settles on `first` and takes its first legal road; then the second seat's view.
    let a = *game.playable_actions.iter().find(|a| matches!(a.kind, ActionKind::BuildSettlement(n) if n == first)).expect("legal first node");
    game.execute(a).expect("settle");
    let road = game.playable_actions[0];
    game.execute(road).expect("road");
    let me = game.state.current_color();
    println!("\nsecond seat's view after the first seat settled on {first} and built {:?}:", road.kind);
    obs_space.encode_into(&game.state, me, &mut obs);
    println!("{:>5} {:>5} {}", "node", "pips", STATE_NAMES.iter().map(|s| format!("{s:>9}")).collect::<String>());
    for &n in order.iter().take(16) {
        let base = off + n * RELATION_PER_NODE + RELATION_CONST;
        let row = &obs[base..base + RELATION_STATE];
        println!("{n:>5} {:>5.1} {}", pips(n), row.iter().map(|v| format!("{v:>9.3}")).collect::<String>());
    }
    println!("\nblock size {} = {n_land} x {RELATION_PER_NODE} + markers; obs {}", obs_space.len() - off, obs_space.len());
}
