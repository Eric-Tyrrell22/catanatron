//! Prints the award-reach block (ObsFeatures::award) along a game played by
//! a network against itself (network only), every few turns, from the mover's
//! view: what each seat needs for longest road and largest army and whether
//! it is still reachable. For checking the block by eye.
//!
//!   cargo run --release --example award_probe -- <board seed> <net key> [every N turns]
use catanatron::encoding::{award_offset_of, ObsFeatures, ObservationSpace, AWARD_PER_PLAYER};
use catanatron::map::{build_map, MapType, NumberPlacement};
use catanatron::players::from_key;
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::*;

const NAMES: [&str; AWARD_PER_PLAYER] = ["lr_need", "lr_ext", "lr_ok", "la_need", "la_poss", "la_ok", "vps+aw", "swing"];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let board: u64 = args[1].parse().expect("board seed");
    let key = &args[2];
    let every: u32 = args.get(3).map_or(10, |s| s.parse().expect("turns"));
    let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(board));
    let rules = GameConfig {
        map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
        dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(7),
        catan_map: Some(map.clone()), record_history: false, fixed_seats: true, ..Default::default()
    };
    let mut game = Game::new(&COLORS[..2], &rules);
    let obs_space = ObservationSpace::with_features(&map, ObsFeatures { award: true, devs: true, ..Default::default() });
    let off = award_offset_of(&obs_space).expect("award block");
    let mut obs = vec![0.0f32; obs_space.len()];
    let mut a = from_key(key, COLORS[0], 1).expect("A");
    let mut b = from_key(key, COLORS[1], 2).expect("B");
    let mut last = u32::MAX;
    println!("{:>5} {:>6} | {}", "turn", "mover", (0..2).map(|r| format!("seat {r}: {}", NAMES.iter().map(|s| format!("{s:>7}")).collect::<String>())).collect::<Vec<_>>().join(" | "));
    while !game.is_over() && game.state.num_turns < 400 {
        let actions = game.playable_actions.clone();
        let mover = game.state.current_color();
        if game.state.num_turns != last && game.state.num_turns % every == 0 && !game.state.is_initial_build_phase {
            last = game.state.num_turns;
            obs_space.encode_into(&game.state, mover, &mut obs);
            let s = &game.state;
            let p = s.player(mover);
            let row = |r: usize| (0..AWARD_PER_PLAYER).map(|j| format!("{:>7.2}", obs[off + r * AWARD_PER_PLAYER + j])).collect::<String>();
            println!("{:>5} {:>6?} | {} | {} | mover: {} vp, road {} ({}), army {}, knights {}, roads left {}",
                     s.num_turns, mover, row(0), row(1), p.victory_points, p.longest_road_length, p.has_road, p.has_army, p.played_dev_cards[0], p.roads_available);
        }
        let action = if actions.len() == 1 { actions[0] } else if mover == COLORS[0] { a.decide(&game, &actions) } else { b.decide(&game, &actions) };
        game.execute(action).expect("legal");
    }
    println!("winner {:?} after {} turns", game.winning_color(), game.state.num_turns);
}
