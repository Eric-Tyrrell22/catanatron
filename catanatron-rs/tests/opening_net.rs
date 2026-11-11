//! An opening network attached to the main one ("main@opening") takes the
//! initial placements and nothing else.
use catanatron::map::MapType;
use catanatron::players::from_key;
use catanatron::state::DiceMode;
use catanatron::*;

const MAIN: &str = "python/openhex_models/ham.ctnn";
const OPENING: &str = "bench/model.ctnn"; // a different network with its own input layout

fn game_at(after_placements: bool) -> Game {
    let rules = GameConfig {
        map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
        dice: DiceMode::OpenHex, seed: Some(77), record_history: false, ..Default::default()
    };
    let mut g = Game::new(&COLORS[..2], &rules);
    if after_placements {
        let mut bot = from_key("R", COLORS[0], 1).unwrap();
        let mut other = from_key("R", COLORS[1], 2).unwrap();
        while g.state.is_initial_build_phase || g.state.num_turns < 6 {
            let actions = g.playable_actions.clone();
            let c = g.state.current_color();
            let a = if c == COLORS[0] { bot.decide(&g, &actions) } else { other.decide(&g, &actions) };
            g.execute(a).unwrap();
        }
    }
    g
}

#[test]
fn opening_network_takes_the_placements_only() {
    if !std::path::Path::new(MAIN).exists() || !std::path::Path::new(OPENING).exists() {
        eprintln!("skipped: weights not present");
        return;
    }
    let both = format!("{MAIN}@{OPENING}");
    let c = COLORS[0];
    // First placement: the combined player decides like the opening network.
    let g = game_at(false);
    assert!(g.state.is_initial_build_phase);
    let actions = g.playable_actions.clone();
    let pick = |key: &str| from_key(key, c, 5).unwrap().decide(&g, &actions);
    assert_eq!(pick(&format!("NN:{both}")), pick(&format!("NN:{OPENING}")), "placement by the opening net");
    // Mid-game: like the main network.
    let g = game_at(true);
    assert!(!g.state.is_initial_build_phase);
    let actions = g.playable_actions.clone();
    let c = g.state.current_color();
    let pick = |key: &str| from_key(key, c, 5).unwrap().decide(&g, &actions);
    assert_eq!(pick(&format!("NN:{both}")), pick(&format!("NN:{MAIN}")), "mid-game by the main net");
    // Searched play works with both and still reaches a decision.
    let mut m = from_key(&format!("MCTS:60,c1,f0.25:{both}"), c, 9).unwrap();
    let a = m.decide(&g, &actions);
    assert!(actions.contains(&a));
}
