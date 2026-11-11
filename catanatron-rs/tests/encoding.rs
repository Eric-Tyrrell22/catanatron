use std::collections::HashSet;

use catanatron::encoding::{ActionSpace, ObsFeatures, ObservationSpace};
use catanatron::map::MapType;
use catanatron::players::{from_key, Player};
use catanatron::*;

fn bots(n: usize, seed: u64) -> Vec<Box<dyn Player>> {
    let keys = ["R", "W", "R", "W"];
    (0..n).map(|i| from_key(keys[i], COLORS[i], seed * 7 + i as u64).unwrap()).collect()
}

/// Plays random games, calling `check` before every decision.
fn for_each_decision(games: u64, mut check: impl FnMut(&Game)) {
    for seed in 0..games {
        let n = 2 + (seed % 3) as usize;
        let config = GameConfig {
            seed: Some(seed),
            friendly_robber: [FriendlyRobber::Off, FriendlyRobber::ActualPoints, FriendlyRobber::PublicPoints][seed as usize % 3],
            dice: if seed % 2 == 1 { catanatron::state::DiceMode::Deck } else { catanatron::state::DiceMode::Fair },
            map_type: if seed % 4 == 0 { MapType::Mini } else { MapType::Base },
            ..Default::default()
        };
        let mut g = Game::new(&COLORS[..n], &config);
        let mut p = bots(n, seed);
        while !g.is_over() {
            check(&g);
            g.play_tick(&mut p).unwrap();
        }
    }
}

#[test]
fn every_playable_action_round_trips() {
    let mut checked = 0;
    for_each_decision(120, |g| {
        let space = ActionSpace::new(&g.state.board.map);
        let mut seen = HashSet::new();
        for a in &g.playable_actions {
            let i = space.encode(&g.state, a).unwrap_or_else(|| panic!("{a} has no slot"));
            assert!(seen.insert(i), "two playable actions share slot {i}");
            assert_eq!(space.decode(&g.state, i), *a, "slot {i} decodes wrong");
            checked += 1;
        }
        let mask = space.mask(&g.state, &g.playable_actions);
        assert_eq!(mask.iter().filter(|&&m| m).count(), g.playable_actions.len());
    });
    assert!(checked > 100_000);
}

#[test]
fn action_space_layout() {
    let g = Game::new(&COLORS, &GameConfig { seed: Some(1), ..Default::default() });
    let space = ActionSpace::new(&g.state.board.map);
    // 2 + 5 + 72 + 54 + 54 + 2 + 20 + 1 + 5 + 19*4 + 60 + 3 + 3
    assert_eq!(space.len(), 357);
    let mini = Game::new(&COLORS[..2], &GameConfig { seed: Some(1), map_type: MapType::Mini, ..Default::default() });
    assert!(ActionSpace::new(&mini.state.board.map).len() < 357);
    // Distinct slots decode to distinct actions.
    let decoded: HashSet<_> = (0..space.len()).map(|i| space.decode(&g.state, i)).collect();
    assert_eq!(decoded.len(), space.len());
}

#[test]
fn robber_victims_are_relative_seats() {
    let g = Game::new(&COLORS, &GameConfig { seed: Some(3), ..Default::default() });
    let s = &g.state;
    let space = ActionSpace::new(&s.board.map);
    let me = s.current_color();
    let next = s.colors[(s.current_player_index + 1) % 4];
    let a = Action::new(me, ActionKind::MoveRobber { coordinate: (0, 0, 0), victim: Some(next) });
    let i = space.encode(s, &a).unwrap();
    assert_eq!(space.slot(i), catanatron::encoding::ActionSlot::MoveRobber { tile: 0, victim: 1 });
}

#[test]
fn trade_responses_are_encodable() {
    let mut g = Game::new(&COLORS[..3], &GameConfig { seed: Some(4), ..Default::default() });
    while g.state.is_initial_build_phase {
        g.execute(g.playable_actions[0]).unwrap();
    }
    let c = g.state.current_color();
    for i in 0..3 {
        g.state.players[i].resources = [2, 2, 2, 2, 2];
    }
    g.state.bank = [13, 13, 13, 13, 13];
    g.state.player_mut(c).has_rolled = true;
    g.execute(Action::new(c, ActionKind::OfferTrade { offering: [1, 0, 0, 0, 0], asking: [0, 1, 0, 0, 0] })).unwrap();
    while g.state.current_prompt == ActionPrompt::DecideTrade {
        let space = ActionSpace::new(&g.state.board.map);
        let accept = g.playable_actions.iter().find(|a| a.kind == ActionKind::AcceptTrade).copied().unwrap();
        assert!(space.encode(&g.state, &accept).is_some());
        g.execute(accept).unwrap();
    }
    let space = ActionSpace::new(&g.state.board.map);
    for a in &g.playable_actions {
        let i = space.encode(&g.state, a).unwrap();
        assert_eq!(space.decode(&g.state, i), *a);
    }
}

#[test]
fn observation_names_and_ranges() {
    let g = Game::new(&COLORS, &GameConfig { seed: Some(1), ..Default::default() });
    let obs = ObservationSpace::new(&g.state.board.map);
    let names = obs.names(&g.state);
    assert_eq!(names.len(), obs.len());
    assert_eq!(names.iter().collect::<HashSet<_>>().len(), names.len(), "duplicate names");
    println!("observation size: {}", obs.len());

    for_each_decision(60, |g| {
        let obs = ObservationSpace::new(&g.state.board.map);
        for &c in g.state.colors.iter() {
            for (k, v) in obs.encode(&g.state, c).into_iter().enumerate() {
                assert!(v.is_finite() && (0.0..=3.0).contains(&v), "value {v} out of range at {}", obs.names(&g.state)[k]);
            }
        }
    });
}

#[test]
fn observation_hides_opponent_cards() {
    let mut g = Game::new(&COLORS, &GameConfig { seed: Some(9), ..Default::default() });
    let mut p = bots(4, 9);
    for _ in 0..400 {
        g.play_tick(&mut p).unwrap();
    }
    let me = g.state.current_color();
    let obs = ObservationSpace::new(&g.state.board.map);
    let before = obs.encode(&g.state, me);

    // Rearrange every opponent's hand and dev cards, keeping totals.
    let mut s = g.state.clone();
    for i in 0..4 {
        if s.players[i].color == me {
            continue;
        }
        let total: u32 = s.players[i].resources.iter().sum();
        s.players[i].resources = [total, 0, 0, 0, 0];
        let devs: u32 = s.players[i].dev_cards.iter().sum();
        s.players[i].dev_cards = [0, 0, 0, 0, devs];
    }
    assert_eq!(obs.encode(&s, me), before, "opponent card types leaked");

    // The observer's own hand is visible.
    let mut s2 = g.state.clone();
    s2.player_mut(me).resources[0] += 1;
    assert_ne!(obs.encode(&s2, me), before);
}

#[test]
fn observation_is_seat_relative() {
    // The same position seen by two different players differs, and the
    // observer's own buildings always land in the P0 block.
    let mut g = Game::new(&COLORS, &GameConfig { seed: Some(2), ..Default::default() });
    while g.state.is_initial_build_phase {
        g.execute(g.playable_actions[0]).unwrap();
    }
    let obs = ObservationSpace::new(&g.state.board.map);
    let names = obs.names(&g.state);
    for &c in g.state.colors.iter() {
        let v = obs.encode(&g.state, c);
        let own: f32 = names
            .iter()
            .zip(&v)
            .filter(|(n, _)| n.contains("_P0_Settlement"))
            .map(|(_, x)| x)
            .sum();
        assert_eq!(own as usize, g.state.player(c).settlements.len());
    }
}

#[test]
fn sparse_observation_matches_dense() {
    let mut nz = Vec::new();
    for_each_decision(12, |g| {
        let space = ObservationSpace::new(&g.state.board.map);
        for &me in &g.state.colors {
            let dense = space.encode(&g.state, me);
            let expected: Vec<(u32, f32)> =
                dense.iter().enumerate().filter(|(_, &v)| v != 0.0).map(|(i, &v)| (i as u32, v)).collect();
            let n = space.encode_sparse_into(&g.state, me, &mut nz);
            assert_eq!(&nz[..n], &expected[..]);
        }
    });
}

#[test]
fn dice_features_track_the_balanced_deck() {
    use catanatron::encoding::DICE_FEATURES;
    let config = GameConfig { seed: Some(5), dice: catanatron::state::DiceMode::Deck, ..Default::default() };
    let mut g = Game::new(&COLORS[..2], &config);
    let base = ObservationSpace::new(&g.state.board.map);
    let space = ObservationSpace::with_dice(&g.state.board.map, true);
    assert_eq!(space.len(), base.len() + DICE_FEATURES);
    assert_eq!(space.names(&g.state).len(), space.len());
    let mut p = bots(2, 5);
    let mut nz = Vec::new();
    while !g.is_over() {
        let me = g.state.current_color();
        let obs = space.encode(&g.state, me);
        assert_eq!(&obs[..base.len()], &base.encode(&g.state, me)[..], "base layout unchanged");
        let deck = g.state.dice_deck.unwrap();
        let left = if deck.is_empty() { 36 } else { deck.len() };
        assert_eq!(obs[space.len() - 1], left as f32 / 36.0);
        let n = space.encode_sparse_into(&g.state, me, &mut nz);
        let dense_nz = obs.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(n, dense_nz);
        g.play_tick(&mut p).unwrap();
    }
}

#[test]
fn opponent_hand_features_and_layout_selection() {
    use catanatron::encoding::{DICE_FEATURES, HAND_FEATURES};
    let config = GameConfig { seed: Some(9), dice: catanatron::state::DiceMode::Deck, ..Default::default() };
    let mut g = Game::new(&COLORS[..2], &config);
    let map = g.state.board.map.clone();
    let base = ObservationSpace::new(&map).len();
    for (dice, hands) in [(false, false), (true, false), (false, true), (true, true)] {
        let space = ObservationSpace::with_features(&map, ObsFeatures { dice, hands, ..Default::default() });
        let want = base + if dice { DICE_FEATURES } else { 0 } + if hands { HAND_FEATURES } else { 0 };
        assert_eq!(space.len(), want);
        assert_eq!(ObservationSpace::for_network(&map, want).len(), want);
        assert_eq!(space.names(&g.state).len(), want);
    }
    let space = ObservationSpace::with_features(&map, ObsFeatures { dice: true, hands: true, ..Default::default() });
    let mut p = bots(2, 9);
    while !g.is_over() {
        let me = g.state.current_color();
        let other = *g.state.colors.iter().find(|&&c| c != me).unwrap();
        let obs = space.encode(&g.state, me);
        let hand = &obs[space.len() - HAND_FEATURES..];
        for (k, r) in RESOURCES.iter().enumerate() {
            assert_eq!(hand[k], g.state.player(other).resources[r.idx()] as f32 / 10.0);
        }
        assert!(hand[5..].iter().all(|&v| v == 0.0), "no third or fourth player");
        g.play_tick(&mut p).unwrap();
    }
}

#[test]
fn board_symmetries_preserve_the_topology() {
    check_board_symmetries(MapType::Base);
    check_board_symmetries(MapType::Mini);
}

fn check_board_symmetries(map_type: MapType) {
    use catanatron::encoding::symmetries;
    let g = Game::new(&COLORS[..2], &GameConfig { seed: Some(1), map_type, ..Default::default() });
    let map = &g.state.board.map;
    let (obs, act) = (ObservationSpace::with_features(map, ObsFeatures::ALL), ActionSpace::new(map));
    let syms = symmetries(map, &obs, &act);
    assert_eq!(syms.len(), 12);
    assert!(syms[0].obs.iter().enumerate().all(|(i, &j)| i == j) && syms[0].actions.iter().enumerate().all(|(i, &j)| i == j));
    let is_perm = |p: &[usize]| p.iter().copied().collect::<HashSet<_>>().len() == p.len() && p.iter().all(|&j| j < p.len());
    let n_nodes = map.land_nodes.iter().filter(|&&x| x).count();
    let land_edges: Vec<(u8, u8)> = (0..map.edges.len()).filter(|&i| map.land_edge[i]).map(|i| map.edges[i]).collect();
    let all: HashSet<Vec<usize>> = syms.iter().map(|s| s.obs.clone()).collect();
    for s in &syms {
        assert!(is_perm(&s.obs) && is_perm(&s.actions));
        // Node permutation, from the node blocks (14 features each): the
        // block at new position m came from node n.
        let mut node = vec![0; n_nodes];
        for m in 0..n_nodes {
            node[s.obs[m * 14] / 14] = m;
        }
        let edge_set: HashSet<(u8, u8)> = land_edges.iter().copied().collect();
        for &(a, b) in &land_edges {
            let (x, y) = (node[a as usize] as u8, node[b as usize] as u8);
            assert!(edge_set.contains(&(x.min(y), x.max(y))), "edge maps to an edge");
        }
        let corner_sets: HashSet<Vec<u8>> = map.land_tiles.iter().map(|&ti| {
            let mut v = map.tiles[ti].nodes.to_vec();
            v.sort();
            v
        }).collect();
        for set in &corner_sets {
            let mut v: Vec<u8> = set.iter().map(|&n| node[n as usize] as u8).collect();
            v.sort();
            assert!(corner_sets.contains(&v), "tile corners map to tile corners");
        }
        // The extras block's per-tile entries move with their tile's block.
        let (o_tile, o_x) = (14 * n_nodes + 4 * land_edges.len(), obs.extras_offset(map).unwrap());
        for u in 0..map.land_tiles.len() {
            let t = (s.obs[o_tile + u * 8] - o_tile) / 8;
            assert_eq!(s.obs[o_x + u * 11], o_x + t * 11, "tile extras follow the tile");
        }
        // Settlement slot of node n moves with it.
        let o_settle = 7 + land_edges.len();
        for n in 0..n_nodes {
            assert_eq!(s.actions[o_settle + node[n]], o_settle + n);
        }
        // Closed under composition.
        for t in &syms {
            let composed: Vec<usize> = (0..s.obs.len()).map(|i| t.obs[s.obs[i]]).collect();
            assert!(all.contains(&composed), "symmetries form a group");
        }
    }
}

/// The observation must not depend on the types of other players' hidden
/// development cards: re-dealing them (same counts) leaves it unchanged.
#[test]
fn observation_hides_opponent_dev_card_types() {
    use catanatron::encoding::DEV_FEATURES;
    let mut checked = 0;
    for seed in 0..20 {
        let config = GameConfig { seed: Some(seed), dice: catanatron::state::DiceMode::Deck, ..Default::default() };
        let mut g = Game::new(&COLORS[..2], &config);
        let map = g.state.board.map.clone();
        let space = ObservationSpace::with_features(&map, ObsFeatures::ALL);
        assert_eq!(space.len(), ObservationSpace::new(&map).len() + 12 + catanatron::encoding::DICE_HISTORY_FEATURES + 15 + DEV_FEATURES + 19 * 11 + 4 + 6
            + 54 * catanatron::encoding::SPOT_PER_NODE + 4 * catanatron::encoding::SPOT_PER_PLAYER
            + 54 * catanatron::encoding::REACH_PER_NODE + 4 * catanatron::encoding::REACH_PER_PLAYER + 1);
        let mut p = bots(2, seed);
        while !g.is_over() {
            let me = g.state.current_color();
            let other = *g.state.colors.iter().find(|&&c| c != me).unwrap();
            let held = g.state.player(other).num_dev_cards();
            if held > 0 {
                let before = space.encode(&g.state, me);
                // Every other split of the same count into types.
                let mut s = g.state.clone();
                for rotation in 1..5 {
                    let hand = &mut s.player_mut(other).dev_cards;
                    *hand = [0; 5];
                    for k in 0..held as usize {
                        hand[(k + rotation) % 5] += 1;
                    }
                    assert_eq!(space.encode(&s, me), before, "observation leaked hidden dev card types");
                }
                checked += 1;
            }
            g.play_tick(&mut p).unwrap();
        }
    }
    assert!(checked > 0, "no positions with hidden opponent dev cards");
}

/// Networks are matched to a layout by input count, so every combination
/// of optional blocks must have a distinct size, on every map.
#[test]
fn observation_layouts_have_distinct_sizes() {
    use catanatron::map::{build_map, MapType, NumberPlacement};
    for mt in [MapType::Base, MapType::Mini] {
        let map = build_map(mt, NumberPlacement::OfficialSpiral, &mut catanatron::rng::Rng::new(0));
        let mut sizes = HashSet::new();
        for k in 0..128 {
            let f = ObsFeatures {
                dice: k & 1 != 0,
                hands: k & 2 != 0,
                devs: k & 4 != 0,
                extras: k & 8 != 0,
                dice_history: k & 16 != 0,
                spots: k & 32 != 0,
                reach: k & 64 != 0,
            };
            if f.dice_history && !f.dice {
                continue;
            }
            let len = ObservationSpace::with_features(&map, f).len();
            assert!(sizes.insert(len), "{mt:?}: layout {f:?} has a duplicate size {len}");
            assert_eq!(ObservationSpace::for_network(&map, len).len(), len);
        }
    }
}

/// The OpenHex dice block reports the deck's own next-roll odds for the
/// observer and the opponent.
#[test]
fn dice_history_features_match_the_deck() {
    let config = GameConfig { seed: Some(4), dice: catanatron::state::DiceMode::OpenHex, ..Default::default() };
    let mut g = Game::new(&COLORS[..2], &config);
    let mut p = bots(2, 4);
    let f = ObsFeatures { dice: true, dice_history: true, ..Default::default() };
    let space = ObservationSpace::with_features(&g.state.board.map.clone(), f);
    let names = space.names(&g.state);
    let mut checked = 0;
    while !g.is_over() && checked < 200 {
        let me = g.state.current_color();
        let obs = space.encode(&g.state, me);
        if let Some(d) = g.state.dice_deck {
            let seat = g.state.seat(me);
            let mine = d.probabilities(seat, 2);
            let theirs = d.probabilities((seat + 1) % 2, 2);
            for t in 2..=12 {
                let at = |n: String| obs[names.iter().position(|x| *x == n).unwrap()];
                assert!((at(format!("DICE_P_{t}_IF_ME_ROLLS")) - mine[t] * 6.0).abs() < 1e-6);
                assert!((at(format!("DICE_P_{t}_IF_NEXT_ROLLS")) - theirs[t] * 6.0).abs() < 1e-6);
            }
            checked += 1;
        }
        g.play_tick(&mut p).unwrap();
    }
    assert!(checked > 50);
}

/// The spots block agrees with the engine: legal spots are the playable
/// settlements, pips are the adjacent tiles' odds, closeness follows roads,
/// and the port discount matches the bank trade rates offered.
#[test]
fn spot_features_match_the_engine() {
    use catanatron::encoding::{SPOT_PER_NODE, SPOT_PER_PLAYER};
    let mut checked = 0;
    for_each_decision(12, |g| {
        let s = &g.state;
        let map = &s.board.map;
        let space = ObservationSpace::with_features(map, ObsFeatures { spots: true, ..Default::default() });
        let me = s.current_color();
        let obs = space.encode(s, me);
        let o = space.spots_offset().unwrap();
        let n_nodes = map.land_nodes.iter().filter(|&&x| x).count();
        let node = |nd: usize, j: usize| obs[o + nd * SPOT_PER_NODE + j];
        let mine: HashSet<u8> = s.board.buildable_node_ids(me, s.is_initial_build_phase).into_iter().collect();
        for nd in 0..n_nodes {
            // Legal for me exactly where the engine lets me settle.
            assert_eq!(node(nd, 5) == 1.0, mine.contains(&(nd as u8)), "node {nd} legality");
            assert_eq!(node(nd, 7) == 1.0, s.board.buildable[nd], "node {nd} open");
            // Pips: 6 x the odds of each adjacent producing tile (robber excluded).
            let mut pips = [0.0f32; 5];
            for &ti in &map.adjacent_tiles[nd] {
                let t = &map.tiles[ti];
                if t.coordinate == s.board.robber_coordinate {
                    continue;
                }
                if let (Some(r), Some(n)) = (t.resource(), t.number()) {
                    pips[r.idx()] += 6.0 * catanatron::map::number_probability(n) as f32;
                }
            }
            for (k, &p) in pips.iter().enumerate() {
                assert!((node(nd, k) - p).abs() < 1e-5, "node {nd} pips");
            }
            // In my network <=> closeness 1; outside it, closeness < 1.
            let in_net = s.board.in_network(me, nd as u8);
            assert_eq!(node(nd, 8) == 1.0, in_net, "node {nd} closeness");
            // One road away: some land edge from a network node, not taken
            // by an opponent's road, through a node I can extend from.
            if node(nd, 8) > 0.6 && node(nd, 8) < 0.7 {
                assert!(map.node_edges[nd].iter().any(|&(v, ei)| {
                    map.land_edge[ei] && s.board.roads[ei].is_none() && s.board.in_network(me, v)
                }), "node {nd} is one road from the network");
            }
        }
        // My port discount matches the rates the bank actually offers me.
        let rates: Vec<(Resource, u8)> = g.playable_actions.iter().filter_map(|a| match a.kind {
            ActionKind::MaritimeTrade { give, rate, .. } => Some((give, rate)),
            _ => None,
        }).collect();
        let o_p = o + n_nodes * SPOT_PER_NODE;
        for (give, rate) in rates {
            assert_eq!(obs[o_p + give.idx()], (4 - rate) as f32 / 2.0, "port discount for {give:?}");
        }
        let _ = SPOT_PER_PLAYER;
        checked += 1;
    });
    assert!(checked > 1000, "checked {checked} decisions");
}

/// Board symmetries move each node's spots entries with the node.
#[test]
fn spot_features_follow_board_symmetries() {
    use catanatron::encoding::{symmetries, SPOT_PER_NODE};
    let config = GameConfig { seed: Some(3), ..Default::default() };
    let mut g = Game::new(&COLORS[..2], &config);
    let mut p = bots(2, 3);
    for _ in 0..60 {
        g.play_tick(&mut p).unwrap();
    }
    let map = g.state.board.map.clone();
    let space = ObservationSpace::with_features(&map, ObsFeatures::ALL);
    let syms = symmetries(&map, &space, &ActionSpace::new(&map));
    let o = space.spots_offset().unwrap();
    let n_nodes = map.land_nodes.iter().filter(|&&x| x).count();
    for s in &syms {
        for m in 0..n_nodes {
            let n = s.obs[m * 14] / 14; // node block at m came from node n
            for j in 0..SPOT_PER_NODE {
                assert_eq!(s.obs[o + m * SPOT_PER_NODE + j], o + n * SPOT_PER_NODE + j);
            }
        }
    }
}

/// Distillation trains on specialists' games under random board symmetries,
/// so a symmetry must be exact end to end: the same game played on a
/// transformed board (tiles, numbers and ports moved; every move mapped
/// through the symmetry) encodes, at every decision and for both players, as
/// the original observation permuted. Covers everything the encoder reads
/// from the board: buildings, roads, robber, ports, spots.
#[test]
fn encoding_commutes_with_board_symmetries() {
    use catanatron::encoding::symmetries;
    use std::sync::Arc;
    let mut checked = 0;
    for seed in 0..3u64 {
        let config = GameConfig {
            seed: Some(seed),
            vps_to_win: 15,
            discard_limit: 9,
            friendly_robber: FriendlyRobber::PublicPoints,
            dice: catanatron::state::DiceMode::OpenHex,
            number_placement: catanatron::map::NumberPlacement::SpiralRandomCorner,
            ..Default::default()
        };
        let map = Game::new(&COLORS[..2], &config).state.board.map.clone();
        let space = ObservationSpace::with_features(&map, ObsFeatures::ALL);
        let act = ActionSpace::new(&map);
        let n_nodes = map.land_nodes.iter().filter(|&&x| x).count();
        let (land, ports) = map.layout_spec();
        for (k, s) in symmetries(&map, &space, &act).iter().enumerate() {
            // Node n moves to node[n]; tile at coordinate c to the tile whose
            // block it lands in.
            let mut node = vec![0u8; n_nodes];
            for m in 0..n_nodes {
                node[s.obs[m * 14] / 14] = m as u8;
            }
            let land_edges = (0..map.edges.len()).filter(|&i| map.land_edge[i]).count();
            let o_tile = 14 * n_nodes + 4 * land_edges;
            let coords: Vec<_> = map.land_tiles.iter().map(|&ti| map.tiles[ti].coordinate).collect();
            let moved_land: Vec<_> = (0..coords.len())
                .map(|u| {
                    let t = (s.obs[o_tile + u * 8] - o_tile) / 8; // tile block at u came from tile t
                    let (_, r, n) = land.iter().find(|(c, _, _)| *c == coords[t]).copied().unwrap();
                    (coords[u], r, n)
                })
                .collect();
            // Ports move with their nodes. The standard ports are not
            // symmetric, so the moved ones need not sit at port slots:
            // with_layout places the land and the port lookup is set directly
            // (it is all that rules and encoder read).
            let mut moved = map.with_layout(&moved_land, &ports).unwrap();
            moved.port_nodes = map.port_nodes.iter().map(|(r, ns)| (*r, ns.iter().map(|&n| node[n as usize]).collect())).collect();
            let rotated = Arc::new(moved);
            let mut a = Game::new(&COLORS[..2], &GameConfig { catan_map: Some(map.clone()), ..config.clone() });
            let mut b = Game::new(&COLORS[..2], &GameConfig { catan_map: Some(rotated.clone()), ..config.clone() });
            let space_b = ObservationSpace::with_features(&rotated, ObsFeatures::ALL);
            let act_b = ActionSpace::new(&rotated);
            let mut p = bots(2, seed + 10 * k as u64);
            while !a.is_over() && a.state.num_turns < 120 {
                assert_eq!(b.is_over(), false, "seed {seed} symmetry {k}: games diverged");
                for &c in &COLORS[..2] {
                    let (oa, ob) = (space.encode(&a.state, c), space_b.encode(&b.state, c));
                    for i in 0..oa.len() {
                        // (Sums like spot pips add the same terms in another order.)
                        assert!((ob[i] - oa[s.obs[i]]).abs() < 1e-5, "seed {seed} symmetry {k}: feature {i} ({}) differs: {} vs {}",
                                space.names(&a.state)[s.obs[i]], ob[i], oa[s.obs[i]]);
                    }
                }
                checked += 1;
                let color = a.state.current_color();
                let action = p.iter_mut().find(|q| q.color() == color).unwrap().decide(&a, &a.playable_actions);
                let mapped = match act.encode(&a.state, &action) {
                    Some(i) => act_b.decode(&b.state, s.actions.iter().position(|&x| x == i).unwrap()),
                    None => action, // board-independent (trade offers)
                };
                a.execute(action).unwrap();
                b.execute(mapped).unwrap_or_else(|e| panic!("seed {seed} symmetry {k}: {e}"));
            }
        }
    }
    assert!(checked > 2000, "checked {checked} decisions");
}
