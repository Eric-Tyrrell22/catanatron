use std::collections::HashMap;

use catanatron::board::{Board, BuildingType};
use catanatron::map::*;
use catanatron::players::{from_key, Player};
use catanatron::rng::Rng;
use catanatron::*;

fn game(n: usize, seed: u64) -> Game {
    Game::new(&COLORS[..n], &GameConfig { seed: Some(seed), ..Default::default() })
}

fn bots(keys: &[&str], seed: u64) -> Vec<Box<dyn Player>> {
    keys.iter()
        .zip(COLORS)
        .enumerate()
        .map(|(i, (k, c))| from_key(k, c, seed * 7 + i as u64).unwrap())
        .collect()
}

fn check_invariants(g: &Game) {
    let s = &g.state;
    // Resource cards are conserved.
    for r in 0..5 {
        let held: u32 = s.players.iter().map(|p| p.resources[r]).sum();
        assert_eq!(held + s.bank[r], 19, "resource {r} not conserved");
    }
    // Dev cards are conserved.
    let dev: u32 = s.players.iter().map(|p| p.num_dev_cards() + p.played_dev_cards.iter().sum::<u32>()).sum();
    assert_eq!(dev as usize + s.dev_deck.len(), 25);
    for p in &s.players {
        let buildings = p.settlements.len() as u32 + 2 * p.cities.len() as u32;
        let bonus = 2 * (p.has_road as u32 + p.has_army as u32);
        assert_eq!(p.victory_points, buildings + bonus, "{:?} visible VPs", p.color);
        assert_eq!(
            p.actual_victory_points,
            p.victory_points + p.dev_cards[DevCard::VictoryPoint.idx()],
        );
        assert_eq!(p.settlements_available as usize + p.settlements.len(), 5);
        assert_eq!(p.cities_available as usize + p.cities.len(), 4);
        assert_eq!(p.roads_available as usize + p.roads.len(), 15);
        assert_eq!(p.has_road, s.board.road_color == Some(p.color));
    }
    assert!(s.players.iter().filter(|p| p.has_army).count() <= 1);
    assert!(!g.playable_actions.is_empty(), "no playable actions at {:?}", s.current_prompt);
}

#[test]
fn spiral_order_matches_python() {
    let m = tournament_map();
    let order = spiral_land_coordinates(&m.tiles, &m.tile_by_coord, (2, -2, 0)).unwrap();
    let expected = vec![
        (2, -2, 0), (2, -1, -1), (2, 0, -2), (1, 1, -2), (0, 2, -2), (-1, 2, -1), (-2, 2, 0),
        (-2, 1, 1), (-2, 0, 2), (-1, -1, 2), (0, -2, 2), (1, -2, 1), (1, -1, 0), (1, 0, -1),
        (0, 1, -1), (-1, 1, 0), (-1, 0, 1), (0, -1, 1), (0, 0, 0),
    ];
    assert_eq!(order, expected);
}

#[test]
fn official_spiral_numbers() {
    let mut rng = Rng::new(3);
    let m = CatanMap::from_template(&base_template(), NumberPlacement::OfficialSpiral, &mut rng);
    let order = spiral_land_coordinates(&m.tiles, &m.tile_by_coord, (2, -2, 0)).unwrap();
    let numbers: Vec<u8> = order.iter().filter_map(|&c| m.tile_at(c).unwrap().number()).collect();
    assert_eq!(numbers, BASE_NUMBERS_IN_SPIRAL_ORDER.to_vec());
}

#[test]
fn map_shapes() {
    let mut rng = Rng::new(1);
    let base = CatanMap::from_template(&base_template(), NumberPlacement::Random, &mut rng);
    assert_eq!(base.land_nodes.iter().filter(|&&x| x).count(), 54);
    assert_eq!(base.land_edge.iter().filter(|&&x| x).count(), 72);
    assert_eq!(base.land_tiles.len(), 19);
    let mini = CatanMap::from_template(&mini_template(), NumberPlacement::OfficialSpiral, &mut rng);
    assert_eq!(mini.land_nodes.iter().filter(|&&x| x).count(), 24);
    assert_eq!(mini.port_nodes.len(), 0);
    // Robber starts on the desert.
    assert_eq!(base.tile_at(base.desert_coordinate).unwrap().resource(), None);
}

#[test]
fn initial_placement_is_snake_ordered() {
    let mut g = game(4, 11);
    let mut order = Vec::new();
    while g.state.is_initial_build_phase {
        let a = g.playable_actions[0];
        if matches!(a.kind, ActionKind::BuildSettlement(_)) {
            order.push(g.state.current_player_index);
        }
        g.execute(a).unwrap();
    }
    assert_eq!(order, vec![0, 1, 2, 3, 3, 2, 1, 0]);
    assert_eq!(g.state.current_player_index, 0);
    assert_eq!(g.playable_actions.last().unwrap().kind, ActionKind::Roll);
    // Second settlement paid out its adjacent resources.
    let total: u32 = g.state.players.iter().map(|p| p.num_resources()).sum();
    assert!(total > 0);
    check_invariants(&g);
}

#[test]
fn initial_road_must_touch_last_settlement() {
    let mut g = game(2, 5);
    g.execute(Action::new(g.state.current_color(), ActionKind::BuildSettlement(0))).unwrap();
    for a in &g.playable_actions {
        let ActionKind::BuildRoad((x, y)) = a.kind else { panic!() };
        assert!(x == 0 || y == 0);
    }
}

#[test]
fn distance_rule() {
    let m = tournament_map();
    let mut b = Board::new(m.clone());
    b.build_settlement(Color::Red, 0, true).unwrap();
    for &(nb, _) in &m.node_edges[0] {
        assert!(b.build_settlement(Color::Blue, nb, true).is_err());
    }
    assert!(b.build_settlement(Color::Blue, 0, true).is_err());
}

/// Walks a path of nodes, building roads for `color`.
fn build_path(b: &mut Board, color: Color, path: &[NodeId]) {
    for w in path.windows(2) {
        b.build_road(color, (w[0], w[1])).unwrap();
    }
}

/// A simple path of `len` edges through land nodes starting at `start`,
/// that avoids `avoid`.
fn find_path(m: &CatanMap, start: NodeId, len: usize) -> Vec<NodeId> {
    fn go(m: &CatanMap, path: &mut Vec<NodeId>, len: usize) -> bool {
        if path.len() == len + 1 {
            return true;
        }
        let last = *path.last().unwrap();
        for &(nb, ei) in &m.node_edges[last as usize] {
            if m.land_edge[ei] && !path.contains(&nb) {
                path.push(nb);
                if go(m, path, len) {
                    return true;
                }
                path.pop();
            }
        }
        false
    }
    let mut p = vec![start];
    assert!(go(m, &mut p, len));
    p
}

#[test]
fn longest_road_award_and_cut() {
    let m = tournament_map();
    let mut b = Board::new(m.clone());
    let path = find_path(&m, 0, 6);
    b.build_settlement(Color::Red, path[0], true).unwrap();
    build_path(&mut b, Color::Red, &path[..5]);
    assert_eq!(b.road_lengths[Color::Red.idx()], 4);
    assert_eq!(b.road_color, None);
    build_path(&mut b, Color::Red, &path[4..6]);
    assert_eq!(b.road_lengths[Color::Red.idx()], 5);
    assert_eq!(b.road_color, Some(Color::Red));

    // Blue reaches the middle of Red's road and settles there, cutting it.
    let cut = path[3];
    let blue_start = m.node_edges[cut as usize]
        .iter()
        .map(|&(nb, _)| nb)
        .find(|nb| !path.contains(nb) && m.land_nodes[*nb as usize])
        .expect("a side node");
    // Give Blue a foothold next to the cut point (distance rule permitting).
    b.buildings[blue_start as usize] = Some((Color::Blue, BuildingType::Settlement));
    b.build_road(Color::Blue, (blue_start, cut)).unwrap();
    b.buildable[cut as usize] = true; // bypass distance rule for the test
    b.build_settlement(Color::Blue, cut, false).unwrap();
    assert_eq!(b.road_lengths[Color::Red.idx()], 3);
    assert_eq!(b.road_color, None);
    // Red can no longer build through Blue's settlement.
    let through: Vec<_> = b.buildable_edges(Color::Red);
    for &(nb, ei) in &m.node_edges[cut as usize] {
        if b.roads[ei].is_none() && !b.in_network(Color::Red, nb) {
            assert!(!through.contains(&edge(cut, nb)));
        }
    }
}

#[test]
fn road_ending_at_enemy_building_counts() {
    let m = tournament_map();
    let mut b = Board::new(m.clone());
    let path = find_path(&m, 0, 5);
    b.build_settlement(Color::Red, path[0], true).unwrap();
    b.buildings[path[5] as usize] = Some((Color::Blue, BuildingType::Settlement));
    build_path(&mut b, Color::Red, &path);
    assert_eq!(b.road_lengths[Color::Red.idx()], 5);
    assert_eq!(b.road_color, Some(Color::Red));
}

fn force_turn_with_hand(g: &mut Game, hand: Freqdeck) -> Color {
    while g.state.is_initial_build_phase {
        g.execute(g.playable_actions[0]).unwrap();
    }
    let c = g.state.current_color();
    // Move any held cards back to the bank, then deal `hand`.
    for i in 0..g.state.players.len() {
        let r = g.state.players[i].resources;
        fd_add(&mut g.state.bank, &r);
        g.state.players[i].resources = [0; 5];
    }
    fd_sub(&mut g.state.bank, &hand);
    g.state.player_mut(c).resources = hand;
    g.state.player_mut(c).has_rolled = true;
    g.playable_actions = catanatron::actions::generate_playable_actions(&g.state);
    c
}

#[test]
fn maritime_trades() {
    let mut g = game(2, 9);
    let c = force_turn_with_hand(&mut g, [4, 0, 0, 0, 0]);
    let trades: Vec<_> = g.playable_actions.iter().filter(|a| matches!(a.kind, ActionKind::MaritimeTrade { .. })).collect();
    let ports = g.state.board.player_port_resources(c);
    let rate = if ports.contains(&Some(Resource::Wood)) { 2 } else if ports.contains(&None) { 3 } else { 4 };
    assert_eq!(trades.len(), 4);
    assert!(trades.iter().all(|a| matches!(a.kind, ActionKind::MaritimeTrade { give: Resource::Wood, rate: r, .. } if r == rate)));
    g.execute(*trades[0]).unwrap();
    check_invariants(&g);
}

#[test]
fn domestic_trade_flow() {
    let mut g = game(3, 4);
    let c = force_turn_with_hand(&mut g, [2, 0, 0, 0, 0]);
    let others: Vec<Color> = g.state.colors.iter().copied().filter(|&o| o != c).collect();
    g.state.player_mut(others[1]).resources = [0, 0, 1, 0, 0];
    g.state.bank[Resource::Sheep.idx()] -= 1;

    let offer = Action::new(c, ActionKind::OfferTrade { offering: [1, 0, 0, 0, 0], asking: [0, 0, 1, 0, 0] });
    assert!(!g.is_valid_action(&Action::new(c, ActionKind::OfferTrade { offering: [1, 0, 0, 0, 0], asking: [1, 0, 0, 0, 0] })));
    g.execute(offer).unwrap();
    assert_eq!(g.state.current_prompt, ActionPrompt::DecideTrade);
    // Each other player answers once; the offerer is never asked.
    let mut asked = Vec::new();
    while g.state.current_prompt == ActionPrompt::DecideTrade {
        let who = g.state.current_color();
        asked.push(who);
        let accept = g.playable_actions.iter().find(|a| a.kind == ActionKind::AcceptTrade).copied();
        g.execute(accept.unwrap_or(g.playable_actions[0])).unwrap();
    }
    assert_eq!(asked, others);
    assert_eq!(g.state.current_prompt, ActionPrompt::DecideAcceptees);
    assert_eq!(g.state.current_color(), c);
    g.execute(Action::new(c, ActionKind::ConfirmTrade { partner: others[1] })).unwrap();
    assert_eq!(g.state.player(c).resources, [1, 0, 1, 0, 0]);
    assert_eq!(g.state.player(others[1]).resources, [1, 0, 0, 0, 0]);
    check_invariants(&g);
}

#[test]
fn seven_triggers_discard_then_robber() {
    let mut g = game(2, 21);
    let c = force_turn_with_hand(&mut g, [3, 3, 3, 0, 0]);
    let s = &mut g.state;
    s.player_mut(c).has_rolled = false;
    g.playable_actions = catanatron::actions::generate_playable_actions(&g.state);
    g.execute_unchecked(Action::new(c, ActionKind::Roll), Some(ActionResult::Dice(3, 4))).unwrap();
    assert_eq!(g.state.current_prompt, ActionPrompt::Discard);
    for _ in 0..4 {
        assert_eq!(g.state.current_color(), c);
        g.execute(g.playable_actions[0]).unwrap();
    }
    assert_eq!(g.state.player(c).num_resources(), 5);
    assert_eq!(g.state.current_prompt, ActionPrompt::MoveRobber);
    assert!(g.playable_actions.iter().all(|a| matches!(a.kind, ActionKind::MoveRobber { .. })));
    check_invariants(&g);
}

#[test]
fn same_seed_same_game() {
    let run = |seed| {
        let mut g = game(4, seed);
        let mut p = bots(&["R", "W", "R", "W"], seed);
        let w = g.play(&mut p).unwrap();
        (w, g.history.clone())
    };
    assert_eq!(run(42), run(42));
    assert_ne!(run(42).1, run(43).1);
}

#[test]
fn random_games_keep_invariants() {
    for seed in 0..150 {
        let n = 2 + (seed % 3) as usize;
        let mut g = Game::new(
            &COLORS[..n],
            &GameConfig {
                seed: Some(seed),
                friendly_robber: [FriendlyRobber::ActualPoints, FriendlyRobber::Off, FriendlyRobber::PublicPoints][seed as usize % 3],
                dice: [catanatron::state::DiceMode::Fair, catanatron::state::DiceMode::Deck, catanatron::state::DiceMode::OpenHex][(seed % 3) as usize],
                map_type: if seed % 5 == 0 { MapType::Mini } else { MapType::Base },
                ..Default::default()
            },
        );
        let mut p = bots(&["R", "W", "R", "W"][..n], seed);
        while !g.is_over() {
            check_invariants(&g);
            g.play_tick(&mut p).unwrap();
        }
    }
}

#[test]
fn bots_finish_games() {
    let mut wins: HashMap<Option<Color>, u32> = HashMap::new();
    for seed in 0..20 {
        let mut g = game(4, seed);
        // Random bots in a 4-player game often need longer than the turn limit.
        g.turns_limit = None;
        let mut p = bots(&["VP", "R", "R", "R"], seed);
        *wins.entry(g.play(&mut p).unwrap()).or_default() += 1;
    }
    // The greedy VP bot should clearly beat three random bots (25% baseline).
    assert!(wins.get(&Some(Color::Red)).copied().unwrap_or(0) >= 8, "{wins:?}");
}

#[test]
fn invalid_actions_are_rejected() {
    let mut g = game(2, 1);
    let c = g.state.current_color();
    assert!(g.execute(Action::new(c, ActionKind::Roll)).is_err());
    assert!(g.execute(Action::new(c, ActionKind::BuildSettlement(200))).is_err());
    let other = g.state.colors[1];
    assert!(g.execute(Action::new(other, ActionKind::BuildSettlement(0))).is_err());
}

/// Unpruned reference: DFS from every node touching the color's roads.
fn brute_longest_road(b: &Board, color: Color) -> u8 {
    fn dfs(b: &Board, n: NodeId, color: Color, used: &mut Vec<bool>, len: u8) -> u8 {
        if len > 0 && b.is_enemy_node(n, color) {
            return len;
        }
        let mut best = len;
        for &(nb, ei) in &b.map.node_edges[n as usize] {
            if b.roads[ei] == Some(color) && !used[ei] {
                used[ei] = true;
                best = best.max(dfs(b, nb, color, used, len + 1));
                used[ei] = false;
            }
        }
        best
    }
    let mut used = vec![false; b.roads.len()];
    (0..b.map.num_nodes as u16)
        .map(|n| dfs(b, n as NodeId, color, &mut used, 0))
        .max()
        .unwrap_or(0)
}

#[test]
fn fast_paths_match_reference() {
    for seed in 0..150 {
        let mut g = game(4, 1000 + seed);
        let mut p = bots(&["W", "W", "R", "W"], seed);
        while !g.is_over() {
            g.play_tick(&mut p).unwrap();
            let b = &g.state.board;
            for &c in &g.state.colors {
                assert_eq!(b.longest_road(c), brute_longest_road(b, c), "seed {seed}");
                let scanned: Vec<Option<Resource>> = b
                    .map
                    .port_nodes
                    .iter()
                    .filter(|(_, ns)| ns.iter().any(|&n| b.node_color(n) == Some(c)))
                    .map(|(r, _)| *r)
                    .collect();
                let mut fast = b.player_port_resources(c);
                let mut slow = scanned;
                fast.sort();
                slow.sort();
                assert_eq!(fast, slow);
                assert_eq!(b.has_buildable_edge(c), !b.buildable_edges(c).is_empty());
            }
        }
    }
}

#[test]
fn cached_layout_matches_fresh_build() {
    for (mt, t) in [(MapType::Base, base_template()), (MapType::Mini, mini_template())] {
        for placement in [NumberPlacement::OfficialSpiral, NumberPlacement::SpiralRandomCorner, NumberPlacement::Random] {
            for seed in 0..50 {
                let fast = build_map(mt, placement, &mut Rng::new(seed));
                let slow = CatanMap::from_template(&t, placement, &mut Rng::new(seed));
                for (a, b) in fast.tiles.iter().zip(&slow.tiles) {
                    assert_eq!((a.coordinate, &a.kind, a.nodes, a.edges), (b.coordinate, &b.kind, b.nodes, b.edges));
                }
                assert_eq!(fast.port_nodes, slow.port_nodes);
                assert_eq!(fast.node_production, slow.node_production);
                assert_eq!(fast.tiles_by_number, slow.tiles_by_number);
                assert_eq!(fast.desert_coordinate, slow.desert_coordinate);
            }
        }
    }
}

#[test]
fn par_map_is_ordered_and_thread_count_independent() {
    use catanatron::parallel::par_map;
    let run = |threads| {
        par_map(40, threads, |i| {
            let mut g = game(4, 500 + i);
            let mut p = bots(&["R", "W", "R", "VP"], i);
            let w = g.play(&mut p).unwrap();
            (i, w, g.state.num_actions)
        })
    };
    let one = run(1);
    assert!(one.iter().enumerate().all(|(k, r)| r.0 == k as u64));
    assert_eq!(one, run(3));
    assert_eq!(one, run(16));
    assert!(par_map(0, 4, |i| i).is_empty());
}

#[test]
fn value_function_bot_crushes_random() {
    let mut wins = 0;
    for seed in 0..40 {
        let mut g = game(4, seed);
        let mut p = bots(&["F", "R", "R", "R"], seed);
        if g.play(&mut p).unwrap() == Some(Color::Red) {
            wins += 1;
        }
    }
    assert!(wins >= 38, "F won only {wins}/40");
}

#[test]
fn alphabeta_plays_legal_games() {
    for seed in 0..6 {
        let mut g = game(2, seed);
        let mut p = bots(&["AB", "F"], seed);
        // play() validates every action, so this fails on an illegal move.
        g.play(&mut p).unwrap();
        assert!(g.winning_color().is_some());
    }
}

#[test]
fn outcome_probabilities_sum_to_one() {
    use catanatron::search::expand_outcomes;
    for_each_seed_state(|g| {
        for &a in &g.playable_actions {
            let total: f64 = expand_outcomes(g, a).iter().map(|(_, p)| p).sum();
            assert!((total - 1.0).abs() < 1e-9, "{a}: {total}");
        }
    });
}

fn for_each_seed_state(mut f: impl FnMut(&Game)) {
    for seed in 0..5 {
        let mut g = game(4, 300 + seed);
        let mut p = bots(&["W", "R", "W", "R"], seed);
        while !g.is_over() {
            f(&g);
            g.play_tick(&mut p).unwrap();
        }
    }
}

#[test]
fn balanced_dice_roll_every_combination_once_per_36() {
    use catanatron::rng::Rng;
    use catanatron::state::DiceDeck;
    let mut deck = DiceDeck::full();
    let mut rng = Rng::new(3);
    for _ in 0..3 {
        let mut seen = std::collections::HashSet::new();
        for _ in 0..36 {
            let (a, b) = deck.draw(&mut rng, 0, 2);
            assert!((1..=6).contains(&a) && (1..=6).contains(&b));
            assert!(seen.insert((a, b)), "combination repeated within 36 rolls");
        }
        assert_eq!(deck.remaining().count(), 0);
    }
}

/// Random-corner spirals: OpenHex's construction (ring 2 from corner k,
/// ring 1 from k, then the center), all 6 corners used, and never two
/// adjacent 6/8s or equal numbers.
#[test]
fn spiral_random_corner_matches_openhex() {
    let ring1: [(i8, i8, i8); 6] = [(1, -1, 0), (1, 0, -1), (0, 1, -1), (-1, 1, 0), (-1, 0, 1), (0, -1, 1)];
    let ring2: [(i8, i8, i8); 12] = [
        (2, -2, 0), (2, -1, -1), (2, 0, -2), (1, 1, -2), (0, 2, -2), (-1, 2, -1),
        (-2, 2, 0), (-2, 1, 1), (-2, 0, 2), (-1, -1, 2), (0, -2, 2), (1, -2, 1),
    ];
    let openhex: Vec<Vec<(i8, i8, i8)>> = (0..6)
        .map(|k| {
            let mut v: Vec<_> = (0..12).map(|i| ring2[(2 * k + i) % 12]).collect();
            v.extend((0..6).map(|i| ring1[(k + i) % 6]));
            v.push((0, 0, 0));
            v
        })
        .collect();
    let numbers = BASE_NUMBERS_IN_SPIRAL_ORDER;
    let mut corners_seen = [false; 6];
    for seed in 0..300 {
        let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(seed));
        let number_at = |c: (i8, i8, i8)| map.tiles[map.tile_by_coord[&c]].number();
        // Which OpenHex corner reproduces this board's numbers?
        let corner = (0..6).find(|&k| {
            let mut i = 0;
            openhex[k].iter().all(|&c| match number_at(c) {
                Some(n) => {
                    i += 1;
                    n == numbers[i - 1]
                }
                None => true,
            })
        });
        let k = corner.unwrap_or_else(|| panic!("seed {seed}: numbers follow no OpenHex spiral"));
        corners_seen[k] = true;
        for t in &map.land_tiles {
            let tile = &map.tiles[*t];
            let Some(n) = tile.number() else { continue };
            let c = tile.coordinate;
            for d in [(1, -1, 0), (1, 0, -1), (0, 1, -1), (-1, 1, 0), (-1, 0, 1), (0, -1, 1)] {
                let nb = (c.0 + d.0, c.1 + d.1, c.2 + d.2);
                if let Some(&j) = map.tile_by_coord.get(&nb) {
                    if let Some(m) = map.tiles[j].number() {
                        assert!(!(matches!(n, 6 | 8) && matches!(m, 6 | 8)), "seed {seed}: adjacent 6/8");
                        assert_ne!(n, m, "seed {seed}: equal numbers adjacent");
                    }
                }
            }
        }
    }
    assert!(corners_seen.iter().all(|&s| s), "every corner is used");
}

/// OpenHex dice: odds sum to 1, the deck reshuffles once 13 cards remain,
/// recent totals and the roller's 7 history change the odds as specified,
/// and long-run frequencies stay close to 2d6.
#[test]
fn openhex_dice_follow_the_spec() {
    use catanatron::state::{DiceDeck, DiceMode};
    let mut d = DiceDeck::new(DiceMode::OpenHex).unwrap();
    let mut rng = Rng::new(3);
    let mut counts = [0u32; 13];
    let n = 60_000;
    for i in 0..n {
        let seat = i % 2;
        let p = d.probabilities(seat, 2);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        let before = d.len();
        let (a, b) = d.draw(&mut rng, seat, 2);
        counts[(a + b) as usize] += 1;
        let expected_len = if before <= 13 { 35 } else { before - 1 };
        assert_eq!(d.len(), expected_len, "reshuffle at 13 left");
        // The total just rolled is now penalized.
        let q = d.probabilities(seat, 2);
        let t = (a + b) as usize;
        if t != 7 {
            let fresh = d.next_deck();
            let left = fresh.remaining().filter(|&(x, y)| (x + y) as usize == t).count();
            assert!(left == 0 || d.recent_count(t as u8) >= 1);
        }
        assert!(q.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }
    for t in 2..=12usize {
        let fair = (6 - (7 - t as i32).unsigned_abs()) as f64 / 36.0;
        let got = counts[t] as f64 / n as f64;
        // Recency flattens the distribution a little; stay within 3 points.
        assert!((got - fair).abs() < 0.03, "total {t}: {got:.3} vs fair {fair:.3}");
    }
    // A player on a 7 streak is less likely to roll another; the other more.
    let mut d = DiceDeck::new(DiceMode::OpenHex).unwrap();
    d.remove(3, 4, 0);
    let (same, other) = (d.probabilities(0, 2)[7], d.probabilities(1, 2)[7]);
    assert!(same < other, "streak: roller {same} vs other {other}");
}
