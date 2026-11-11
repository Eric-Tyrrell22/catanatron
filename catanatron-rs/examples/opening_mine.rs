//! Opening feature mining against a board specialist: at every initial
//! settlement placement in generalist self-play on one board, one row per
//! legal spot with the generalist's and the specialist's prior on it and
//! precomputed properties of the spot. Regressing the specialist's
//! preference shift on the properties says what it values in an opening
//! spot that the generalist doesn't.
//!
//!   cargo run --release --example opening_mine -- <generalist.ctnn> <specialist.ctnn> <board seed> [games] [sims] [threads] > rows.tsv
use catanatron::encoding::{add_production, ActionSpace, ObservationSpace};
use catanatron::map::{build_map, MapType, NumberPlacement, TileKind};
use catanatron::mcts::{load_net, MctsConfig, MctsPlayer, Utility};
use catanatron::nn::Scratch;
use catanatron::parallel::{default_threads, par_map};
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::*;

fn softmax_over(logits: &[f32], slots: &[usize]) -> Vec<f32> {
    let m = slots.iter().map(|&s| logits[s]).fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = slots.iter().map(|&s| (logits[s] - m).exp()).collect();
    let z: f32 = e.iter().sum();
    e.iter().map(|x| x / z).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let gen = load_net(&args[1]).expect("generalist");
    let spec = load_net(&args[2]).expect("specialist");
    let board: u64 = args[3].parse().expect("board seed");
    let games: u64 = args.get(4).map_or(50, |s| s.parse().expect("games"));
    let sims: u32 = args.get(5).map_or(100, |s| s.parse().expect("sims"));
    let threads: usize = args.get(6).map_or(default_threads(), |s| s.parse().expect("threads"));
    let play = MctsConfig { simulations: sims, c_puct: 1.0, fpu_reduction: 0.25, utility: Utility { margin: 0.1, speed: 0.3, speed_turns: 150.0 }, ..Default::default() };
    let cols = [
        "board", "game", "placement", "seat_order", "node", "p_gen", "p_spec",
        "pips", "pips_wood", "pips_brick", "pips_sheep", "pips_wheat", "pips_ore", "n_resources", "n_tiles", "n_numbers",
        "has_6_8", "n_6_8", "has_2_12", "scarcity_pips", "ore_wheat", "wood_brick",
        "port_any", "port_match", "port_3to1",
        "expand_best", "expand_count", "expand_second",
        "own_combined_resources", "own_new_resources", "own_combined_pips", "own_dist",
        "opp_dist", "opp_contest",
    ];
    println!("{}", cols.join("\t"));
    let rows = par_map(games, threads, |g| {
        let seed = 910_000 + g;
        let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(board));
        let rules = GameConfig {
            map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
            dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(seed),
            catan_map: Some(map.clone()), record_history: false, ..Default::default()
        };
        let mut game = Game::new(&COLORS[..2], &rules);
        let obs_space = ObservationSpace::for_network(&map, gen.obs_size);
        let act_space = ActionSpace::new(&map);
        let mut obs = vec![0.0f32; gen.obs_size];
        let mut scratch = Scratch::default();
        let (mut lg, mut ls) = (Vec::new(), Vec::new());
        let num_land = map.land_nodes.iter().filter(|&&x| x).count();
        // Board supply per resource (pips over all tiles), for scarcity weighting.
        let mut supply = [0.0f32; 5];
        for t in &map.tiles {
            if let TileKind::Land { resource: Some(r), number: Some(n), .. } = t.kind {
                supply[r.idx()] += [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 5.0, 4.0, 3.0, 2.0, 1.0][n as usize];
            }
        }
        let mut out = String::new();
        let mut k = 0u64;
        let mut placements = [0u32; 4];
        while !game.is_over() && game.state.is_initial_build_phase {
            let actions = game.playable_actions.clone();
            let me = game.state.current_color();
            if actions.len() > 1 && matches!(actions[0].kind, ActionKind::BuildSettlement(_)) {
                let s = &game.state;
                let b = &s.board;
                let opp = s.colors.iter().copied().find(|&x| x != me).unwrap();
                obs_space.encode_into(s, me, &mut obs);
                gen.forward(&obs, &mut scratch, &mut lg);
                spec.forward(&obs, &mut scratch, &mut ls);
                let slots: Vec<usize> = actions.iter().map(|a| act_space.encode(s, a).expect("encodable")).collect();
                let (pg, ps) = (softmax_over(&lg, &slots), softmax_over(&ls, &slots));
                let pips_of = |nd: NodeId| {
                    let mut p = [0.0f32; 5];
                    add_production(s, nd, 36.0, &mut p);
                    p
                };
                // Graph distance over land nodes (roads ignored: nothing but the placements exists yet).
                let dist_from = |src: &[NodeId]| {
                    let mut d = vec![u8::MAX; map.num_nodes];
                    let mut q: std::collections::VecDeque<NodeId> = Default::default();
                    for &n in src { d[n as usize] = 0; q.push_back(n); }
                    while let Some(n) = q.pop_front() {
                        for &(m, _) in &map.node_edges[n as usize] {
                            if (m as usize) < num_land && d[m as usize] == u8::MAX { d[m as usize] = d[n as usize] + 1; q.push_back(m); }
                        }
                    }
                    d
                };
                let own: Vec<NodeId> = s.player(me).settlements.iter().copied().collect();
                let theirs: Vec<NodeId> = s.player(opp).settlements.iter().copied().collect();
                let (d_own, d_opp) = (dist_from(&own), dist_from(&theirs));
                let placement = placements[me.idx()] + 1;
                let seat_order = s.seat(me);
                for (i, a) in actions.iter().enumerate() {
                    let ActionKind::BuildSettlement(nd) = a.kind else { continue };
                    let p = pips_of(nd);
                    let total: f32 = p.iter().sum();
                    let n_res = p.iter().filter(|&&x| x > 0.0).count();
                    let tiles: Vec<&map::Tile> = map.adjacent_tiles[nd as usize].iter().map(|&ti| &map.tiles[ti]).collect();
                    let numbers: Vec<u8> = tiles.iter().filter_map(|t| if let TileKind::Land { number: Some(n), .. } = t.kind { Some(n) } else { None }).collect();
                    let mut distinct = numbers.clone(); distinct.sort(); distinct.dedup();
                    let n68 = numbers.iter().filter(|&&n| n == 6 || n == 8).count();
                    let has212 = numbers.iter().any(|&n| n == 2 || n == 12);
                    let scarcity: f32 = p.iter().zip(&supply).map(|(&x, &sup)| x / sup.max(1.0)).sum::<f32>() * 10.0;
                    let port = map.port_nodes.iter().find(|(_, ns)| ns.contains(&nd));
                    let (port_any, port_match, port_3) = match port {
                        None => (0.0, 0.0, 0.0),
                        Some((None, _)) => (1.0, 0.0, 1.0),
                        Some((Some(r), _)) => (1.0, p[r.idx()] / 10.0, 0.0),
                    };
                    // Expansion: open spots within two roads of this spot (distance rule applied to it).
                    let d_here = dist_from(&[nd]);
                    let mut near: Vec<f32> = (0..num_land)
                        .filter(|&m| m != nd as usize && b.buildable[m] && d_here[m] == 2)
                        .map(|m| pips_of(m as NodeId).iter().sum())
                        .collect();
                    near.sort_by(|a, b| b.total_cmp(a));
                    let own_p = own.iter().map(|&n| pips_of(n)).fold([0.0f32; 5], |mut acc, q| { for i in 0..5 { acc[i] += q[i]; } acc });
                    let combined = (0..5).filter(|&i| own_p[i] + p[i] > 0.0).count();
                    let new_res = (0..5).filter(|&i| own_p[i] == 0.0 && p[i] > 0.0).count();
                    let min_dist = |d: &Vec<u8>| if d[nd as usize] == u8::MAX { 9.0 } else { d[nd as usize] as f32 };
                    let contest = (0..num_land).filter(|&m| b.buildable[m] && d_here[m] == 2 && d_opp[m] <= 2).count();
                    let f = |x: f32| format!("{x:.2}");
                    let row = [
                        board.to_string(), g.to_string(), placement.to_string(), seat_order.to_string(), nd.to_string(), format!("{:.5}", pg[i]), format!("{:.5}", ps[i]),
                        f(total), f(p[0]), f(p[1]), f(p[2]), f(p[3]), f(p[4]), n_res.to_string(), tiles.len().to_string(), distinct.len().to_string(),
                        f((n68 > 0) as u8 as f32), n68.to_string(), f(has212 as u8 as f32), f(scarcity), f((p[3] > 0.0 && p[4] > 0.0) as u8 as f32), f((p[0] > 0.0 && p[1] > 0.0) as u8 as f32),
                        f(port_any), f(port_match), f(port_3),
                        f(near.first().copied().unwrap_or(0.0)), near.len().to_string(), f(near.get(1).copied().unwrap_or(0.0)),
                        combined.to_string(), new_res.to_string(), f(own_p.iter().sum::<f32>() + total), f(if own.is_empty() { 0.0 } else { min_dist(&d_own) }),
                        f(if theirs.is_empty() { 0.0 } else { min_dist(&d_opp) }), contest.to_string(),
                    ];
                    out.push_str(&row.join("\t"));
                    out.push('\n');
                }
                placements[me.idx()] += 1;
            }
            if actions.len() == 1 {
                game.execute(actions[0]).expect("legal");
                continue;
            }
            let mut player = MctsPlayer::new(me, gen.clone(), play.clone(), seed * 1000 + k);
            k += 1;
            let visits = player.search(&game).expect("search");
            let pick = (0..visits.len()).max_by_key(|&j| visits[j]).unwrap();
            game.execute(actions[pick]).expect("legal");
        }
        out
    });
    for r in rows {
        print!("{r}");
    }
}
