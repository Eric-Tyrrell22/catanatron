//! Feature mining against a board specialist: on one board, plays the
//! generalist against itself and at every real decision records both
//! networks' values of the position (the specialist's minus the
//! generalist's is "what the specialist knows about this board") together
//! with candidate hand features computed from the state, for a regression.
//!
//!   cargo run --release --example feature_mine -- <generalist.ctnn> <specialist.ctnn> <board seed> [games] [sims] [threads] > rows.tsv
use catanatron::encoding::{add_production, road_distances, ObservationSpace};
use catanatron::map::{build_map, MapType, NumberPlacement};
use catanatron::mcts::{load_net, MctsConfig, MctsPlayer, Utility};
use catanatron::nn::Scratch;
use catanatron::parallel::{default_threads, par_map};
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::*;

const CAP: u8 = 6;

fn afford(p: &state::PlayerState, cost: &Freqdeck) -> (f32, f32) {
    // (can build now, exactly one card short)
    let short: u32 = cost.iter().zip(&p.resources).map(|(&c, &h)| c.saturating_sub(h)).sum();
    ((short == 0) as u8 as f32, (short == 1) as u8 as f32)
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
        "board", "game", "turn", "initial", "kind", "me_vps", "opp_vps", "v_gen", "v_spec",
        // baselines the observation already carries
        "my_prod", "opp_prod", "my_open2", "opp_open2", "my_cards", "opp_cards", "my_roads_left", "my_setts_left",
        // candidate features (me)
        "my_secure", "my_secure_pips", "my_secure_top", "my_contested", "my_contested_top",
        "my_far3", "my_far3_top", "my_far4", "my_far4_top", "my_best_reach_value", "my_plan",
        "my_can_road", "my_short_road", "my_can_sett", "my_short_sett", "my_can_city", "my_short_city", "my_can_dev", "my_short_dev",
        // candidate features (opponent)
        "opp_secure", "opp_secure_pips", "opp_secure_top", "opp_contested", "opp_contested_top",
        "opp_far3", "opp_far3_top", "opp_far4", "opp_far4_top", "opp_best_reach_value", "opp_plan",
        "opp_can_road", "opp_can_sett", "opp_can_city", "opp_can_dev",
        // board-wide
        "open_top1", "open_top2", "open_top3", "open_top4", "open_rich", "open_count",
    ];
    println!("{}", cols.join("\t"));
    let rows = par_map(games, threads, |g| {
        let seed = 900_000 + g;
        let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(board));
        let rules = GameConfig {
            map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
            dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(seed),
            catan_map: Some(map.clone()), record_history: false, ..Default::default()
        };
        let mut game = Game::new(&COLORS[..2], &rules);
        let obs_space = ObservationSpace::for_network(&map, gen.obs_size);
        let mut obs = vec![0.0f32; gen.obs_size];
        let mut scratch = Scratch::default();
        let mut logits = Vec::new();
        let mut out = String::new();
        let mut k = 0u64;
        let num_land = map.land_nodes.iter().filter(|&&x| x).count();
        while !game.is_over() && game.state.num_turns < 400 {
            let actions = game.playable_actions.clone();
            let me = game.state.current_color();
            if actions.len() == 1 {
                game.execute(actions[0]).expect("legal");
                continue;
            }
            let s = &game.state;
            let b = &s.board;
            let opp = s.colors.iter().copied().find(|&x| x != me).unwrap();
            obs_space.encode_into(s, me, &mut obs);
            let v_gen = gen.forward(&obs, &mut scratch, &mut logits);
            let v_spec = spec.forward(&obs, &mut scratch, &mut logits);
            let pips = |nd: usize| {
                let mut p = [0.0f32; 5];
                add_production(s, nd as NodeId, 36.0, &mut p);
                p.iter().sum::<f32>()
            };
            let prod = |c: Color| {
                let p = s.player(c);
                let mut t = [0.0f32; 5];
                for &n in p.settlements.iter() { add_production(s, n, 36.0, &mut t); }
                for &n in p.cities.iter() { add_production(s, n, 72.0, &mut t); }
                t.iter().sum::<f32>()
            };
            let d_me = road_distances(b, me, CAP);
            let d_opp = road_distances(b, opp, CAP);
            let open: Vec<usize> = (0..num_land).filter(|&nd| b.buildable[nd]).collect();
            let mut tops: Vec<f32> = open.iter().map(|&nd| pips(nd)).collect();
            tops.sort_by(|a, b| b.total_cmp(a));
            let top = |i: usize| tops.get(i).copied().unwrap_or(0.0);
            let rich = tops.iter().filter(|&&p| p >= 10.0).count() as f32;
            // Per-side spot summaries from the two distance maps.
            let side = |mine: &[u8; 96], theirs: &[u8; 96]| -> Vec<f32> {
                let (mut secure, mut secure_pips, mut secure_top, mut contested, mut contested_top) = (0.0, 0.0, 0.0f32, 0.0, 0.0f32);
                let (mut far3, mut far3_top, mut far4, mut far4_top, mut best_value) = (0.0, 0.0f32, 0.0, 0.0f32, 0.0f32);
                let mut plan: Vec<f32> = Vec::new();
                for &nd in &open {
                    let (r, t, p) = (mine[nd], theirs[nd], pips(nd));
                    if r >= 5 { continue; }
                    if r == 3 { far3 += 1.0; far3_top = far3_top.max(p); }
                    if r == 4 { far4 += 1.0; far4_top = far4_top.max(p); }
                    if r < t { secure += 1.0; secure_pips += p; secure_top = secure_top.max(p); }
                    else if r == t { contested += 1.0; contested_top = contested_top.max(p); }
                    if r <= t && r <= 4 { let v = p - 2.0 * r as f32; best_value = best_value.max(v); plan.push(v); }
                }
                plan.sort_by(|a, b| b.total_cmp(a));
                let fade = [1.0, 0.8, 0.64, 0.51, 0.41];
                let plan_v: f32 = plan.iter().take(5).zip(fade).map(|(v, f)| v.max(0.0) * f).sum();
                vec![secure, secure_pips, secure_top, contested, contested_top, far3, far3_top, far4, far4_top, best_value, plan_v]
            };
            let (pm, po) = (s.player(me), s.player(opp));
            let open2 = |d: &[u8; 96]| open.iter().filter(|&&nd| d[nd] <= 2).count() as f32;
            let kind = match actions[0].kind {
                ActionKind::BuildSettlement(_) if s.is_initial_build_phase => "place_settlement",
                ActionKind::BuildRoad(_) if s.is_initial_build_phase => "place_road",
                ActionKind::MoveRobber { .. } => "robber",
                ActionKind::DiscardResource(_) => "discard",
                ActionKind::Roll => "pre_roll",
                _ => "turn",
            };
            let mut row: Vec<String> = vec![
                board.to_string(), g.to_string(), s.num_turns.to_string(), (s.is_initial_build_phase as u8).to_string(), kind.to_string(),
                pm.actual_victory_points.to_string(), po.actual_victory_points.to_string(), format!("{v_gen:.4}"), format!("{v_spec:.4}"),
                format!("{:.2}", prod(me)), format!("{:.2}", prod(opp)), open2(&d_me).to_string(), open2(&d_opp).to_string(),
                pm.resources.iter().sum::<u32>().to_string(), po.resources.iter().sum::<u32>().to_string(),
                pm.roads_available.to_string(), pm.settlements_available.to_string(),
            ];
            let f = |x: f32| format!("{x:.2}");
            row.extend(side(&d_me, &d_opp).into_iter().map(f));
            for cost in [&ROAD_COST, &SETTLEMENT_COST, &CITY_COST, &DEV_CARD_COST] {
                let (c, sh) = afford(pm, cost);
                row.push(f(c)); row.push(f(sh));
            }
            row.extend(side(&d_opp, &d_me).into_iter().map(f));
            for cost in [&ROAD_COST, &SETTLEMENT_COST, &CITY_COST, &DEV_CARD_COST] {
                row.push(f(afford(po, cost).0));
            }
            row.extend([top(0), top(1), top(2), top(3), rich, open.len() as f32].into_iter().map(f));
            out.push_str(&row.join("\t"));
            out.push('\n');
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
