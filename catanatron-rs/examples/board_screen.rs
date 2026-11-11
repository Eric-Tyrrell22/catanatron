//! Screens random OpenHex 1v1 boards for how far a network's opening play is
//! from deep search's: candidates for board specialists (the boards where
//! the network has the most to learn).
//!
//! For each board (game seed `first + i`; the board is the one `Game::new`
//! draws from that seed), the setup phase is played with search, and at each
//! of the four settlement decisions it records
//!   - regret: the search value of its best settlement minus that of the
//!     network's top choice (mover's value scale, about 2 per unit of win
//!     probability), and
//!   - kl: KL(search visits || network prior).
//! Plus board features, for choosing a varied set.
//!
//!   cargo run --release --example board_screen -- <model.ctnn> <boards> [first seed] [sims] [threads] > screen.tsv
//!
//! With SEEDS=a,b,c in the environment it screens those seeds instead (the
//! board count and first seed are then ignored) and prints, for every
//! settlement decision, the network's and the search's spot ("decision"
//! lines: the spot's production by resource, port, value).
use catanatron::map::{scarcest_resource, BoardStyle, CatanMap, MapType, NumberPlacement};
use catanatron::mcts::{load_net, MctsConfig, MctsPlayer, Utility};
use catanatron::parallel::{default_threads, par_map};
use catanatron::state::DiceMode;
use catanatron::*;

fn pips(n: Option<u8>) -> u32 {
    n.map_or(0, |n| 6 - (7 - n as i32).unsigned_abs())
}

fn rules(seed: u64) -> GameConfig {
    GameConfig {
        map_type: MapType::Base,
        vps_to_win: 15,
        discard_limit: 9,
        friendly_robber: FriendlyRobber::PublicPoints,
        dice: DiceMode::OpenHex,
        number_placement: NumberPlacement::SpiralRandomCorner,
        seed: Some(seed),
        record_history: false,
        ..Default::default()
    }
}

/// Tab-separated board features (see the header in `main`).
fn features(m: &CatanMap) -> String {
    let mut by = [0u32; 5];
    let mut desert_ring = 0;
    for &i in &m.land_tiles {
        let t = &m.tiles[i];
        match t.resource() {
            Some(r) => by[r.idx()] += pips(t.number()),
            None => {
                let (x, y, z) = t.coordinate;
                desert_ring = x.abs().max(y.abs()).max(z.abs());
            }
        }
    }
    let node_pips = |n: usize| m.adjacent_tiles[n].iter().map(|&i| pips(m.tiles[i].number())).sum::<u32>();
    let best_node = (0..m.num_nodes).map(node_pips).max().unwrap_or(0);
    // Nodes touching two or more 6/8 tiles.
    let hot = (0..m.num_nodes)
        .filter(|&n| m.adjacent_tiles[n].iter().filter(|&&i| pips(m.tiles[i].number()) == 5).count() >= 2)
        .count();
    let ows = by[Resource::Ore.idx()] + by[Resource::Wheat.idx()] + by[Resource::Sheep.idx()];
    format!(
        "{}\t{}\t{}\t{}\t{}\t{ows}\t{desert_ring}\t{best_node}\t{hot}\t{}\t{}\t{:?}",
        by[Resource::Wood.idx()], by[Resource::Brick.idx()], by[Resource::Sheep.idx()], by[Resource::Wheat.idx()], by[Resource::Ore.idx()],
        BoardStyle::ScarceOws.matches(m) as u8, BoardStyle::PortStart.matches(m) as u8, scarcest_resource(m)
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = args.get(1).expect("model path");
    let boards: u64 = args.get(2).expect("number of boards").parse().expect("boards");
    let first: u64 = args.get(3).map_or(1_000_000, |s| s.parse().expect("first seed"));
    let sims: u32 = args.get(4).map_or(1600, |s| s.parse().expect("sims"));
    let threads: usize = args.get(5).map_or(default_threads(), |s| s.parse().expect("threads"));
    let net = load_net(model).expect("model");
    let config = MctsConfig {
        simulations: sims,
        c_puct: 1.0,
        fpu_reduction: 0.25,
        utility: Utility { margin: 0.1, speed: 0.3, speed_turns: 150.0 },
        ..Default::default()
    };
    println!("seed\tregret_sum\tregret_max\tkl_sum\tagree\twood\tbrick\tsheep\twheat\tore\tows\tdesert_ring\tbest_node\thot_nodes\tscarce\tport_start\tgroup");
    let seeds: Vec<u64> = match std::env::var("SEEDS") {
        Ok(list) => list.split(',').map(|x| x.trim().parse().expect("seed")).collect(),
        Err(_) => (first..first + boards).collect(),
    };
    let detail = std::env::var("SEEDS").is_ok();
    // A settlement spot: production pips by resource, and its port.
    let spot = |m: &CatanMap, n: NodeId| {
        let mut parts = Vec::new();
        for &ti in &m.adjacent_tiles[n as usize] {
            if let Some(r) = m.tiles[ti].resource() {
                parts.push(format!("{}{}", &format!("{r:?}")[..2], pips(m.tiles[ti].number())));
            }
        }
        let port = m.port_nodes.iter().find(|(_, ns)| ns.contains(&n)).map(|(r, _)| match r {
            Some(r) => format!(" port 2:1 {r:?}"),
            None => " port 3:1".to_string(),
        });
        format!("{}{}", parts.join("+"), port.unwrap_or_default())
    };
    let rows = par_map(seeds.len() as u64, threads, |i| {
        let seed = seeds[i as usize];
        let mut lines = String::new();
        let mut g = Game::new(&COLORS[..2], &rules(seed));
        let (mut regret_sum, mut regret_max, mut kl_sum, mut agree) = (0f32, 0f32, 0f32, 0u32);
        let mut k = 0u64;
        while g.state.is_initial_build_phase {
            let mut m = MctsPlayer::new(g.state.current_color(), net.clone(), config.clone(), seed * 16 + k);
            k += 1;
            let visits = m.search(&g).expect("search");
            let best = (0..visits.len()).max_by_key(|&j| visits[j]).expect("a move");
            if matches!(g.playable_actions[best].kind, ActionKind::BuildSettlement(_)) {
                let (stats, _) = m.root_stats();
                let top_prior = (0..stats.len()).max_by(|&a, &b| stats[a].0.total_cmp(&stats[b].0)).expect("a move");
                let q_best = stats[best].2.expect("best move visited");
                // An unvisited top prior: search rejected it outright; count it at the worst visited value.
                let q_net = stats[top_prior].2.unwrap_or_else(|| stats.iter().filter_map(|s| s.2).fold(f32::INFINITY, f32::min));
                let regret = (q_best - q_net).max(0.0);
                let total: f32 = visits.iter().sum::<u32>() as f32;
                let kl: f32 = stats
                    .iter()
                    .zip(&visits)
                    .filter(|(_, &n)| n > 0)
                    .map(|(s, &n)| {
                        let q = n as f32 / total;
                        q * (q / s.0.max(1e-6)).ln()
                    })
                    .sum();
                if detail && top_prior != best {
                    let node = |j: usize| match g.playable_actions[j].kind {
                        ActionKind::BuildSettlement(n) => n,
                        _ => unreachable!(),
                    };
                    let m = &g.state.board.map;
                    lines.push_str(&format!(
                        "decision\t{seed}\tnet {} (q {:+.2}, prior {:.2})\tsearch {} (q {:+.2}, prior {:.2})\n",
                        spot(m, node(top_prior)), q_net, stats[top_prior].0, spot(m, node(best)), q_best, stats[best].0
                    ));
                }
                regret_sum += regret;
                regret_max = regret_max.max(regret);
                kl_sum += kl;
                agree += (top_prior == best) as u32;
            }
            g.execute(g.playable_actions[best]).expect("legal");
        }
        format!("{lines}{seed}\t{regret_sum:.4}\t{regret_max:.4}\t{kl_sum:.4}\t{agree}\t{}", features(&g.state.board.map))
    });
    for r in rows {
        println!("{r}");
    }
}
