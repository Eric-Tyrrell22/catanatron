//! Feature mining for openings: for a board seed, every land node's hand
//! features (production, resource pairs, scarcity, ports, numbers, geometry,
//! what the second settlement could reach) and, for each given network, its
//! first-placement policy over the nodes (network only, first seat, empty
//! board). A "main.ctnn@opening.ctnn" key places with its opening network.
//!
//!   cargo run --release --example node_features -- <board seed> [label=net.ctnn ...]
//! Output: a "#board" line, a header line, one TSV row per land node, then one
//! "#policy <label> node:prob ..." line per network.
use catanatron::encoding::{ActionSpace, ObservationSpace};
use catanatron::map::{build_map, number_probability, MapType, NumberPlacement};
use catanatron::mcts::load_net;
use catanatron::nn::Scratch;
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::*;
use std::collections::HashSet;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let board: u64 = args[1].parse().expect("board seed");
    let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(board));
    let pips = |n: u8| (6 - (7 - n as i32).abs()).max(0) as f32;
    let mut board_pips = [0.0f32; 5];
    for &ti in &map.land_tiles {
        let t = &map.tiles[ti];
        if let (Some(r), Some(num)) = (t.resource(), t.number()) {
            board_pips[r.idx()] += pips(num);
        }
    }
    let land: Vec<usize> = (0..map.num_nodes).filter(|&n| map.land_nodes[n]).collect();
    // Per node: pips by resource, numbers, tile count.
    let res_pips = |n: usize| -> ([f32; 5], Vec<u8>) {
        let mut p = [0.0f32; 5];
        let mut nums = Vec::new();
        for &ti in &map.adjacent_tiles[n] {
            let t = &map.tiles[ti];
            if let (Some(r), Some(num)) = (t.resource(), t.number()) {
                p[r.idx()] += pips(num);
                nums.push(num);
            }
        }
        (p, nums)
    };
    let total = |p: &[f32; 5]| p.iter().sum::<f32>();
    // Spots two roads away (the distance rule forbids neighbours).
    let two_away = |n: usize| -> Vec<usize> {
        let one: HashSet<usize> = map.node_edges[n].iter().map(|&(m, _)| m as usize).collect();
        let mut two: HashSet<usize> = HashSet::new();
        for &m in &one {
            for &(k, _) in &map.node_edges[m] {
                let k = k as usize;
                if k != n && !one.contains(&k) && map.land_nodes[k] {
                    two.insert(k);
                }
            }
        }
        two.into_iter().collect()
    };
    let port_of = |n: usize| -> Option<Option<usize>> {
        map.port_nodes.iter().find(|(_, nodes)| nodes.contains(&(n as u8))).map(|(r, _)| r.map(|r| r.idx()))
    };
    println!("#board {board} board_pips {}", board_pips.iter().map(|v| format!("{v:.0}")).collect::<Vec<_>>().join(","));
    println!("node\tpips\tp_wood\tp_brick\tp_sheep\tp_wheat\tp_ore\texp\tn_res\troad_pair\tcity_pair\tsettle_set\tscarce\tmax_share\tport_any\tport_match\tport31\tred\tdup_num\tn_tiles\tbest2\ttop2sum\topen2\tcompl2\tcoast");
    for &n in &land {
        let (p, nums) = res_pips(n);
        let tot = total(&p);
        let exp: f64 = nums.iter().map(|&k| number_probability(k)).sum();
        let n_res = p.iter().filter(|&&v| v > 0.0).count();
        let (brick, wood, sheep, wheat, ore) = (p[0], p[1], p[2], p[3], p[4]);
        let road_pair = brick.min(wood);
        let city_pair = ore.min(wheat);
        let settle_set = brick.min(wood).min(sheep).min(wheat);
        let scarce: f32 = (0..5).map(|r| p[r] / board_pips[r].max(1.0)).sum();
        let max_share = (0..5).map(|r| p[r] / board_pips[r].max(1.0)).fold(0.0f32, f32::max);
        let port = port_of(n);
        let port_any = port.is_some() as u8;
        let port_match = matches!(port, Some(Some(r)) if p[r] > 0.0) as u8;
        let port31 = matches!(port, Some(None)) as u8;
        let red = nums.iter().filter(|&&k| k == 6 || k == 8).count();
        let mut sorted = nums.clone();
        sorted.sort();
        sorted.dedup();
        let dup_num = nums.len() - sorted.len();
        let two = two_away(n);
        let mut two_pips: Vec<(f32, [f32; 5])> = two.iter().map(|&k| { let (q, _) = res_pips(k); (total(&q), q) }).collect();
        two_pips.sort_by(|a, b| b.0.total_cmp(&a.0));
        let best2 = two_pips.first().map_or(0.0, |x| x.0);
        let top2sum: f32 = two_pips.iter().take(2).map(|x| x.0).sum();
        let open2 = two_pips.iter().filter(|x| x.0 >= 8.0).count();
        // Complement: pips the best two-away spot adds in resources this node lacks.
        let compl2 = two_pips.first().map_or(0.0, |x| (0..5).filter(|&r| p[r] == 0.0).map(|r| x.1[r]).sum::<f32>());
        let coast = (map.adjacent_tiles[n].len() < 3) as u8;
        println!("{n}\t{tot:.0}\t{brick:.0}\t{wood:.0}\t{sheep:.0}\t{wheat:.0}\t{ore:.0}\t{exp:.3}\t{n_res}\t{road_pair:.0}\t{city_pair:.0}\t{settle_set:.0}\t{scarce:.3}\t{max_share:.3}\t{port_any}\t{port_match}\t{port31}\t{red}\t{dup_num}\t{}\t{best2:.0}\t{top2sum:.0}\t{open2}\t{compl2:.0}\t{coast}", map.adjacent_tiles[n].len());
    }
    // First-placement policies.
    let rules = GameConfig {
        map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
        dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(1),
        catan_map: Some(map.clone()), record_history: false, fixed_seats: true, ..Default::default()
    };
    let game = Game::new(&COLORS[..2], &rules);
    for spec in &args[2..] {
        let (label, path) = spec.split_once('=').expect("label=net");
        let net = load_net(path).expect("net");
        let net = net.opening.as_deref().unwrap_or(&net);
        let obs_space = ObservationSpace::for_network(&map, net.obs_size);
        let act_space = ActionSpace::new(&map);
        let mut obs = vec![0.0f32; net.obs_size];
        let (mut scratch, mut logits) = (Scratch::default(), Vec::new());
        let s = &game.state;
        obs_space.encode_into(s, s.current_color(), &mut obs);
        net.forward(&obs, &mut scratch, &mut logits);
        let mut items: Vec<(u8, f32)> = game.playable_actions.iter().filter_map(|a| match a.kind {
            ActionKind::BuildSettlement(n) => act_space.encode(s, a).map(|i| (n, logits[i])),
            _ => None,
        }).collect();
        let m = items.iter().map(|x| x.1).fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = items.iter().map(|x| (x.1 - m).exp()).sum();
        for it in items.iter_mut() {
            it.1 = (it.1 - m).exp() / z;
        }
        items.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!("#policy {label} {}", items.iter().map(|(n, p)| format!("{n}:{p:.4}")).collect::<Vec<_>>().join(" "));
    }
}
