//! What a settlement spot is made of: for a board seed, every land node's
//! adjacent tiles (resource, number, pips), total pips, distinct resources,
//! port, and the best neighbouring spot one and two roads away, sorted by
//! pips. Listed node ids are marked with `*`.
//!
//!   cargo run --release --example node_info -- <board seed> [node id ...]
use catanatron::map::{build_map, number_probability, MapType, NumberPlacement};
use catanatron::rng::Rng;
use std::collections::HashSet;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let board: u64 = args[1].parse().expect("board seed");
    let marked: HashSet<usize> = args[2..].iter().map(|s| s.parse().expect("node id")).collect();
    let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(board));
    let pips = |n: u8| (6 - (7 - n as i32).abs()).max(0);
    let node_pips = |node: usize| -> i32 {
        map.adjacent_tiles[node].iter().filter_map(|&ti| map.tiles[ti].number()).map(pips).sum()
    };
    let port_of = |node: usize| -> Option<String> {
        map.port_nodes.iter().find(|(_, nodes)| nodes.contains(&(node as u8)))
            .map(|(r, _)| r.map_or("3:1".to_string(), |r| format!("2:1 {r:?}")))
    };
    let mut rows: Vec<(usize, i32)> = (0..map.num_nodes).filter(|&n| map.land_nodes[n]).map(|n| (n, node_pips(n))).collect();
    rows.sort_by_key(|&(n, p)| (-p, n));
    println!("board {board}: land nodes by pips (tiles as resource/number/pips; exp = expected cards per roll)");
    let shown: Vec<(usize, i32)> = rows.iter().enumerate().filter(|(i, (n, _))| *i < 16 || marked.contains(n)).map(|(_, r)| *r).collect();
    for (n, p) in &shown {
        let (n, p) = (*n, node_pips(*n));
        let mut tiles: Vec<String> = Vec::new();
        let mut kinds = HashSet::new();
        let mut exp = 0.0;
        for &ti in &map.adjacent_tiles[n] {
            let t = &map.tiles[ti];
            match (t.resource(), t.number()) {
                (Some(r), Some(num)) => {
                    kinds.insert(format!("{r:?}"));
                    exp += number_probability(num);
                    tiles.push(format!("{r:?}/{num}/{}", pips(num)));
                }
                _ => tiles.push("desert".to_string()),
            }
        }
        // Best spots one and two roads away (distinct nodes not adjacent to this one, as the distance rule requires).
        let one: Vec<usize> = map.node_edges[n].iter().map(|&(m, _)| m as usize).collect();
        let two: HashSet<usize> = one.iter().flat_map(|&m| map.node_edges[m].iter().map(|&(k, _)| k as usize)).filter(|k| *k != n && !one.contains(k)).collect();
        let best2 = two.iter().filter(|&&k| map.land_nodes[k]).map(|&k| (node_pips(k), k)).max().unwrap_or((0, 0));
        println!("{}{n:3}: {p:2} pips  exp {exp:.2}  {} resources  [{}]  port {}  best spot 2 roads away: node {} ({} pips)",
                 if marked.contains(&n) { "*" } else { " " }, kinds.len(), tiles.join(", "),
                 port_of(n).unwrap_or_else(|| "-".to_string()), best2.1, best2.0);
    }
}
