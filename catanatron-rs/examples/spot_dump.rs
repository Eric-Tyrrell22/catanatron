//! Raw spot structure of boards, for the spot-net experiments (runs/spot_fit.py):
//! for each board seed read from stdin (one per line) prints
//!   #board <seed> <land node count>
//!   <node> <tiles: res/num;res/num;res/num> <port> <distance-1 nodes a,b,c> <distance-2 land nodes k,l,...>
//! res = 0..4 (brick, wood, sheep, wheat, ore) or 5 (desert); num = 0 for none.
//! port = - (none), 3 (3:1) or the resource index of a 2:1 port.
//!
//!   cargo run --release --example spot_dump < seeds.txt > spots.txt
use catanatron::map::{build_map, MapType, NumberPlacement};
use catanatron::rng::Rng;
use std::collections::BTreeSet;
use std::io::{self, BufRead, Write};

fn main() {
    let stdin = io::stdin();
    let out = io::stdout();
    let mut out = io::BufWriter::new(out.lock());
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let seed: u64 = match line.trim().parse() { Ok(s) => s, Err(_) => continue };
        let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(seed));
        let land: Vec<usize> = (0..map.num_nodes).filter(|&n| map.land_nodes[n]).collect();
        writeln!(out, "#board {seed} {}", land.len()).unwrap();
        for &n in &land {
            let tiles: Vec<String> = map.adjacent_tiles[n].iter().map(|&ti| {
                let t = &map.tiles[ti];
                format!("{}/{}", t.resource().map_or(5, |r| r.idx()), t.number().unwrap_or(0))
            }).collect();
            let port = map.port_nodes.iter().find(|(_, nodes)| nodes.contains(&(n as u8)))
                .map_or("-".to_string(), |(r, _)| r.map_or("3".to_string(), |r| r.idx().to_string()));
            let one: Vec<usize> = map.node_edges[n].iter().map(|&(m, _)| m as usize).filter(|&m| map.land_nodes[m]).collect();
            let mut two = BTreeSet::new();
            for &m in &one {
                for &(k, _) in &map.node_edges[m] {
                    let k = k as usize;
                    if k != n && !one.contains(&k) && map.land_nodes[k] {
                        two.insert(k);
                    }
                }
            }
            writeln!(out, "{n} {} {port} {} {}", tiles.join(";"),
                     one.iter().map(|m| m.to_string()).collect::<Vec<_>>().join(","),
                     two.iter().map(|k| k.to_string()).collect::<Vec<_>>().join(",")).unwrap();
        }
    }
}
