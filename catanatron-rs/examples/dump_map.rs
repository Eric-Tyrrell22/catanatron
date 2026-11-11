//! Prints the tournament map in the same format as the Python dump used to
//! cross-check node ids.
use catanatron::map::{tournament_map, TileKind};

fn main() {
    let m = tournament_map();
    let res = |r: Option<catanatron::Resource>| r.map_or("None".to_string(), |r| r.name().to_string());
    let dir = |d: catanatron::map::Direction| format!("{:?}", d).to_uppercase();
    for t in &m.tiles {
        let nodes: Vec<String> = t.nodes.iter().map(|n| n.to_string()).collect();
        let c = t.coordinate;
        match &t.kind {
            TileKind::Land { resource, number, .. } => println!(
                "L ({}, {}, {}) {} {} [{}]",
                c.0, c.1, c.2, res(*resource),
                number.map_or("None".into(), |n| n.to_string()), nodes.join(", ")
            ),
            TileKind::Port { resource, direction, .. } => println!(
                "P ({}, {}, {}) {} {} [{}]",
                c.0, c.1, c.2, res(*resource), dir(*direction), nodes.join(", ")
            ),
            TileKind::Water => {}
        }
    }
    let mut ports: Vec<(String, Vec<u8>)> = m.port_nodes.iter().map(|(r, n)| {
        let mut n = n.clone(); n.sort(); (res(*r), n)
    }).collect();
    ports.sort();
    let p: Vec<String> = ports.iter().map(|(r, n)| format!("('{}', {:?})", r, n)).collect();
    println!("ports [{}]", p.join(", "));
    println!("graph {} {} {}", m.edges.len(), m.num_nodes, m.land_nodes.iter().filter(|&&x| x).count());
}
