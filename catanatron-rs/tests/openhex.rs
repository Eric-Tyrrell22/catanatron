use catanatron::map::{build_map, MapType, NumberPlacement};
use catanatron::openhex::{canonical_edge, hex_vertices, turn_hex, turn_vertex, Geometry, Vertex};
use catanatron::rng::Rng;
use catanatron::types::*;

/// OpenHex's 9 standard port positions (its maps.py STANDARD_PORT_POSITIONS).
const OPENHEX_PORTS: [(Vertex, Vertex); 9] = [
    ((2, -2, 1), (2, -2, 2)),
    ((3, -1, -1), (3, -1, 0)),
    ((3, 0, -2), (3, 1, -2)),
    ((1, 2, -2), (2, 2, -2)),
    ((-1, 3, -1), (0, 3, -1)),
    ((-2, 3, 0), (-2, 3, 1)),
    ((-2, 1, 2), (-2, 2, 2)),
    ((-1, -1, 3), (-1, 0, 3)),
    ((0, -2, 3), (1, -2, 3)),
];

fn adjacent(a: Vertex, b: Vertex) -> bool {
    let (lo, hi) = canonical_edge(a, b);
    lo.0 + lo.1 + lo.2 == 1
        && hi.0 + hi.1 + hi.2 == 2
        && [hi.0 - lo.0, hi.1 - lo.1, hi.2 - lo.2].iter().all(|&d| d == 0 || d == 1)
}

#[test]
fn openhex_geometry_matches_topology() {
    let map = build_map(MapType::Base, NumberPlacement::OfficialSpiral, &mut Rng::new(0));
    let g = Geometry::new(&map).unwrap();
    assert_eq!(g.node_vertex.len(), 54);
    assert_eq!(g.edge_vertices.len(), 72);
    for (&(a, b), &(va, vb)) in &g.edge_vertices {
        assert!(adjacent(va, vb), "edge {a}-{b} maps to non-adjacent {va:?} {vb:?}");
        assert_eq!(g.edge(vb, va).unwrap(), (a, b));
    }
    // Every land tile's six corners are exactly OpenHex's six corners of it.
    for &t in &map.land_tiles {
        let tile = &map.tiles[t];
        let mut ours: Vec<Vertex> = tile.nodes.iter().map(|n| g.node_vertex[n]).collect();
        let mut theirs = hex_vertices(tile.coordinate).to_vec();
        ours.sort();
        theirs.sort();
        assert_eq!(ours, theirs);
    }
    // Turning a hex turns its corners.
    for &t in &map.land_tiles {
        let h = map.tiles[t].coordinate;
        let mut a: Vec<Vertex> = hex_vertices(h).iter().map(|&v| turn_vertex(v)).collect();
        let mut b = hex_vertices(turn_hex(h)).to_vec();
        a.sort();
        b.sort();
        assert_eq!(a, b);
    }
    // OpenHex's port positions, turned into our frame, are our port slots,
    // and a layout built from them keeps the given resources.
    let resources = [Some(Resource::Wood), None, Some(Resource::Brick), None, Some(Resource::Sheep), None, Some(Resource::Wheat), None, Some(Resource::Ore)];
    let ports: Vec<([NodeId; 2], Option<Resource>)> = OPENHEX_PORTS
        .iter()
        .zip(resources)
        .map(|(&(a, b), r)| ([g.node(turn_vertex(a)).unwrap(), g.node(turn_vertex(b)).unwrap()], r))
        .collect();
    let land: Vec<_> = map.land_tiles.iter().map(|&t| (map.tiles[t].coordinate, map.tiles[t].resource(), map.tiles[t].number())).collect();
    let rebuilt = map.with_layout(&land, &ports).unwrap();
    for (nodes, r) in &ports {
        let (_, port_nodes) = rebuilt.port_nodes.iter().find(|(pr, _)| pr == r).unwrap();
        assert!(nodes.iter().all(|n| port_nodes.contains(n)), "port {r:?} at {nodes:?}");
    }
}
