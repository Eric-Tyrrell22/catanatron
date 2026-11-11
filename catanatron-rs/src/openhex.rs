//! Geometry shared with OpenHex (github.com/bovard/open-hex, SPEC.md
//! section 3): hexes use the same cube coordinates as ours; a vertex is a
//! cube triple with x+y+z = 1 or 2, and an edge is its two vertices (the
//! x+y+z = 1 one first).
//!
//! Our tile corners (`Tile::nodes`, N..NW in our orientation) relate to
//! OpenHex's corner labels (N, NE, SE, S, SW, NW in theirs) by a mirror and a
//! two-step rotation: our corner i is their corner (2 - i) mod 6. That is the
//! only relabeling under which every shared corner and every edge agree
//! (checked by `openhex_geometry_matches_topology`).

use std::collections::HashMap;

use crate::map::CatanMap;
use crate::types::{Coordinate, EdgeId, NodeId};

pub type Vertex = (i8, i8, i8);

/// OpenHex's standard board is ours turned 180 degrees: its 9 harbors sit at
/// our harbor slots only after the turn. A mirror therefore turns every
/// OpenHex hex and vertex into our frame on the way in (and back on the way
/// out), so its boards are exactly the ones our networks trained on. The
/// turn is its own inverse.
pub fn turn_hex(h: Coordinate) -> Coordinate {
    (-h.0, -h.1, -h.2)
}

/// `turn_hex` for vertices: corner offsets map onto corner offsets.
pub fn turn_vertex(v: Vertex) -> Vertex {
    (1 - v.0, 1 - v.1, 1 - v.2)
}

/// OpenHex's corners of hex `h`, in its label order N, NE, SE, S, SW, NW.
pub fn hex_vertices(h: Coordinate) -> [Vertex; 6] {
    let (x, y, z) = h;
    [(x + 1, y, z + 1), (x + 1, y, z), (x + 1, y + 1, z), (x, y + 1, z), (x, y + 1, z + 1), (x, y, z + 1)]
}

/// Our corner index -> OpenHex corner index.
fn their_corner(ours: usize) -> usize {
    (8 - ours) % 6
}

/// Two-way maps between our land nodes and edges and OpenHex vertices.
#[derive(Clone, Debug)]
pub struct Geometry {
    pub node_vertex: HashMap<NodeId, Vertex>,
    pub vertex_node: HashMap<Vertex, NodeId>,
    pub edge_vertices: HashMap<EdgeId, (Vertex, Vertex)>,
    pub vertices_edge: HashMap<(Vertex, Vertex), EdgeId>,
}

/// Canonical OpenHex edge: the x+y+z = 1 vertex first.
pub fn canonical_edge(a: Vertex, b: Vertex) -> (Vertex, Vertex) {
    if a.0 + a.1 + a.2 == 1 { (a, b) } else { (b, a) }
}

impl Geometry {
    pub fn new(map: &CatanMap) -> Result<Geometry, String> {
        let mut node_vertex = HashMap::new();
        for &t in &map.land_tiles {
            let tile = &map.tiles[t];
            let corners = hex_vertices(tile.coordinate);
            for (i, &n) in tile.nodes.iter().enumerate() {
                let v = corners[their_corner(i)];
                if let Some(old) = node_vertex.insert(n, v) {
                    if old != v {
                        return Err(format!("node {n} maps to both {old:?} and {v:?}"));
                    }
                }
            }
        }
        let vertex_node: HashMap<Vertex, NodeId> = node_vertex.iter().map(|(&n, &v)| (v, n)).collect();
        if vertex_node.len() != node_vertex.len() {
            return Err("two nodes share a vertex".into());
        }
        let mut edge_vertices = HashMap::new();
        for (i, &(a, b)) in map.edges.iter().enumerate() {
            if !map.land_edge[i] {
                continue;
            }
            let (va, vb) = (node_vertex[&a], node_vertex[&b]);
            edge_vertices.insert((a, b), canonical_edge(va, vb));
        }
        let vertices_edge = edge_vertices.iter().map(|(&e, &p)| (p, e)).collect();
        Ok(Geometry { node_vertex, vertex_node, edge_vertices, vertices_edge })
    }

    pub fn node(&self, v: Vertex) -> Result<NodeId, String> {
        self.vertex_node.get(&v).copied().ok_or_else(|| format!("no node at vertex {v:?}"))
    }

    pub fn edge(&self, a: Vertex, b: Vertex) -> Result<EdgeId, String> {
        self.vertices_edge.get(&canonical_edge(a, b)).copied().ok_or_else(|| format!("no edge between {a:?} and {b:?}"))
    }
}
