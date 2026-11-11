//! Map templates, tile placement, and the static lookup tables derived from
//! a placed map. Node and edge ids are assigned in the same order as the
//! Python implementation, so ids line up between the two.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use crate::rng::Rng;
use crate::types::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    East,
    SouthEast,
    SouthWest,
    West,
    NorthWest,
    NorthEast,
}

pub const DIRECTIONS: [Direction; 6] = [
    Direction::East,
    Direction::SouthEast,
    Direction::SouthWest,
    Direction::West,
    Direction::NorthWest,
    Direction::NorthEast,
];

impl Direction {
    pub fn unit(self) -> Coordinate {
        match self {
            Direction::NorthEast => (1, 0, -1),
            Direction::SouthWest => (-1, 0, 1),
            Direction::NorthWest => (0, 1, -1),
            Direction::SouthEast => (0, -1, 1),
            Direction::East => (1, -1, 0),
            Direction::West => (-1, 1, 0),
        }
    }
}

pub fn add(a: Coordinate, b: Coordinate) -> Coordinate {
    (a.0 + b.0, a.1 + b.1, a.2 + b.2)
}

// Node references within a tile (indices into Tile::nodes).
pub const N: usize = 0;
pub const NE: usize = 1;
pub const SE: usize = 2;
pub const S: usize = 3;
pub const SW: usize = 4;
pub const NW: usize = 5;

// Edge references within a tile (indices into Tile::edges).
pub const E_EAST: usize = 0;
pub const E_SOUTHEAST: usize = 1;
pub const E_SOUTHWEST: usize = 2;
pub const E_WEST: usize = 3;
pub const E_NORTHWEST: usize = 4;
pub const E_NORTHEAST: usize = 5;

const EDGE_NODEREFS: [(usize, usize); 6] = [(NE, SE), (SE, S), (S, SW), (SW, NW), (NW, N), (N, NE)];

pub fn port_noderefs(d: Direction) -> (usize, usize) {
    match d {
        Direction::West => (NW, SW),
        Direction::NorthWest => (N, NW),
        Direction::NorthEast => (NE, N),
        Direction::East => (SE, NE),
        Direction::SouthEast => (S, SE),
        Direction::SouthWest => (SW, S),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum TileKind {
    Land {
        id: usize,
        resource: Option<Resource>, // None = desert
        number: Option<u8>,
    },
    Port {
        id: usize,
        resource: Option<Resource>, // None = 3:1
        direction: Direction,
    },
    Water,
}

#[derive(Clone, Debug)]
pub struct Tile {
    pub coordinate: Coordinate,
    pub kind: TileKind,
    pub nodes: [NodeId; 6],
    pub edges: [EdgeId; 6],
}

impl Tile {
    pub fn resource(&self) -> Option<Resource> {
        match self.kind {
            TileKind::Land { resource, .. } => resource,
            _ => None,
        }
    }
    pub fn number(&self) -> Option<u8> {
        match self.kind {
            TileKind::Land { number, .. } => number,
            _ => None,
        }
    }
    pub fn is_land(&self) -> bool {
        matches!(self.kind, TileKind::Land { .. })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Topo {
    Land,
    Water,
    Port(Direction),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapType {
    Base,
    Mini,
    Tournament,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumberPlacement {
    /// The official token order along a spiral from the template's fixed
    /// starting corner, skipping the desert.
    OfficialSpiral,
    /// The same, from one of the 6 outer corners chosen at random (the
    /// physical game's rule and OpenHex's). Never puts 6s and 8s, or two
    /// equal numbers, next to each other.
    SpiralRandomCorner,
    Random,
}

/// `c` rotated 60 degrees about the center, in the direction the spiral's
/// corners are numbered (OpenHex's ring order: (2,-2,0) -> (2,0,-2) -> ...).
fn rotate60(c: Coordinate) -> Coordinate {
    (-c.1, -c.2, -c.0)
}

/// The spiral's starting corner `k` (0..6) for `template`.
fn spiral_corner(template: &MapTemplate, k: usize) -> Coordinate {
    (0..k).fold(template.spiral_start, |c, _| rotate60(c))
}

#[derive(Clone, Debug)]
pub struct MapTemplate {
    pub numbers: Vec<u8>,
    pub port_resources: Vec<Option<Resource>>,
    pub tile_resources: Vec<Option<Resource>>,
    pub topology: Vec<(Coordinate, Topo)>,
    /// Outer land tile where the official number spiral starts.
    pub spiral_start: Coordinate,
}

/// Numbers by the letter on their back (A, B, C, ...).
pub const BASE_NUMBERS_IN_SPIRAL_ORDER: [u8; 18] =
    [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11];

use Resource::*;

fn inner_land() -> Vec<(Coordinate, Topo)> {
    vec![
        ((0, 0, 0), Topo::Land),
        ((1, -1, 0), Topo::Land),
        ((0, -1, 1), Topo::Land),
        ((-1, 0, 1), Topo::Land),
        ((-1, 1, 0), Topo::Land),
        ((0, 1, -1), Topo::Land),
        ((1, 0, -1), Topo::Land),
    ]
}

const SECOND_LAYER: [Coordinate; 12] = [
    (2, -2, 0),
    (1, -2, 1),
    (0, -2, 2),
    (-1, -1, 2),
    (-2, 0, 2),
    (-2, 1, 1),
    (-2, 2, 0),
    (-1, 2, -1),
    (0, 2, -2),
    (1, 1, -2),
    (2, 0, -2),
    (2, -1, -1),
];

/// Small 7-tile map, no ports.
pub fn mini_template() -> MapTemplate {
    let mut topology = inner_land();
    topology.extend(SECOND_LAYER.iter().map(|&c| (c, Topo::Water)));
    MapTemplate {
        numbers: vec![3, 4, 5, 6, 8, 9, 10],
        port_resources: vec![],
        tile_resources: vec![Some(Wood), None, Some(Brick), Some(Sheep), Some(Wheat), Some(Wheat), Some(Ore)],
        topology,
        spiral_start: (1, -1, 0),
    }
}

/// Standard 4-player map.
pub fn base_template() -> MapTemplate {
    use Direction as D;
    let mut topology = inner_land();
    topology.extend(SECOND_LAYER.iter().map(|&c| (c, Topo::Land)));
    topology.extend([
        ((3, -3, 0), Topo::Port(D::West)),
        ((2, -3, 1), Topo::Water),
        ((1, -3, 2), Topo::Port(D::NorthWest)),
        ((0, -3, 3), Topo::Water),
        ((-1, -2, 3), Topo::Port(D::NorthWest)),
        ((-2, -1, 3), Topo::Water),
        ((-3, 0, 3), Topo::Port(D::NorthEast)),
        ((-3, 1, 2), Topo::Water),
        ((-3, 2, 1), Topo::Port(D::East)),
        ((-3, 3, 0), Topo::Water),
        ((-2, 3, -1), Topo::Port(D::East)),
        ((-1, 3, -2), Topo::Water),
        ((0, 3, -3), Topo::Port(D::SouthEast)),
        ((1, 2, -3), Topo::Water),
        ((2, 1, -3), Topo::Port(D::SouthWest)),
        ((3, 0, -3), Topo::Water),
        ((3, -1, -2), Topo::Port(D::SouthWest)),
        ((3, -2, -1), Topo::Water),
    ]);
    let mut tile_resources = Vec::new();
    for (r, n) in [(Some(Wood), 4), (Some(Brick), 3), (Some(Sheep), 4), (Some(Wheat), 4), (Some(Ore), 3), (None, 1)] {
        tile_resources.extend(std::iter::repeat_n(r, n));
    }
    MapTemplate {
        numbers: vec![2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12],
        port_resources: vec![Some(Wood), Some(Brick), Some(Sheep), Some(Wheat), Some(Ore), None, None, None, None],
        tile_resources,
        topology,
        spiral_start: (2, -2, 0),
    }
}

/// Fill in node/edge ids for a tile at `coordinate`, reusing ids of
/// already-placed neighbors.
fn nodes_and_edges(
    tiles: &[Tile],
    by_coord: &HashMap<Coordinate, usize>,
    coordinate: Coordinate,
    node_autoinc: &mut u16,
) -> ([NodeId; 6], [EdgeId; 6]) {
    let mut nodes: [Option<NodeId>; 6] = [None; 6];
    let mut edges: [Option<EdgeId>; 6] = [None; 6];

    for d in DIRECTIONS {
        let Some(&i) = by_coord.get(&add(coordinate, d.unit())) else {
            continue;
        };
        let nb = &tiles[i];
        match d {
            Direction::East => {
                nodes[NE] = Some(nb.nodes[NW]);
                nodes[SE] = Some(nb.nodes[SW]);
                edges[E_EAST] = Some(nb.edges[E_WEST]);
            }
            Direction::SouthEast => {
                nodes[S] = Some(nb.nodes[NW]);
                nodes[SE] = Some(nb.nodes[N]);
                edges[E_SOUTHEAST] = Some(nb.edges[E_NORTHWEST]);
            }
            Direction::SouthWest => {
                nodes[S] = Some(nb.nodes[NE]);
                nodes[SW] = Some(nb.nodes[N]);
                edges[E_SOUTHWEST] = Some(nb.edges[E_NORTHEAST]);
            }
            Direction::West => {
                nodes[NW] = Some(nb.nodes[NE]);
                nodes[SW] = Some(nb.nodes[SE]);
                edges[E_WEST] = Some(nb.edges[E_EAST]);
            }
            Direction::NorthWest => {
                nodes[N] = Some(nb.nodes[SE]);
                nodes[NW] = Some(nb.nodes[S]);
                edges[E_NORTHWEST] = Some(nb.edges[E_SOUTHEAST]);
            }
            Direction::NorthEast => {
                nodes[N] = Some(nb.nodes[SW]);
                nodes[NE] = Some(nb.nodes[S]);
                edges[E_NORTHEAST] = Some(nb.edges[E_SOUTHWEST]);
            }
        }
    }

    let mut out_nodes = [0 as NodeId; 6];
    for (i, n) in nodes.iter().enumerate() {
        out_nodes[i] = match n {
            Some(id) => *id,
            None => {
                let id = *node_autoinc as NodeId;
                *node_autoinc += 1;
                id
            }
        };
    }
    let mut out_edges = [(0, 0); 6];
    for (i, e) in edges.iter().enumerate() {
        out_edges[i] = match e {
            Some(id) => *id,
            None => {
                let (a, b) = EDGE_NODEREFS[i];
                edge(out_nodes[a], out_nodes[b])
            }
        };
    }
    (out_nodes, out_edges)
}

fn is_land(tiles: &[Tile], by_coord: &HashMap<Coordinate, usize>, c: Coordinate) -> bool {
    by_coord.get(&c).is_some_and(|&i| tiles[i].is_land())
}

/// Land coordinates in coast-following spiral order, from an outer-edge
/// start toward the center.
pub fn spiral_land_coordinates(
    tiles: &[Tile],
    by_coord: &HashMap<Coordinate, usize>,
    start: Coordinate,
) -> Result<Vec<Coordinate>, String> {
    if !is_land(tiles, by_coord, start) {
        return Err("start must be a land tile".into());
    }
    if DIRECTIONS.iter().all(|d| is_land(tiles, by_coord, add(start, d.unit()))) {
        return Err("start must be on the outer edge of the land mass".into());
    }
    let mut directions = DIRECTIONS;
    directions.reverse();

    let mut dir_idx = None;
    for i in 0..6 {
        let prev = directions[(i + 5) % 6];
        if is_land(tiles, by_coord, add(start, directions[i].unit()))
            && !is_land(tiles, by_coord, add(start, prev.unit()))
        {
            dir_idx = Some(i);
            break;
        }
    }
    let mut dir_idx = dir_idx.ok_or("could not determine an initial coast-following direction")?;

    let total = tiles.iter().filter(|t| t.is_land()).count();
    let mut visited: Vec<Coordinate> = Vec::with_capacity(total);
    let mut coord = start;
    let mut stuck = 0;
    while visited.len() < total {
        if !visited.contains(&coord) {
            visited.push(coord);
            stuck = 0;
        }
        let next = add(coord, directions[dir_idx].unit());
        if is_land(tiles, by_coord, next) && !visited.contains(&next) {
            coord = next;
            continue;
        }
        dir_idx = (dir_idx + 1) % 6;
        stuck += 1;
        if stuck > 6 {
            return Err("land mass is not spiral-walkable".into());
        }
    }
    Ok(visited)
}

/// Places tiles per the template. Any of the `*_param` lists, if given, are
/// used as-is (consumed from the end) instead of being shuffled.
pub fn initialize_tiles(
    template: &MapTemplate,
    numbers_param: Option<Vec<u8>>,
    ports_param: Option<Vec<Option<Resource>>>,
    resources_param: Option<Vec<Option<Resource>>>,
    placement: NumberPlacement,
    rng: &mut Rng,
) -> Result<Vec<Tile>, String> {
    if placement != NumberPlacement::Random && numbers_param.is_some() {
        return Err("official_spiral number placement does not support custom numbers".into());
    }
    let mut ports = ports_param.unwrap_or_else(|| {
        let mut v = template.port_resources.clone();
        rng.shuffle(&mut v);
        v
    });
    let mut resources = resources_param.unwrap_or_else(|| {
        let mut v = template.tile_resources.clone();
        rng.shuffle(&mut v);
        v
    });
    let mut numbers = numbers_param.unwrap_or_else(|| {
        let mut v = template.numbers.clone();
        rng.shuffle(&mut v);
        v
    });

    let mut tiles: Vec<Tile> = Vec::with_capacity(template.topology.len());
    let mut by_coord = HashMap::new();
    let mut node_autoinc: u16 = 0;
    let (mut tile_id, mut port_id) = (0, 0);
    for &(coordinate, topo) in &template.topology {
        let (nodes, edges) = nodes_and_edges(&tiles, &by_coord, coordinate, &mut node_autoinc);
        let kind = match topo {
            Topo::Port(direction) => {
                let resource = ports.pop().ok_or("not enough port resources")?;
                port_id += 1;
                TileKind::Port { id: port_id - 1, resource, direction }
            }
            Topo::Land => {
                let resource = resources.pop().ok_or("not enough tile resources")?;
                let number = match resource {
                    Some(_) => Some(numbers.pop().ok_or("not enough numbers")?),
                    None => None,
                };
                tile_id += 1;
                TileKind::Land { id: tile_id - 1, resource, number }
            }
            Topo::Water => TileKind::Water,
        };
        by_coord.insert(coordinate, tiles.len());
        tiles.push(Tile { coordinate, kind, nodes, edges });
    }
    if node_autoinc > NodeId::MAX as u16 + 1 {
        return Err("too many nodes for NodeId".into());
    }

    let start = match placement {
        NumberPlacement::OfficialSpiral => Some(template.spiral_start),
        NumberPlacement::SpiralRandomCorner => Some(spiral_corner(template, rng.below(6))),
        NumberPlacement::Random => None,
    };
    if let Some(start) = start {
        let order = spiral_land_coordinates(&tiles, &by_coord, start)?;
        let mut i = 0;
        for c in order {
            if let TileKind::Land { resource: Some(_), number, .. } = &mut tiles[by_coord[&c]].kind {
                *number = Some(BASE_NUMBERS_IN_SPIRAL_ORDER[i]);
                i += 1;
            }
        }
    }
    Ok(tiles)
}

/// A placed map plus fast-lookup tables. Immutable once built; shared
/// between board copies through an `Arc`.
#[derive(Clone, Debug)]
pub struct CatanMap {
    pub tiles: Vec<Tile>,
    pub tile_by_coord: HashMap<Coordinate, usize>,
    /// Indices into `tiles` of land tiles, in placement order.
    pub land_tiles: Vec<usize>,
    pub num_nodes: usize,
    pub land_nodes: Vec<bool>,
    /// Port resource (None = 3:1) => node ids that grant it.
    pub port_nodes: Vec<(Option<Resource>, Vec<NodeId>)>,
    /// node id => indices into `tiles` of adjacent land tiles.
    pub adjacent_tiles: Vec<Vec<usize>>,
    /// node id => expected production per roll, by resource.
    pub node_production: Vec<[f64; 5]>,
    /// All distinct edges, sorted.
    pub edges: Vec<EdgeId>,
    pub edge_index: HashMap<EdgeId, usize>,
    /// Edges whose both endpoints are land nodes (where roads may go).
    pub land_edge: Vec<bool>,
    /// node id => (neighbor node, edge index).
    pub node_edges: Vec<Vec<(NodeId, usize)>>,
    /// Where the robber starts (first desert tile).
    pub desert_coordinate: Coordinate,
    /// Dice number => indices into `tiles` of land tiles with that number.
    pub tiles_by_number: [Vec<usize>; 13],
    /// Tile index => bitset of its six nodes.
    pub tile_node_bits: Vec<Bits>,
}

pub fn number_probability(number: u8) -> f64 {
    let ways = 6 - (7 - number as i32).abs();
    if ways <= 0 { 0.0 } else { ways as f64 / 36.0 }
}

impl CatanMap {
    pub fn from_template(template: &MapTemplate, placement: NumberPlacement, rng: &mut Rng) -> CatanMap {
        let tiles = initialize_tiles(template, None, None, None, placement, rng)
            .expect("builtin templates are valid");
        CatanMap::from_tiles(tiles)
    }

    pub fn from_tiles(tiles: Vec<Tile>) -> CatanMap {
        let tile_by_coord: HashMap<_, _> = tiles.iter().enumerate().map(|(i, t)| (t.coordinate, i)).collect();
        let land_tiles: Vec<usize> = (0..tiles.len()).filter(|&i| tiles[i].is_land()).collect();
        let num_nodes = tiles.iter().flat_map(|t| t.nodes.iter()).map(|&n| n as usize + 1).max().unwrap_or(0);

        let mut land_nodes = vec![false; num_nodes];
        let mut adjacent_tiles = vec![Vec::new(); num_nodes];
        for &i in &land_tiles {
            for &n in &tiles[i].nodes {
                land_nodes[n as usize] = true;
                adjacent_tiles[n as usize].push(i);
            }
        }

        let mut edges: Vec<EdgeId> = tiles.iter().flat_map(|t| t.edges.iter().copied()).collect();
        edges.sort_unstable();
        edges.dedup();
        let edge_index: HashMap<_, _> = edges.iter().enumerate().map(|(i, &e)| (e, i)).collect();
        let land_edge = edges.iter().map(|&(a, b)| land_nodes[a as usize] && land_nodes[b as usize]).collect();
        let mut node_edges = vec![Vec::new(); num_nodes];
        for (i, &(a, b)) in edges.iter().enumerate() {
            node_edges[a as usize].push((b, i));
            node_edges[b as usize].push((a, i));
        }

        let tile_node_bits = tiles
            .iter()
            .map(|t| {
                let mut b = [0; 4];
                for &n in &t.nodes {
                    bit_set(&mut b, n as usize);
                }
                b
            })
            .collect();

        let mut map = CatanMap {
            tile_node_bits,
            tiles_by_number: Default::default(),
            tiles,
            tile_by_coord,
            land_tiles,
            num_nodes,
            land_nodes,
            port_nodes: Vec::new(),
            adjacent_tiles,
            node_production: Vec::new(),
            edges,
            edge_index,
            land_edge,
            node_edges,
            desert_coordinate: (0, 0, 0),
        };
        map.refresh_resource_tables();
        map
    }

    /// This map's topology with the given layout: each land tile's resource
    /// (None = desert) and number, and each port's resource (None = 3:1),
    /// ports identified by their two nodes. Every land tile and every port
    /// must be given exactly once.
    pub fn with_layout(
        &self,
        land: &[(Coordinate, Option<Resource>, Option<u8>)],
        ports: &[([NodeId; 2], Option<Resource>)],
    ) -> Result<CatanMap, String> {
        let mut m = self.clone();
        let mut set = vec![false; m.tiles.len()];
        for &(c, r, n) in land {
            let &i = m.tile_by_coord.get(&c).ok_or_else(|| format!("no tile at {c:?}"))?;
            match &mut m.tiles[i].kind {
                TileKind::Land { resource, number, .. } => {
                    if set[i] {
                        return Err(format!("land tile {c:?} given twice"));
                    }
                    if r.is_some() != n.is_some() {
                        return Err(format!("tile {c:?}: a resource needs a number and the desert none"));
                    }
                    *resource = r;
                    *number = n;
                    set[i] = true;
                }
                _ => return Err(format!("{c:?} is not a land tile")),
            }
        }
        if let Some(&i) = m.land_tiles.iter().find(|&&i| !set[i]) {
            return Err(format!("land tile {:?} not given", m.tiles[i].coordinate));
        }
        let mut used = vec![false; ports.len()];
        for t in m.tiles.iter_mut() {
            if let TileKind::Port { resource, direction, .. } = &mut t.kind {
                let (a, b) = port_noderefs(*direction);
                let pair = [t.nodes[a], t.nodes[b]];
                let k = ports
                    .iter()
                    .position(|(p, _)| (p[0] == pair[0] && p[1] == pair[1]) || (p[0] == pair[1] && p[1] == pair[0]))
                    .ok_or_else(|| format!("no port given for nodes {pair:?}"))?;
                if used[k] {
                    return Err(format!("port {pair:?} given twice"));
                }
                used[k] = true;
                *resource = ports[k].1;
            }
        }
        if let Some(k) = used.iter().position(|&u| !u) {
            return Err(format!("port at nodes {:?} matches no port of this map", ports[k].0));
        }
        m.refresh_resource_tables();
        Ok(m)
    }

    /// Recomputes the tables that depend on tile resources and numbers
    /// (everything else depends only on the topology).
    fn refresh_resource_tables(&mut self) {
        let tiles = &self.tiles;
        let mut port_nodes: Vec<(Option<Resource>, Vec<NodeId>)> = Vec::new();
        for t in tiles {
            if let TileKind::Port { resource, direction, .. } = t.kind {
                let (a, b) = port_noderefs(direction);
                let pos = match port_nodes.iter().position(|(r, _)| *r == resource) {
                    Some(p) => p,
                    None => {
                        port_nodes.push((resource, Vec::new()));
                        port_nodes.len() - 1
                    }
                };
                for n in [t.nodes[a], t.nodes[b]] {
                    if !port_nodes[pos].1.contains(&n) {
                        port_nodes[pos].1.push(n);
                    }
                }
            }
        }

        let mut node_production = vec![[0.0; 5]; self.num_nodes];
        for (n, adj) in self.adjacent_tiles.iter().enumerate() {
            for &i in adj {
                if let (Some(r), Some(num)) = (tiles[i].resource(), tiles[i].number()) {
                    node_production[n][r.idx()] += number_probability(num);
                }
            }
        }

        let mut tiles_by_number: [Vec<usize>; 13] = Default::default();
        for &i in &self.land_tiles {
            if let Some(n) = tiles[i].number() {
                tiles_by_number[n as usize].push(i);
            }
        }

        let desert = self
            .land_tile_iter()
            .find(|t| t.resource().is_none())
            .map(|t| t.coordinate)
            .unwrap_or_else(|| self.tiles[self.land_tiles[0]].coordinate);
        self.desert_coordinate = desert;
        self.port_nodes = port_nodes;
        self.node_production = node_production;
        self.tiles_by_number = tiles_by_number;
    }

    pub fn tile_at(&self, c: Coordinate) -> Option<&Tile> {
        // Linear scan: maps have at most a few dozen tiles, and this beats
        // hashing the coordinate.
        self.tiles.iter().find(|t| t.coordinate == c)
    }

    /// Index into `edges` of the edge between `a` and `b`, if any.
    #[inline]
    pub fn edge_idx(&self, a: NodeId, b: NodeId) -> Option<usize> {
        self.node_edges
            .get(a as usize)?
            .iter()
            .find(|&&(nb, _)| nb == b)
            .map(|&(_, i)| i)
    }

    pub fn land_tile_iter(&self) -> impl Iterator<Item = &Tile> {
        self.land_tiles.iter().map(move |&i| &self.tiles[i])
    }
}

/// The fixed tournament map, built once.
pub fn tournament_map() -> Arc<CatanMap> {
    static MAP: OnceLock<Arc<CatanMap>> = OnceLock::new();
    MAP.get_or_init(|| {
        let numbers = vec![10, 8, 3, 6, 2, 5, 10, 8, 4, 11, 12, 9, 5, 4, 9, 11, 3, 6];
        let ports = vec![None, Some(Sheep), None, Some(Ore), Some(Wheat), None, Some(Wood), Some(Brick), None];
        let resources = vec![
            None, Some(Wood), Some(Sheep), Some(Sheep), Some(Wood), Some(Wheat), Some(Wood), Some(Wheat),
            Some(Brick), Some(Sheep), Some(Brick), Some(Sheep), Some(Wheat), Some(Wheat), Some(Ore),
            Some(Brick), Some(Ore), Some(Wood), Some(Ore), None,
        ];
        let mut rng = Rng::new(0); // unused: every list is given
        let tiles = initialize_tiles(
            &base_template(),
            Some(numbers),
            Some(ports),
            Some(resources),
            NumberPlacement::Random,
            &mut rng,
        )
        .expect("tournament map is valid");
        Arc::new(CatanMap::from_tiles(tiles))
    })
    .clone()
}

/// A template's topology, built once: a placed map (with arbitrary
/// resources) plus the official spiral order as tile indices.
struct Layout {
    template: MapTemplate,
    map: CatanMap,
    /// Tile indices in spiral order from each corner (`spiral_corner`);
    /// corner 0 is the template's `spiral_start`.
    spirals: Vec<Vec<usize>>,
}

fn layout(map_type: MapType) -> &'static Layout {
    fn make(template: MapTemplate) -> Layout {
        let mut rng = Rng::new(0);
        let map = CatanMap::from_template(&template, NumberPlacement::Random, &mut rng);
        let spirals = (0..6)
            .map(|k| {
                spiral_land_coordinates(&map.tiles, &map.tile_by_coord, spiral_corner(&template, k))
                    .expect("builtin templates are spiral-walkable from every corner")
                    .iter()
                    .map(|c| map.tile_by_coord[c])
                    .collect()
            })
            .collect();
        Layout { template, map, spirals }
    }
    static BASE: OnceLock<Layout> = OnceLock::new();
    static MINI: OnceLock<Layout> = OnceLock::new();
    match map_type {
        MapType::Mini => MINI.get_or_init(|| make(mini_template())),
        _ => BASE.get_or_init(|| make(base_template())),
    }
}

/// Same result as `CatanMap::from_template` for the same RNG state, but
/// reuses the cached topology instead of rebuilding it.
fn randomize_layout(l: &Layout, placement: NumberPlacement, rng: &mut Rng) -> CatanMap {
    // Shuffle in the same order as `initialize_tiles`.
    let mut ports = l.template.port_resources.clone();
    rng.shuffle(&mut ports);
    let mut resources = l.template.tile_resources.clone();
    rng.shuffle(&mut resources);
    let mut numbers = l.template.numbers.clone();
    rng.shuffle(&mut numbers);

    let mut map = l.map.clone();
    for tile in map.tiles.iter_mut() {
        match &mut tile.kind {
            TileKind::Port { resource, .. } => *resource = ports.pop().expect("enough ports"),
            TileKind::Land { resource, number, .. } => {
                *resource = resources.pop().expect("enough resources");
                *number = match resource {
                    Some(_) => Some(numbers.pop().expect("enough numbers")),
                    None => None,
                };
            }
            TileKind::Water => {}
        }
    }
    let spiral = match placement {
        NumberPlacement::OfficialSpiral => Some(&l.spirals[0]),
        NumberPlacement::SpiralRandomCorner => Some(&l.spirals[rng.below(6)]),
        NumberPlacement::Random => None,
    };
    if let Some(spiral) = spiral {
        let mut i = 0;
        for &t in spiral {
            if let TileKind::Land { resource: Some(_), number, .. } = &mut map.tiles[t].kind {
                *number = Some(BASE_NUMBERS_IN_SPIRAL_ORDER[i]);
                i += 1;
            }
        }
    }
    map.refresh_resource_tables();
    map
}

/// Shares of games played on selected boards (the rest are standard). The
/// boards are real ones: a selected game redraws its standard board until it
/// has the style's property (see `BoardStyle::matches`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BoardMix {
    pub scarce_ows: f32,
    pub port_start: f32,
}

/// A property a selected board must have.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoardStyle {
    /// Ore, wheat and sheep tiles carry at most `SCARCE_OWS_PIPS` pips
    /// together (about the scarcest 10% of real boards; median 35 of 58).
    ScarceOws,
    /// A 2:1 port has a node touching a tile of its own resource on a 6 or 8,
    /// with at least `PORT_START_PIPS` pips on the node in all.
    PortStart,
}

pub const SCARCE_OWS_PIPS: u32 = 32;
pub const PORT_START_PIPS: u32 = 8;

fn pips(number: Option<u8>) -> u32 {
    number.map_or(0, |n| 6 - (7 - n as i32).unsigned_abs())
}

impl BoardStyle {
    pub fn matches(&self, map: &CatanMap) -> bool {
        match self {
            BoardStyle::ScarceOws => {
                let ows: u32 = map
                    .land_tiles
                    .iter()
                    .filter(|&&i| matches!(map.tiles[i].resource(), Some(Resource::Ore | Resource::Wheat | Resource::Sheep)))
                    .map(|&i| pips(map.tiles[i].number()))
                    .sum();
                ows <= SCARCE_OWS_PIPS
            }
            BoardStyle::PortStart => map.port_nodes.iter().any(|(r, nodes)| {
                r.is_some_and(|r| {
                    nodes.iter().any(|&n| {
                        let adj = &map.adjacent_tiles[n as usize];
                        adj.iter().any(|&i| map.tiles[i].resource() == Some(r) && pips(map.tiles[i].number()) == 5)
                            && adj.iter().map(|&i| pips(map.tiles[i].number())).sum::<u32>() >= PORT_START_PIPS
                    })
                })
            }),
        }
    }
}

/// A board's group for group experts: the resource with the fewest pips for
/// its number of tiles (pips per tile, lowest first; ties go to the earlier
/// resource in `RESOURCES` order).
pub fn scarcest_resource(map: &CatanMap) -> Resource {
    let (mut pips_by, mut tiles_by) = ([0u32; 5], [0u32; 5]);
    for &i in &map.land_tiles {
        if let Some(r) = map.tiles[i].resource() {
            pips_by[r.idx()] += pips(map.tiles[i].number());
            tiles_by[r.idx()] += 1;
        }
    }
    // Compare pips_a / tiles_a with pips_b / tiles_b without dividing.
    *crate::types::RESOURCES
        .iter()
        .min_by(|a, b| (pips_by[a.idx()] * tiles_by[b.idx()]).cmp(&(pips_by[b.idx()] * tiles_by[a.idx()])))
        .unwrap()
}

impl BoardMix {
    pub fn is_empty(&self) -> bool {
        self.scarce_ows <= 0.0 && self.port_start <= 0.0
    }

    /// The style for one game, if any.
    pub fn pick(&self, rng: &mut Rng) -> Option<BoardStyle> {
        let u = (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        if u < self.scarce_ows {
            Some(BoardStyle::ScarceOws)
        } else if u < self.scarce_ows + self.port_start {
            Some(BoardStyle::PortStart)
        } else {
            None
        }
    }
}

impl CatanMap {
    /// Land layout (coordinate, resource, number) and ports (two nodes,
    /// resource), as `with_layout` takes them.
    #[allow(clippy::type_complexity)]
    pub fn layout_spec(&self) -> (Vec<(Coordinate, Option<Resource>, Option<u8>)>, Vec<([NodeId; 2], Option<Resource>)>) {
        let land = self.land_tiles.iter().map(|&i| (self.tiles[i].coordinate, self.tiles[i].resource(), self.tiles[i].number())).collect();
        let ports = self
            .tiles
            .iter()
            .filter_map(|t| match t.kind {
                TileKind::Port { resource, direction, .. } => {
                    let (a, b) = port_noderefs(direction);
                    Some(([t.nodes[a], t.nodes[b]], resource))
                }
                _ => None,
            })
            .collect();
        (land, ports)
    }
}

pub fn build_map(map_type: MapType, placement: NumberPlacement, rng: &mut Rng) -> Arc<CatanMap> {
    match map_type {
        MapType::Tournament => tournament_map(),
        _ => Arc::new(randomize_layout(layout(map_type), placement, rng)),
    }
}
