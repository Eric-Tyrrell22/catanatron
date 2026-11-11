//! Board state: buildings, roads, robber, and longest-road tracking.
//!
//! Unlike the Python version, this does not cache connected components.
//! A player's road network is derived directly: a node belongs to it if the
//! player has a building there, or has a road touching it and no opponent
//! has a building there. Longest road is recomputed per player on change.

use std::sync::Arc;

use crate::map::CatanMap;
use crate::types::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildingType {
    Settlement,
    City,
}

#[derive(Clone, Debug)]
pub struct Board {
    pub map: Arc<CatanMap>,
    /// node id => building, if any.
    pub buildings: [Option<(Color, BuildingType)>; MAX_NODES],
    /// edge index (see `CatanMap::edges`) => road owner, if any.
    pub roads: [Option<Color>; MAX_EDGES],
    /// Land nodes still open for settlements (distance rule applied).
    pub buildable: [bool; MAX_NODES],
    pub robber_coordinate: Coordinate,
    /// Longest road length per color, indexed by `Color::idx`.
    pub road_lengths: [u8; 4],
    /// Holder of the Longest Road card.
    pub road_color: Option<Color>,
    /// Length the holder needs to be beaten by.
    pub road_length: u8,
    /// Per color (`Color::idx`): bitset of nodes touched by its roads.
    road_nodes: [Bits; 4],
    /// Per color: bitset of nodes with its buildings.
    owned: [Bits; 4],
    /// Per color: bitmask of port access. Bit `Resource::idx` for 2:1
    /// ports, bit 5 for 3:1.
    port_mask: [u8; 4],
}

/// Capacity of the fixed-size board arrays: the base map's 96 nodes and
/// 132 edges (water tiles included). Smaller maps use a prefix.
pub const MAX_NODES: usize = 96;
pub const MAX_EDGES: usize = 132;

/// Most roads one player can own, which bounds any trail's length.
const MAX_TRAIL: usize = 15;

impl Board {
    pub fn new(map: Arc<CatanMap>) -> Board {
        assert!(
            map.num_nodes <= MAX_NODES && map.edges.len() <= MAX_EDGES,
            "map too large for fixed-size board ({} nodes, {} edges)",
            map.num_nodes,
            map.edges.len()
        );
        let mut buildable = [false; MAX_NODES];
        buildable[..map.num_nodes].copy_from_slice(&map.land_nodes);
        Board {
            buildings: [None; MAX_NODES],
            roads: [None; MAX_EDGES],
            buildable,
            robber_coordinate: map.desert_coordinate,
            road_lengths: [0; 4],
            road_color: None,
            road_length: 0,
            road_nodes: [[0; 4]; 4],
            owned: [[0; 4]; 4],
            port_mask: [0; 4],
            map,
        }
    }

    #[inline]
    pub fn node_color(&self, n: NodeId) -> Option<Color> {
        self.buildings[n as usize].map(|(c, _)| c)
    }

    #[inline]
    pub fn is_enemy_node(&self, n: NodeId, color: Color) -> bool {
        matches!(self.node_color(n), Some(c) if c != color)
    }

    pub fn edge_color(&self, e: EdgeId) -> Option<Color> {
        self.map.edge_idx(e.0, e.1).and_then(|i| self.roads[i])
    }

    /// Whether `n` is part of `color`'s network (can be built from).
    pub fn in_network(&self, color: Color, n: NodeId) -> bool {
        match self.node_color(n) {
            Some(c) => c == color,
            None => bit_get(&self.road_nodes[color.idx()], n as usize),
        }
    }

    /// Bitset of nodes with `color`'s buildings.
    #[inline]
    pub fn owned_bits(&self, color: Color) -> &Bits {
        &self.owned[color.idx()]
    }

    /// Bitset of `color`'s network: its buildings, plus nodes its roads touch
    /// that no opponent has built on.
    pub fn network_bits(&self, color: Color) -> Bits {
        let c = color.idx();
        let mut out = [0; 4];
        for w in 0..4 {
            let others = (0..4).filter(|&o| o != c).fold(0, |acc, o| acc | self.owned[o][w]);
            out[w] = self.owned[c][w] | (self.road_nodes[c][w] & !others);
        }
        out
    }

    /// Bitset of edge indices where `color` may build a road.
    pub fn buildable_edge_bits(&self, color: Color) -> Bits {
        let mut out = [0; 4];
        for n in bits_iter(self.network_bits(color)) {
            for &(_, ei) in &self.map.node_edges[n] {
                if self.map.land_edge[ei] && self.roads[ei].is_none() {
                    bit_set(&mut out, ei);
                }
            }
        }
        out
    }

    /// Sorted node ids where `color` may place a settlement.
    pub fn buildable_node_ids(&self, color: Color, initial_build_phase: bool) -> Vec<NodeId> {
        (0..self.map.num_nodes as u16)
            .map(|n| n as NodeId)
            .filter(|&n| self.buildable[n as usize])
            .filter(|&n| initial_build_phase || self.in_network(color, n))
            .collect()
    }

    #[inline]
    pub fn is_buildable_edge(&self, color: Color, i: usize) -> bool {
        let (a, b) = self.map.edges[i];
        self.map.land_edge[i]
            && self.roads[i].is_none()
            && (self.in_network(color, a) || self.in_network(color, b))
    }

    /// Whether `color` could place any road right now.
    pub fn has_buildable_edge(&self, color: Color) -> bool {
        bits_any(&self.buildable_edge_bits(color))
    }

    /// Sorted empty land edges touching `color`'s network.
    pub fn buildable_edges(&self, color: Color) -> Vec<EdgeId> {
        bits_iter(self.buildable_edge_bits(color)).map(|i| self.map.edges[i]).collect()
    }

    /// Places a settlement. Returns the longest-road holder before the build.
    pub fn build_settlement(
        &mut self,
        color: Color,
        n: NodeId,
        initial_build_phase: bool,
    ) -> Result<Option<Color>, String> {
        if (n as usize) >= self.map.num_nodes || self.buildings[n as usize].is_some() {
            return Err("Invalid Settlement Placement: a building exists there".into());
        }
        if !self.buildable[n as usize]
            || (!initial_build_phase && !self.in_network(color, n))
        {
            return Err("Invalid Settlement Placement: not connected and not initial-placement".into());
        }
        let previous = self.road_color;
        self.buildings[n as usize] = Some((color, BuildingType::Settlement));
        bit_set(&mut self.owned[color.idx()], n as usize);
        self.buildable[n as usize] = false;
        for (r, nodes) in &self.map.port_nodes {
            if nodes.contains(&n) {
                self.port_mask[color.idx()] |= 1 << r.map_or(5, |r| r.idx());
            }
        }
        for &(nb, _) in &self.map.node_edges[n as usize] {
            self.buildable[nb as usize] = false;
        }

        if !initial_build_phase {
            // An opponent with two roads through this node just got cut.
            let mut cut = false;
            for c in COLORS {
                if c == color {
                    continue;
                }
                let through = self.map.node_edges[n as usize]
                    .iter()
                    .filter(|&&(_, ei)| self.roads[ei] == Some(c))
                    .count();
                if through >= 2 {
                    self.road_lengths[c.idx()] = self.longest_road(c);
                    cut = true;
                }
            }
            if cut {
                self.reassign_longest_road_after_cut();
            }
        }
        Ok(previous)
    }

    /// Places a road. Returns the longest-road holder before the build.
    pub fn build_road(&mut self, color: Color, e: EdgeId) -> Result<Option<Color>, String> {
        let e = edge(e.0, e.1);
        let Some(i) = self.map.edge_idx(e.0, e.1) else {
            return Err("Invalid Road Placement: no such edge".into());
        };
        if !self.map.land_edge[i]
            || self.roads[i].is_some()
            || !(self.in_network(color, e.0) || self.in_network(color, e.1))
        {
            return Err("Invalid Road Placement".into());
        }
        let previous = self.road_color;
        self.roads[i] = Some(color);
        bit_set(&mut self.road_nodes[color.idx()], e.0 as usize);
        bit_set(&mut self.road_nodes[color.idx()], e.1 as usize);

        // A new road can only lengthen trails in its own component.
        let length = self.road_lengths[color.idx()].max(self.component_longest_road(color, e.0, &mut [0; 4]));
        self.road_lengths[color.idx()] = length;
        if length >= 5 && length > self.road_length {
            self.road_color = Some(color);
            self.road_length = length;
        }
        Ok(previous)
    }

    pub fn build_city(&mut self, color: Color, n: NodeId) -> Result<(), String> {
        match self.buildings.get(n as usize).copied().flatten() {
            Some((c, BuildingType::Settlement)) if c == color => {
                self.buildings[n as usize] = Some((color, BuildingType::City));
                Ok(())
            }
            _ => Err("Invalid City Placement: no player settlement there".into()),
        }
    }

    /// After a road is cut: the holder keeps the card if still at least tied
    /// for longest (and >= 5); otherwise it goes to a unique leader with
    /// >= 5, or to nobody.
    fn reassign_longest_road_after_cut(&mut self) {
        let max = *self.road_lengths.iter().max().unwrap();
        if let Some(h) = self.road_color {
            let len = self.road_lengths[h.idx()];
            if len >= 5 && len == max {
                self.road_length = len;
                return;
            }
        }
        let leaders: Vec<Color> = COLORS
            .iter()
            .copied()
            .filter(|c| self.road_lengths[c.idx()] == max)
            .collect();
        if max >= 5 && leaders.len() == 1 {
            self.road_color = Some(leaders[0]);
        } else {
            self.road_color = None;
        }
        self.road_length = max;
    }

    /// Longest trail over `color`'s roads. A trail may end at, but not pass
    /// through, an opponent's building.
    ///
    /// A longest trail can always be taken to start at a road end, a
    /// junction, or an opponent's building; only a component that is a bare
    /// loop has none of those, and then any node on it works.
    pub fn longest_road(&self, color: Color) -> u8 {
        let mut seen: Bits = [0; 4];
        let mut best = 0;
        for n in bits_iter(self.road_nodes[color.idx()]) {
            if !bit_get(&seen, n) {
                best = best.max(self.component_longest_road(color, n as NodeId, &mut seen));
            }
        }
        best
    }

    /// Longest trail in the component of `color`'s roads containing
    /// `start`. Marks the component's nodes in `seen`.
    fn component_longest_road(&self, color: Color, start: NodeId, seen: &mut Bits) -> u8 {
        let own_degree = |x: NodeId| {
            self.map.node_edges[x as usize]
                .iter()
                .filter(|&&(_, ei)| self.roads[ei] == Some(color))
                .count()
        };
        let mut stack = [0 as NodeId; 256];
        let mut component = [0 as NodeId; 256];
        let (mut sp, mut len) = (1, 0);
        stack[0] = start;
        bit_set(seen, start as usize);
        while sp > 0 {
            sp -= 1;
            let x = stack[sp];
            component[len] = x;
            len += 1;
            for &(nb, ei) in &self.map.node_edges[x as usize] {
                if self.roads[ei] == Some(color) && !bit_get(seen, nb as usize) {
                    bit_set(seen, nb as usize);
                    stack[sp] = nb;
                    sp += 1;
                }
            }
        }
        let mut used: Bits = [0; 4];
        let mut best = 0;
        let mut started = false;
        for &x in &component[..len] {
            if own_degree(x) != 2 || self.is_enemy_node(x, color) {
                best = best.max(self.longest_trail_from(x, color, &mut used));
                started = true;
            }
        }
        if !started {
            best = self.longest_trail_from(start, color, &mut used);
        }
        best
    }

    /// Longest trail of `color`'s roads starting at `start`, not passing
    /// through opponent buildings. Iterative DFS over an explicit stack;
    /// depth is bounded by the number of roads a player can own.
    fn longest_trail_from(&self, start: NodeId, color: Color, used: &mut Bits) -> u8 {
        const NO_EDGE: u8 = u8::MAX;
        // (node, next neighbor slot to try, edge taken to reach node)
        let mut stack = [(0 as NodeId, 0u8, NO_EDGE); MAX_TRAIL + 1];
        stack[0] = (start, 0, NO_EDGE);
        let mut sp = 1;
        let mut best = 0;
        while sp > 0 {
            let depth = sp - 1;
            let (node, next, _) = stack[depth];
            // A trail may end at an opponent's building but not continue.
            let blocked = depth > 0 && self.is_enemy_node(node, color);
            let edges = &self.map.node_edges[node as usize];
            let mut k = next as usize;
            if !blocked {
                while k < edges.len() {
                    let ei = edges[k].1;
                    if self.roads[ei] == Some(color) && !bit_get(used, ei) {
                        break;
                    }
                    k += 1;
                }
            } else {
                k = edges.len();
            }
            if k < edges.len() {
                let (nb, ei) = edges[k];
                stack[depth].1 = k as u8 + 1;
                bit_set(used, ei);
                stack[sp] = (nb, 0, ei as u8);
                sp += 1;
                best = best.max(sp as u8 - 1);
            } else {
                let (_, _, came_by) = stack[depth];
                if came_by != NO_EDGE {
                    bit_clear(used, came_by as usize);
                }
                sp -= 1;
            }
        }
        best
    }

    /// Bitmask of `color`'s port access: bit `Resource::idx` for 2:1,
    /// bit 5 for 3:1.
    #[inline]
    pub fn port_mask(&self, color: Color) -> u8 {
        self.port_mask[color.idx()]
    }

    /// Port resources (None = 3:1) available to `color`.
    pub fn player_port_resources(&self, color: Color) -> Vec<Option<Resource>> {
        let m = self.port_mask(color);
        let mut out: Vec<Option<Resource>> =
            RESOURCES.iter().filter(|r| m >> r.idx() & 1 == 1).map(|&r| Some(r)).collect();
        if m >> 5 & 1 == 1 {
            out.push(None);
        }
        out
    }
}
