//! Opening-placement dataset for one board: the board's geometry (tiles,
//! land nodes with their tiles, neighbours and port) so a script can compute
//! any spot feature, then each given network's settlement policy (network
//! only) for the four initial placements of a two-player game. Placements
//! 2-4 are conditioned on an opening line: the placements and roads so far
//! are the top moves of one "conditioning" network (every listed network
//! conditions one line), so the file holds the reply of every network to
//! every network's first pick, that network's own second pick, and the
//! first seat's second pick. A "main.ctnn@opening.ctnn" key places with its
//! opening network.
//!
//!   cargo run --release --example node_features2 -- <board seed> [label=net.ctnn ...]
//! Output lines:
//!   #board <seed> board_pips <wood,brick,sheep,wheat,ore>
//!   #tile <index> <WOOD|BRICK|SHEEP|WHEAT|ORE|DESERT> <number or 0>
//!   #node <id> tiles=<i;i;i> adj=<n;n;n> adjall=<n;n;n> port=<resource|3:1|->   (adj: land edges only)
//!   #policy <label> node:prob ...                   first seat, empty board
//!   #line <cond> s1 r1a-r1b s2 r2a-r2b s3 r3a-r3b s4 r4a-r4b   cond's own top-move line
//!   #policy2 <cond> <label> node:prob ...           second seat's first pick after cond's s1+r1
//!   #policy3 <cond> <label> node:prob ...           second seat's second pick after cond's s2+r2
//!   #policy4 <cond> <label> node:prob ...           first seat's second pick after cond's s3+r3
use catanatron::encoding::{ActionSpace, ObservationSpace};
use catanatron::map::{build_map, MapType, NumberPlacement};
use catanatron::mcts::load_net;
use catanatron::nn::{PolicyValueNet, Scratch};
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::*;
use std::sync::Arc;

struct Net {
    label: String,
    net: Arc<PolicyValueNet>,
    obs_space: ObservationSpace,
    obs: Vec<f32>,
    scratch: Scratch,
    logits: Vec<f32>,
}

impl Net {
    /// Logits over the playable actions (None for unencodable ones).
    fn logits(&mut self, game: &Game, act_space: &ActionSpace) -> Vec<Option<f32>> {
        let s = &game.state;
        self.obs_space.encode_into(s, s.current_color(), &mut self.obs);
        let net = self.net.opening.as_deref().unwrap_or(&self.net);
        net.forward(&self.obs, &mut self.scratch, &mut self.logits);
        game.playable_actions.iter().map(|a| act_space.encode(s, a).map(|i| self.logits[i])).collect()
    }
    /// Softmax over the settlement moves, sorted by probability.
    fn settlement_policy(&mut self, game: &Game, act_space: &ActionSpace) -> Vec<(u8, f32)> {
        let lg = self.logits(game, act_space);
        let mut items: Vec<(u8, f32)> = game.playable_actions.iter().zip(lg).filter_map(|(a, l)| match (a.kind, l) {
            (ActionKind::BuildSettlement(n), Some(l)) => Some((n, l)),
            _ => None,
        }).collect();
        let m = items.iter().map(|x| x.1).fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = items.iter().map(|x| (x.1 - m).exp()).sum();
        for it in items.iter_mut() {
            it.1 = (it.1 - m).exp() / z;
        }
        items.sort_by(|a, b| b.1.total_cmp(&a.1));
        items
    }
    /// The network's top move among the playable actions.
    fn top(&mut self, game: &Game, act_space: &ActionSpace) -> Action {
        let lg = self.logits(game, act_space);
        let i = (0..lg.len()).filter(|&i| lg[i].is_some()).max_by(|&i, &j| lg[i].unwrap().total_cmp(&lg[j].unwrap())).expect("a move");
        game.playable_actions[i]
    }
}

fn fmt_policy(items: &[(u8, f32)]) -> String {
    items.iter().map(|(n, p)| format!("{n}:{p:.4}")).collect::<Vec<_>>().join(" ")
}

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
    println!("#board {board} board_pips {}", board_pips.iter().map(|v| format!("{v:.0}")).collect::<Vec<_>>().join(","));
    for &ti in &map.land_tiles {
        let t = &map.tiles[ti];
        println!("#tile {ti} {} {}", t.resource().map_or("DESERT", |r| r.name()), t.number().unwrap_or(0));
    }
    for n in 0..map.num_nodes {
        if !map.land_nodes[n] {
            continue;
        }
        let tiles: Vec<String> = map.adjacent_tiles[n].iter().map(|t| t.to_string()).collect();
        // Neighbours over land edges only (where roads may go), and over all edges (water nodes included).
        let adj: Vec<String> = map.node_edges[n].iter().filter(|&&(_, e)| map.land_edge[e]).map(|&(m, _)| m.to_string()).collect();
        let adjall: Vec<String> = map.node_edges[n].iter().map(|&(m, _)| m.to_string()).collect();
        let port = map.port_nodes.iter().find(|(_, nodes)| nodes.contains(&(n as u8))).map_or("-".to_string(), |(r, _)| r.map_or("3:1".to_string(), |r| r.name().to_string()));
        println!("#node {n} tiles={} adj={} adjall={} port={port}", tiles.join(";"), adj.join(";"), adjall.join(";"));
    }
    let rules = GameConfig {
        map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
        dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(1),
        catan_map: Some(map.clone()), record_history: false, fixed_seats: true, ..Default::default()
    };
    let act_space = ActionSpace::new(&map);
    let mut nets: Vec<Net> = args[2..].iter().map(|spec| {
        let (label, path) = spec.split_once('=').expect("label=net");
        let net = load_net(path).expect("net");
        let obs_size = net.opening.as_deref().unwrap_or(&net).obs_size;
        Net { label: label.to_string(), obs_space: ObservationSpace::for_network(&map, obs_size), obs: vec![0.0f32; obs_size], scratch: Scratch::default(), logits: Vec::new(), net }
    }).collect();
    // First seat, empty board.
    let game = Game::new(&COLORS[..2], &rules);
    for i in 0..nets.len() {
        let pol = nets[i].settlement_policy(&game, &act_space);
        println!("#policy {} {}", nets[i].label, fmt_policy(&pol));
    }
    // Conditioned placements 2-4: the conditioning network plays its top line.
    for c in 0..nets.len() {
        let mut game = Game::new(&COLORS[..2], &rules);
        let mut line = Vec::new();
        for placement in 1..=4 {
            if placement > 1 {
                for i in 0..nets.len() {
                    let pol = nets[i].settlement_policy(&game, &act_space);
                    println!("#policy{placement} {} {} {}", nets[c].label, nets[i].label, fmt_policy(&pol));
                }
            }
            let settle = nets[c].top(&game, &act_space);
            let ActionKind::BuildSettlement(n) = settle.kind else { panic!("expected a settlement, got {settle:?}") };
            game.execute(settle).expect("legal");
            let road = nets[c].top(&game, &act_space);
            let ActionKind::BuildRoad((a, b)) = road.kind else { panic!("expected a road, got {road:?}") };
            game.execute(road).expect("legal");
            line.push(format!("{n} {a}-{b}"));
        }
        println!("#line {} {}", nets[c].label, line.join(" "));
    }
}
