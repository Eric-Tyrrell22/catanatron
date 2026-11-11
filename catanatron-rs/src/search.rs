//! Ports of the Python heuristic bots: the hand-tuned value function
//! (`ValueFunctionPlayer`, key `F`) and expectiminimax AlphaBeta search over
//! it (`AlphaBetaPlayer`, key `AB`). They are the baselines a learned bot
//! has to beat.

use crate::game::Game;
use crate::map::number_probability;
use crate::players::Player;
use crate::state::State;
use crate::types::*;

/// Weights of the value function (Python `DEFAULT_WEIGHTS`).
#[derive(Clone, Copy, Debug)]
pub struct Weights {
    pub public_vps: f64,
    pub production: f64,
    pub enemy_production: f64,
    pub num_tiles: f64,
    pub reachable_production_0: f64,
    pub reachable_production_1: f64,
    pub buildable_nodes: f64,
    pub longest_road: f64,
    pub hand_synergy: f64,
    pub hand_resources: f64,
    pub discard_penalty: f64,
    pub hand_devs: f64,
    pub army_size: f64,
}

pub const DEFAULT_WEIGHTS: Weights = Weights {
    public_vps: 3e14,
    production: 1e8,
    enemy_production: -1e8,
    num_tiles: 1.0,
    reachable_production_0: 0.0,
    reachable_production_1: 1e4,
    buildable_nodes: 1e3,
    longest_road: 10.0,
    hand_synergy: 1e2,
    hand_resources: 1.0,
    discard_penalty: -5.0,
    hand_devs: 10.0,
    army_size: 10.1,
};

/// Each resource type produced counts like this many production points.
const TRANSLATE_VARIETY: f64 = 4.0;
const PROBA_POINT: f64 = 2.778 / 100.0;

/// Expected cards per roll of each resource for `color`, skipping the
/// robber's tile (Python "EFFECTIVE_" production).
pub fn effective_production(s: &State, color: Color) -> [f64; 5] {
    let p = s.player(color);
    let mut prod = [0.0; 5];
    let map = &s.board.map;
    let mut add = |node: NodeId, weight: f64| {
        for &ti in &map.adjacent_tiles[node as usize] {
            let t = &map.tiles[ti];
            if t.coordinate == s.board.robber_coordinate {
                continue;
            }
            if let (Some(r), Some(n)) = (t.resource(), t.number()) {
                prod[r.idx()] += weight * number_probability(n);
            }
        }
    };
    for &n in &p.settlements {
        add(n, 1.0);
    }
    for &n in &p.cities {
        add(n, 2.0);
    }
    prod
}

pub fn value_production(prod: &[f64; 5], include_variety: bool) -> f64 {
    let sum: f64 = prod.iter().sum();
    if !include_variety {
        return sum;
    }
    let variety = prod.iter().filter(|&&x| x != 0.0).count() as f64;
    sum + variety * TRANSLATE_VARIETY * PROBA_POINT
}

/// Production (ignoring the robber) of nodes `color` owns or could still
/// settle, reachable with 0 and with 1 more road. Mirrors Python's
/// `reachability_features(levels=1)`, summed over resources.
pub fn reachable_production(s: &State, color: Color) -> (f64, f64) {
    let b = &s.board;
    let map = &b.map;
    let mut owned_or_buildable: Bits = *b.owned_bits(color);
    for n in 0..map.num_nodes {
        if b.buildable[n] {
            bit_set(&mut owned_or_buildable, n);
        }
    }
    let zero = b.network_bits(color);

    let mut level1 = zero;
    for n in bits_iter(zero) {
        if b.is_enemy_node(n as NodeId, color) {
            continue;
        }
        for &(nb, ei) in &map.node_edges[n] {
            let enemy_road = matches!(b.roads[ei], Some(c) if c != color);
            if !enemy_road {
                bit_set(&mut level1, nb as usize);
            }
        }
    }

    let sum = |nodes: Bits| -> f64 {
        bits_iter(bits_and(&nodes, &owned_or_buildable))
            .map(|n| map.node_production[n].iter().sum::<f64>())
            .sum()
    };
    (sum(zero), sum(level1))
}

/// Python `base_fn`: heuristic value of the position for `me`.
pub fn base_value(s: &State, me: Color, w: &Weights) -> f64 {
    let p = s.player(me);
    let production = value_production(&effective_production(s, me), true);
    // Python only considers one enemy: the next player in seat order.
    let enemy = s.colors[(s.seat(me) + 1) % s.colors.len()];
    let enemy_production = value_production(&effective_production(s, enemy), false);

    let (reach0, reach1) = reachable_production(s, me);

    let hand = &p.resources;
    let (wood, brick, sheep, wheat, ore) = (hand[0] as f64, hand[1] as f64, hand[2] as f64, hand[3] as f64, hand[4] as f64);
    let distance_to_city = ((2.0 - wheat).max(0.0) + (3.0 - ore).max(0.0)) / 5.0;
    let distance_to_settlement =
        ((1.0 - wheat).max(0.0) + (1.0 - sheep).max(0.0) + (1.0 - brick).max(0.0) + (1.0 - wood).max(0.0)) / 4.0;
    let hand_synergy = (2.0 - distance_to_city - distance_to_settlement) / 2.0;

    let num_in_hand = p.num_resources() as f64;
    let discard_penalty = if num_in_hand > 7.0 { w.discard_penalty } else { 0.0 };

    let map = &s.board.map;
    let mut tiles: Vec<usize> = p
        .settlements
        .iter()
        .chain(p.cities.iter())
        .flat_map(|&n| map.adjacent_tiles[n as usize].iter().copied())
        .collect();
    tiles.sort_unstable();
    tiles.dedup();
    let num_tiles = tiles.len() as f64;

    let num_buildable_nodes = bits_iter(s.board.network_bits(me))
        .filter(|&n| s.board.buildable[n])
        .count() as f64;
    let longest_road_factor = if num_buildable_nodes == 0.0 { w.longest_road } else { 0.1 };

    p.victory_points as f64 * w.public_vps
        + production * w.production
        + enemy_production * w.enemy_production
        + reach0 * w.reachable_production_0
        + reach1 * w.reachable_production_1
        + hand_synergy * w.hand_synergy
        + num_buildable_nodes * w.buildable_nodes
        + num_tiles * w.num_tiles
        + num_in_hand * w.hand_resources
        + discard_penalty
        + p.longest_road_length as f64 * longest_road_factor
        + p.num_dev_cards() as f64 * w.hand_devs
        + p.played_dev_cards[DevCard::Knight.idx()] as f64 * w.army_size
}

/// Greedy one-step lookahead over `base_value` (Python `F`).
pub struct ValueFunctionPlayer {
    pub color: Color,
    pub weights: Weights,
}

impl ValueFunctionPlayer {
    pub fn new(color: Color) -> Self {
        ValueFunctionPlayer { color, weights: DEFAULT_WEIGHTS }
    }
}

impl Player for ValueFunctionPlayer {
    fn color(&self) -> Color {
        self.color
    }

    fn decide(&mut self, game: &Game, actions: &[Action]) -> Action {
        if actions.len() == 1 {
            return actions[0];
        }
        let mut best = (f64::NEG_INFINITY, actions[0]);
        for &a in actions {
            let mut copy = game.copy_without_history();
            if copy.execute_unchecked(a, None).is_err() {
                continue;
            }
            let v = base_value(&copy.state, self.color, &self.weights);
            if v > best.0 {
                best = (v, a);
            }
        }
        best.1
    }
}

/// Possible outcomes of `action` with their probabilities (Python
/// `execute_spectrum`). Outcomes the player can't rule out but that turn
/// out impossible (e.g. stealing a card the victim lacks) leave the state
/// unchanged, as in Python.
pub fn expand_outcomes(game: &Game, action: Action) -> Vec<(Game, f64)> {
    let apply = |replay: Option<ActionResult>| {
        let mut g = game.copy_without_history();
        if g.execute_unchecked(action, replay).is_err() {
            g = game.copy_without_history();
        }
        g
    };
    match action.kind {
        ActionKind::Roll => (2u8..=12)
            .map(|roll| {
                let dice = ActionResult::Dice(roll / 2, roll.div_ceil(2));
                (apply(Some(dice)), number_probability(roll))
            })
            .collect(),
        ActionKind::BuyDevelopmentCard => {
            // Unseen cards from the buyer's view: the deck plus opponents'
            // unplayed cards.
            let s = &game.state;
            let mut counts = [0u32; 5];
            for &c in s.dev_deck.iter() {
                counts[c.idx()] += 1;
            }
            for p in s.players.iter().filter(|p| p.color != action.color) {
                for d in DEV_CARDS {
                    counts[d.idx()] += p.dev_cards[d.idx()];
                }
            }
            let total: u32 = counts.iter().sum();
            DEV_CARDS
                .iter()
                .filter(|d| counts[d.idx()] > 0)
                .map(|&d| (apply(Some(ActionResult::DevCard(d))), counts[d.idx()] as f64 / total as f64))
                .collect()
        }
        ActionKind::MoveRobber { victim: Some(v), .. } if game.state.player(v).num_resources() > 0 => RESOURCES
            .iter()
            .map(|&r| (apply(Some(ActionResult::Stolen(Some(r)))), 0.2))
            .collect(),
        _ => vec![(apply(None), 1.0)],
    }
}

/// Expectiminimax with alpha-beta cutoffs over `base_value` (Python `AB`).
/// `depth` counts decisions, not turns.
pub struct AlphaBetaPlayer {
    pub color: Color,
    pub depth: u32,
    pub weights: Weights,
}

impl AlphaBetaPlayer {
    pub fn new(color: Color, depth: u32) -> Self {
        AlphaBetaPlayer { color, depth, weights: DEFAULT_WEIGHTS }
    }

    fn alphabeta(&self, game: &Game, depth: u32, mut alpha: f64, mut beta: f64) -> (Option<Action>, f64) {
        if depth == 0 || game.winning_color().is_some() {
            return (None, base_value(&game.state, self.color, &self.weights));
        }
        let maximizing = game.state.current_color() == self.color;
        let mut best_action = None;
        let mut best_value = if maximizing { f64::NEG_INFINITY } else { f64::INFINITY };
        for &action in &game.playable_actions {
            let expected: f64 = expand_outcomes(game, action)
                .iter()
                .map(|(g, p)| p * self.alphabeta(g, depth - 1, alpha, beta).1)
                .sum();
            if maximizing {
                if expected > best_value {
                    best_value = expected;
                    best_action = Some(action);
                }
                alpha = alpha.max(best_value);
            } else {
                if expected < best_value {
                    best_value = expected;
                    best_action = Some(action);
                }
                beta = beta.min(best_value);
            }
            if alpha >= beta {
                break;
            }
        }
        (best_action, best_value)
    }
}

impl Player for AlphaBetaPlayer {
    fn color(&self) -> Color {
        self.color
    }

    fn decide(&mut self, game: &Game, actions: &[Action]) -> Action {
        if actions.len() == 1 {
            return actions[0];
        }
        self.alphabeta(game, self.depth, f64::NEG_INFINITY, f64::INFINITY)
            .0
            .unwrap_or(actions[0])
    }
}
