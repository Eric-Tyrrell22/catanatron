//! Network-guided search (AlphaZero-style MCTS), plus a plain network
//! player.
//!
//! Catan-specific parts:
//! - Chance: dice, dev card draws and steals are sampled each time a
//!   simulation crosses them; each distinct outcome gets its own child, so
//!   visit counts follow the true odds.
//! - Hidden information: before searching, the searcher reshuffles what it
//!   can't know (dev deck order, which unplayed dev cards opponents hold)
//!   and reseeds the dice, so the search can't peek at the future.
//!   Opponents' resource hands are kept as-is (they are mostly public).
//! - Forced moves (one legal action) are passed through without a network
//!   evaluation.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use crate::encoding::{ActionSpace, ObservationSpace};
use crate::game::Game;
use crate::nn::{BatchScratch, PolicyValueNet, Scratch, MAX_BATCH};
use crate::players::Player;
use crate::rng::Rng;
use crate::types::*;

/// Networks kept loaded after their last player is gone, so a path used game
/// after game (an evaluation, a self-play pool) isn't re-read every game.
const KEEP_LOADED: usize = 32;

#[derive(Default)]
struct NetCache {
    by_path: HashMap<String, Weak<PolicyValueNet>>,
    /// Most recently loaded last.
    recent: VecDeque<Arc<PolicyValueNet>>,
}

/// Loads a network once per path and shares it (games create players often).
/// A network stays cached while some player uses it or it is among the last
/// KEEP_LOADED loaded, so cycling through many snapshot files doesn't
/// accumulate memory. Paths are assumed not to be rewritten with new weights.
pub fn load_net(path: &str) -> Result<Arc<PolicyValueNet>, String> {
    static CACHE: OnceLock<Mutex<NetCache>> = OnceLock::new();
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    if let Some(net) = cache.by_path.get(path).and_then(Weak::upgrade) {
        return Ok(net);
    }
    let net = Arc::new(PolicyValueNet::load(path)?);
    cache.recent.push_back(net.clone());
    if cache.recent.len() > KEEP_LOADED {
        cache.recent.pop_front();
    }
    cache.by_path.retain(|_, w| w.strong_count() > 0);
    cache.by_path.insert(path.to_string(), Arc::downgrade(&net));
    Ok(net)
}

/// Softmax of `logits` over `game.playable_actions` (unencodable moves get 0).
fn priors_from(act_space: &ActionSpace, game: &Game, logits: &[f32]) -> Vec<f32> {
    let s = &game.state;
    let priors: Vec<f32> = game
        .playable_actions
        .iter()
        .map(|a| act_space.encode(s, a).map_or(f32::NEG_INFINITY, |i| logits[i]))
        .collect();
    softmax(priors)
}

/// Softmax of per-move logits (-inf = 0; all -inf = uniform).
fn softmax(mut priors: Vec<f32>) -> Vec<f32> {
    let max = priors.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for p in priors.iter_mut() {
        *p = if max.is_finite() { (*p - max).exp() } else { 1.0 };
        sum += *p;
    }
    for p in priors.iter_mut() {
        *p /= sum;
    }
    priors
}

/// Runs the network on positions: legal-move priors and a value for the
/// player to move.
pub(crate) struct Evaluator {
    net: Arc<PolicyValueNet>,
    spaces: Option<(ObservationSpace, ActionSpace)>,
    obs_nz: Vec<(u32, f32)>,
    /// Action slot per playable action (`u32::MAX` = not encodable).
    slots: Vec<u32>,
    logits: Vec<f32>,
    scratch: Scratch,
    /// Per batch position: sparse observation, logits.
    batch_nz: Vec<Vec<(u32, f32)>>,
    batch_logits: Vec<Vec<f32>>,
    batch: BatchScratch,
    evals: u64,
}

impl Evaluator {
    pub(crate) fn new(net: Arc<PolicyValueNet>) -> Self {
        Evaluator {
            net,
            spaces: None,
            obs_nz: Vec::new(),
            slots: Vec::new(),
            logits: Vec::new(),
            scratch: Scratch::default(),
            batch_nz: vec![Vec::new(); MAX_BATCH],
            batch_logits: vec![Vec::new(); MAX_BATCH],
            batch: BatchScratch::default(),
            evals: 0,
        }
    }

    fn spaces(&mut self, game: &Game) -> &(ObservationSpace, ActionSpace) {
        let spaces = self.spaces.get_or_insert_with(|| {
            let map = &game.state.board.map;
            (ObservationSpace::for_network(map, self.net.obs_size), ActionSpace::new(map))
        });
        assert_eq!(spaces.0.len(), self.net.obs_size, "network was trained for a different map");
        spaces
    }

    /// (softmax priors over `game.playable_actions`, value for the mover).
    pub(crate) fn evaluate(&mut self, game: &Game) -> (Vec<f32>, f32) {
        self.spaces(game);
        let (obs_space, act_space) = self.spaces.as_ref().unwrap();
        let s = &game.state;
        let n = obs_space.encode_sparse_into(s, s.current_color(), &mut self.obs_nz);
        // Only the legal moves' logits are needed.
        self.slots.clear();
        self.slots.extend(game.playable_actions.iter().map(|a| act_space.encode(s, a).map_or(u32::MAX, |i| i as u32)));
        let value = self.net.forward_sparse_for(&self.obs_nz[..n], &self.slots, &mut self.scratch, &mut self.logits);
        self.evals += 1;
        (softmax(self.logits.clone()), value)
    }

    /// The network's observation of `game` from `me`'s seat (sparse).
    fn observe(&mut self, game: &Game, me: Color) -> Vec<(u32, f32)> {
        self.spaces(game);
        let mut nz = Vec::new();
        let n = self.spaces.as_ref().unwrap().0.encode_sparse_into(&game.state, me, &mut nz);
        nz.truncate(n);
        nz
    }

    /// Network evaluations so far.
    pub(crate) fn evaluations(&self) -> u64 {
        self.evals
    }

    /// `evaluate` for several positions, sharing network passes.
    pub(crate) fn evaluate_batch(&mut self, games: &[&Game]) -> Vec<(Vec<f32>, f32)> {
        self.spaces(games[0]);
        let (obs_space, act_space) = self.spaces.as_ref().unwrap();
        let mut out = Vec::with_capacity(games.len());
        let mut values = [0.0; MAX_BATCH];
        for chunk in games.chunks(MAX_BATCH) {
            let mut lens = [0; MAX_BATCH];
            for (b, g) in chunk.iter().enumerate() {
                lens[b] = obs_space.encode_sparse_into(&g.state, g.state.current_color(), &mut self.batch_nz[b]);
            }
            let xs: Vec<&[(u32, f32)]> = self.batch_nz[..chunk.len()].iter().zip(lens).map(|(nz, n)| &nz[..n]).collect();
            self.net.forward_batch(&xs, &mut self.batch, &mut self.batch_logits, &mut values);
            for (b, g) in chunk.iter().enumerate() {
                out.push((priors_from(act_space, g, &self.batch_logits[b]), values[b]));
            }
        }
        self.evals += games.len() as u64;
        out
    }
}

/// Plays from the network without search: its most likely legal move, or
/// (with `sampled`) a move drawn from its policy, as self-play opponents do.
pub struct NetPlayer {
    pub color: Color,
    eval: Evaluator,
    sampler: Option<Rng>,
}

impl NetPlayer {
    pub fn new(color: Color, net: Arc<PolicyValueNet>) -> Self {
        NetPlayer { color, eval: Evaluator::new(net), sampler: None }
    }

    pub fn sampled(color: Color, net: Arc<PolicyValueNet>, seed: u64) -> Self {
        NetPlayer { color, eval: Evaluator::new(net), sampler: Some(Rng::new(seed)) }
    }
}

impl Player for NetPlayer {
    fn color(&self) -> Color {
        self.color
    }
    fn decide(&mut self, game: &Game, actions: &[Action]) -> Action {
        if actions.len() == 1 {
            return actions[0];
        }
        let (priors, _) = self.eval.evaluate(game);
        let pick = match &mut self.sampler {
            Some(rng) => sample_index(&priors, rng),
            None => (0..priors.len()).max_by(|&a, &b| priors[a].total_cmp(&priors[b])).unwrap(),
        };
        game.playable_actions[pick]
    }
}

/// Index drawn with probability proportional to `weights`.
pub fn sample_index(weights: &[f32], rng: &mut Rng) -> usize {
    let total: f32 = weights.iter().sum();
    let mut x = (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * total;
    for (i, &w) in weights.iter().enumerate() {
        if x < w {
            return i;
        }
        x -= w;
    }
    // Rounding can leave x just past the end: take the last nonzero weight.
    weights.iter().rposition(|&w| w > 0.0).unwrap_or(0)
}

#[derive(Clone, Copy, Debug)]
pub struct MctsConfig {
    /// Simulations per decision.
    pub simulations: u32,
    /// Rollout openings: this many of the player's initial settlements (0 =
    /// none, 1 = the first, 2 = both) are chosen by outcomes instead of by
    /// search: the top `rollout_k` spots by prior each get `rollout_n` fast
    /// network-only games with fresh dice, and the best win rate is played.
    /// The value head is weakest on openings; game results are not.
    pub rollout_openings: u32,
    pub rollout_k: usize,
    pub rollout_n: u32,
    /// Exploration constant in the PUCT formula.
    pub c_puct: f32,
    /// Value assumed for unvisited moves: the node's estimate minus this.
    pub fpu_reduction: f32,
    /// Leaves selected per round and evaluated together (virtual loss keeps
    /// them apart). 1 = plain sequential search.
    pub batch: u32,
    /// Share of the root priors replaced by Dirichlet noise (0 = off), for
    /// exploration in self-play.
    pub root_noise: f32,
    pub dirichlet_alpha: f32,
    /// Simulations for initial placements (0 = `simulations`): few
    /// decisions that shape the whole game.
    pub placement_sims: u32,
    /// How finished games are valued beyond win/loss.
    pub utility: Utility,
    /// Gumbel root search (Danihelka et al. 2022) over this many candidate
    /// moves (0 = plain PUCT at the root): the candidates are the top moves
    /// by log prior (+ Gumbel noise with `gumbel_noise`), the budget is
    /// spread over them by sequential halving, and the move played is the
    /// survivor with the best log prior + noise + scaled value. Below the
    /// root, search is PUCT as usual. Needs `batch` = 1; Dirichlet noise is
    /// not used with it.
    pub gumbel_m: u32,
    /// Gumbel noise on the root candidates (exploration, for self-play).
    pub gumbel_noise: bool,
    /// Wall-clock budget per search in milliseconds (0 = none). When set,
    /// simulations continue until it runs out; `simulations` is ignored.
    pub time_limit_ms: u64,
    /// Root parallelism for playing (`Player::decide`): this many independent
    /// searches run at once, each with its own determinization and dice,
    /// and their root visit counts are summed (0 or 1 = one search).
    pub root_threads: u32,
    /// Shared-tree parallelism for playing: this many worker threads grow
    /// one tree together (virtual loss keeps them apart; the tree lock is
    /// only held for selection, insertion and backup, never for game steps
    /// or network evaluations). 0 or 1 = one thread.
    pub tree_threads: u32,
    /// Lock-free shared-tree parallelism for playing (`search_lockfree`):
    /// this many worker threads grow one tree with atomic statistics. 0 or
    /// 1 = one thread.
    pub lockfree_threads: u32,
    /// Progressive widening at chance nodes (Couëtoux et al. 2011): a
    /// random event's outcome children grow to at most
    /// ceil(`chance_k` * visits^`chance_alpha`); past that, a simulation
    /// follows an existing outcome, chosen in proportion to its visits (each
    /// outcome was first drawn with the true odds). Keeps the tree narrow
    /// at dice so it reaches deeper. 0 = off (every draw may branch).
    pub chance_alpha: f32,
    pub chance_k: f32,
    /// Keep the tree between decisions (single-threaded PUCT search only):
    /// a search starts from the node of the previous tree that is the
    /// current position as far as this player can tell (same legal moves,
    /// same observation, same dice odds), with its statistics, and adds
    /// `simulations` to them. Needs no root noise.
    pub tree_reuse: bool,
    /// Prior-mass pruning below the root (single-threaded PUCT search): at
    /// a non-root node only the moves with the highest network priors that
    /// together reach this share of the prior mass can be selected (at least
    /// one). 1.0 = off.
    pub prune_mass: f32,
    /// Value pruning below the root (single-threaded PUCT search): at a
    /// non-root node, a move with at least `Q_PRUNE_VISITS` visits whose mean
    /// value is more than this below the best such move's is no longer
    /// selected. 0 = off.
    pub q_prune_margin: f32,
}

/// Visits a move needs before `MctsConfig::q_prune_margin` judges it.
pub const Q_PRUNE_VISITS: u32 = 16;

/// Value of a finished game for each player: the result (+1 win, -1 loss)
/// plus
/// - `margin` * (own points - best opponent's) / points to win, and
/// - for the winner `speed` * exp(-turns / `speed_turns`), with the same
///   amount taken from the others: winning sooner is worth more, losing
///   later costs less. It shrinks with game length but never vanishes, so a
///   sure win still has a reason to come sooner.
///
/// A game with no winner (the turn limit) is a loss for everyone, -1 each:
/// stalling is never safe. All zero = plain win/loss.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Utility {
    pub margin: f32,
    pub speed: f32,
    pub speed_turns: f32,
}

impl Utility {
    /// Largest possible |value|.
    pub fn bound(&self) -> f32 {
        1.0 + self.margin + self.speed
    }
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig { simulations: 200, rollout_openings: 0, rollout_k: 6, rollout_n: 100, c_puct: 1.5, fpu_reduction: 0.1, batch: 1, root_noise: 0.0, dirichlet_alpha: 0.3, placement_sims: 0, utility: Utility { margin: 0.0, speed: 0.0, speed_turns: 0.0 }, gumbel_m: 0, gumbel_noise: false, time_limit_ms: 0, root_threads: 0, tree_threads: 0, lockfree_threads: 0, chance_alpha: 0.0, chance_k: 1.0, tree_reuse: false, prune_mass: 1.0, q_prune_margin: 0.0 }
    }
}

/// Per-color values (indexed by `Color::idx`).
type Values = [f32; 4];

enum Child {
    Unvisited,
    Node(usize),
    /// Stochastic action: one child per sampled outcome.
    Chance(Vec<(ActionResult, usize)>),
}

struct Edge {
    action: Action,
    /// Prior used by the search (root noise mixed in at the root).
    prior: f32,
    /// The network's prior, without noise.
    net_prior: f32,
    visits: u32,
    value_sum: Values,
    child: Child,
}

struct Node {
    game: Game,
    edges: Vec<Edge>,
    expanded: bool,
    visits: u32,
    /// Network value for the mover at expansion (for first-play urgency).
    value: f32,
    /// Lowest prior a move may have to be selected below the root
    /// (`MctsConfig::prune_mass`; 0 = all).
    prior_floor: f32,
}

/// A move that wins the game outright for the player to move, if any:
/// deterministic moves only (a dev card purchase or roll may not), and only
/// looked for within reach (no single move is worth more than 2 points).
pub fn immediate_win(game: &Game) -> Option<usize> {
    let s = &game.state;
    let me = s.current_color();
    if s.player(me).actual_victory_points + 2 < game.vps_to_win {
        return None;
    }
    game.playable_actions.iter().position(|a| {
        if is_stochastic(a, game) {
            return false;
        }
        let mut g = game.copy_without_history();
        g.execute_unchecked(*a, None).is_ok() && g.winning_color() == Some(me)
    })
}

pub(crate) fn is_stochastic(a: &Action, g: &Game) -> bool {
    match a.kind {
        ActionKind::Roll | ActionKind::BuyDevelopmentCard => true,
        ActionKind::MoveRobber { victim: Some(v), .. } => g.state.player(v).num_resources() > 0,
        _ => false,
    }
}

/// Mover gets `v`; in 2-player games the opponent gets `-v`, with more
/// players the loss is split among them.
fn spread(g: &Game, mover: Color, v: f32) -> Values {
    let n = g.state.colors.len() as f32;
    let mut out = [0.0; 4];
    for &c in g.state.colors.iter() {
        out[c.idx()] = if c == mover { v } else { -v / (n - 1.0) };
    }
    out
}

/// Values of a finished game under `u` (see `Utility`).
pub(crate) fn terminal_values(g: &Game, u: &Utility) -> Values {
    let Some(w) = g.winning_color() else {
        let mut v = [0.0; 4];
        for &c in g.state.colors.iter() {
            v[c.idx()] = -1.0;
        }
        return v;
    };
    let quick = if u.speed > 0.0 { u.speed * (-(g.state.num_turns as f32) / u.speed_turns).exp() } else { 0.0 };
    let mut v = spread(g, w, 1.0 + quick);
    let margin_weight = u.margin;
    if margin_weight != 0.0 {
        let players = &g.state.players;
        for p in players.iter() {
            let best_other = players.iter().filter(|o| o.color != p.color).map(|o| o.actual_victory_points).max().unwrap_or(0);
            let margin = p.actual_victory_points as f32 - best_other as f32;
            v[p.color.idx()] += margin_weight * margin / g.vps_to_win as f32;
        }
    }
    v
}

/// Uniform in (0, 1).
fn uniform(rng: &mut Rng) -> f64 {
    ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// Gamma(shape, 1) variate (Marsaglia-Tsang), for Dirichlet noise.
fn gamma(rng: &mut Rng, shape: f64) -> f64 {
    if shape < 1.0 {
        return gamma(rng, shape + 1.0) * uniform(rng).powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        // Standard normal by Box-Muller.
        let x = (-2.0 * uniform(rng).ln()).sqrt() * (std::f64::consts::TAU * uniform(rng)).cos();
        let v = (1.0 + c * x).powi(3);
        if v > 0.0 && uniform(rng).ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

/// Value taken from a mover for each pending descent through an edge.
const VIRTUAL_LOSS: f32 = 1.0;

/// PUCT selection at `id` (as `MctsPlayer::select`).
fn select_in(nodes: &[Node], id: usize, cfg: &MctsConfig) -> usize {
    let node = &nodes[id];
    let mover = node.game.state.current_color().idx();
    let sqrt_n = (node.visits.max(1) as f32).sqrt();
    let fpu = node.value - cfg.fpu_reduction;
    let mut best = (f32::NEG_INFINITY, 0);
    for (i, e) in node.edges.iter().enumerate() {
        let q = if e.visits == 0 { fpu } else { e.value_sum[mover] / e.visits as f32 };
        let u = cfg.c_puct * e.prior * sqrt_n / (1.0 + e.visits as f32);
        if q + u > best.0 {
            best = (q + u, i);
        }
    }
    best.1
}

fn expand_in(nodes: &mut [Node], id: usize, priors: Vec<f32>, value: f32) {
    let node = &mut nodes[id];
    node.edges = node
        .game
        .playable_actions
        .iter()
        .zip(priors)
        .map(|(&action, prior)| Edge { action, prior, net_prior: prior, visits: 0, value_sum: [0.0; 4], child: Child::Unvisited })
        .collect();
    node.expanded = true;
    node.value = value;
}

fn new_node_in(nodes: &mut Vec<Node>, game: Game) -> usize {
    nodes.push(Node { game, edges: Vec::new(), expanded: false, visits: 0, value: 0.0, prior_floor: 0.0 });
    nodes.len() - 1
}

/// One simulation of the shared-tree search. Holds the tree lock only to
/// walk, insert and back up; game steps and network evaluations run
/// without it, under virtual loss on the path taken.
fn shared_simulation(
    tree: &Mutex<Vec<Node>>,
    cfg: &MctsConfig,
    eval: &mut Evaluator,
    rng: &mut Rng,
    path: &mut Vec<(usize, usize)>,
) -> Result<(), String> {
    let vl = VIRTUAL_LOSS;
    path.clear();
    let mut nodes = tree.lock().unwrap();
    let mut id = 0usize;
    let value: Values = loop {
        if nodes[id].game.is_over() {
            break terminal_values(&nodes[id].game, &cfg.utility);
        }
        if !nodes[id].expanded {
            if nodes[id].game.playable_actions.len() == 1 {
                expand_in(&mut nodes, id, vec![1.0], 0.0);
            } else {
                // Evaluate outside the lock (another thread may expand it
                // meanwhile: then its expansion stands, our value still counts).
                nodes[id].visits += 1;
                let g = nodes[id].game.copy_without_history();
                drop(nodes);
                let (priors, v) = eval.evaluate(&g);
                nodes = tree.lock().unwrap();
                if !nodes[id].expanded {
                    expand_in(&mut nodes, id, priors, v);
                }
                break spread(&g, g.state.current_color(), v);
            }
        }
        let e = select_in(&nodes, id, cfg);
        let mover = nodes[id].game.state.current_color().idx();
        nodes[id].visits += 1;
        let edge = &mut nodes[id].edges[e];
        edge.visits += 1;
        edge.value_sum[mover] -= vl;
        path.push((id, e));
        let action = edge.action;
        // Find or make the child.
        let stochastic = is_stochastic(&action, &nodes[id].game);
        if !stochastic {
            if let Child::Node(c) = nodes[id].edges[e].child {
                id = c;
                continue;
            }
        }
        let mut g = nodes[id].game.copy_without_history();
        drop(nodes);
        let outcome = if stochastic {
            g.state.rng = Rng::new(rng.next_u64());
            let mut deck = g.state.dev_deck;
            rng.shuffle(&mut deck);
            g.state.dev_deck = deck;
            match g.execute_unchecked(action, None)?.result {
                ActionResult::Dice(a, b) => Some(ActionResult::Dice(a + b, 0)),
                r => Some(r),
            }
        } else {
            g.execute_unchecked(action, None)?;
            None
        };
        nodes = tree.lock().unwrap();
        let parent = id;
        id = match (outcome, &nodes[parent].edges[e].child) {
            (None, Child::Node(c)) => *c, // another thread made it meanwhile
            (Some(o), Child::Chance(list)) if list.iter().any(|(r, _)| *r == o) => {
                list.iter().find(|(r, _)| *r == o).unwrap().1
            }
            _ => {
                let c = new_node_in(&mut nodes, g);
                match (outcome, &mut nodes[parent].edges[e].child) {
                    (None, slot) => *slot = Child::Node(c),
                    (Some(o), Child::Chance(list)) => list.push((o, c)),
                    (Some(o), slot) => *slot = Child::Chance(vec![(o, c)]),
                }
                c
            }
        };
    };
    for &(n, e) in path.iter() {
        let mover = nodes[n].game.state.current_color().idx();
        let edge = &mut nodes[n].edges[e];
        edge.value_sum[mover] += vl;
        for (s, x) in edge.value_sum.iter_mut().zip(value) {
            *s += x;
        }
    }
    Ok(())
}

/// Where one descent ended.
enum Leaf {
    /// Value known without the network (game over).
    Value(Values),
    /// Node waiting for a network evaluation.
    Eval(usize),
}

/// Network-guided Monte Carlo tree search player.
/// Gumbel(0, 1) sample.
fn gumbel(rng: &mut Rng) -> f32 {
    -(-(uniform(rng)).ln()).ln() as f32
}

/// Gumbel MuZero's value transform for comparing root moves: the value
/// mapped from [-bound, bound] to [0, 1], times (50 + most visits) * 0.1
/// (mctx's defaults).
fn sigma(q: f32, bound: f32, max_visits: u32) -> f32 {
    (50.0 + max_visits as f32) * 0.1 * (q.clamp(-bound, bound) + bound) / (2.0 * bound)
}

pub struct MctsPlayer {
    pub color: Color,
    pub config: MctsConfig,
    eval: Evaluator,
    rng: Rng,
    nodes: Vec<Node>,
    gumbel_choice: Option<usize>,
    /// Tree reuse: searches that started from a kept node, and the root
    /// visits they started with.
    reused_searches: u64,
    reused_visits: u64,
}

impl MctsPlayer {
    pub fn new(color: Color, net: Arc<PolicyValueNet>, config: MctsConfig, seed: u64) -> Self {
        MctsPlayer { color, config, eval: Evaluator::new(net), rng: Rng::new(seed), nodes: Vec::new(), gumbel_choice: None, reused_searches: 0, reused_visits: 0 }
    }

    /// Network evaluations so far (for benchmarking).
    pub fn evaluations(&self) -> u64 {
        self.eval.evals
    }

    /// Tree reuse so far: (searches that started from a kept node, root
    /// visits they started with).
    pub fn reuse_stats(&self) -> (u64, u64) {
        (self.reused_searches, self.reused_visits)
    }

    /// `game` as this player sees it, at node `id`: same mover, prompt and
    /// legal moves (root edges index `playable_actions`), same observation
    /// from this seat, and the same dice odds. Hidden cards may differ (the
    /// node's are a determinization).
    fn same_position(&mut self, id: usize, game: &Game, obs: &[(u32, f32)]) -> bool {
        let g = &self.nodes[id].game;
        let s = &g.state;
        if s.current_color() != game.state.current_color()
            || s.current_prompt != game.state.current_prompt
            || s.num_turns != game.state.num_turns
            || g.playable_actions != game.playable_actions
        {
            return false;
        }
        let dice = match (&s.dice_deck, &game.state.dice_deck) {
            (None, None) => true,
            (Some(a), Some(b)) => a.same_odds(b),
            _ => false,
        };
        if !dice {
            return false;
        }
        let g = self.nodes[id].game.copy_without_history();
        self.eval.observe(&g, self.color) == obs
    }

    /// Keeps the subtree of the previous tree at `game`, if some expanded
    /// node of it (within `MAX_REUSE_DEPTH` steps of its root) is `game` as
    /// far as this player can tell; it becomes node 0. Returns whether one
    /// was found (otherwise the tree is left as it was).
    fn reuse_tree(&mut self, game: &Game) -> bool {
        const MAX_REUSE_DEPTH: usize = 32;
        if self.nodes.is_empty() {
            return false;
        }
        let obs = self.eval.observe(game, self.color);
        let mut frontier = vec![0usize];
        let mut found = None;
        'search: for _ in 0..=MAX_REUSE_DEPTH {
            let mut next = Vec::new();
            for &id in &frontier {
                if !self.nodes[id].expanded {
                    continue;
                }
                if self.same_position(id, game, &obs) {
                    found = Some(id);
                    break 'search;
                }
                for e in &self.nodes[id].edges {
                    match &e.child {
                        Child::Unvisited => {}
                        Child::Node(c) => next.push(*c),
                        Child::Chance(list) => next.extend(list.iter().map(|&(_, c)| c)),
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        let Some(root) = found else { return false };
        // Copy the subtree out, renumbered from 0 in breadth-first order.
        let mut old: Vec<Option<Node>> = std::mem::take(&mut self.nodes).into_iter().map(Some).collect();
        let mut new_id = vec![usize::MAX; old.len()];
        let mut order = vec![root];
        new_id[root] = 0;
        let mut k = 0;
        while k < order.len() {
            let node = old[order[k]].as_ref().unwrap();
            for e in &node.edges {
                let kids: Vec<usize> = match &e.child {
                    Child::Unvisited => Vec::new(),
                    Child::Node(c) => vec![*c],
                    Child::Chance(list) => list.iter().map(|&(_, c)| c).collect(),
                };
                for c in kids {
                    if new_id[c] == usize::MAX {
                        new_id[c] = order.len();
                        order.push(c);
                    }
                }
            }
            k += 1;
        }
        self.nodes = order
            .iter()
            .map(|&i| {
                let mut node = old[i].take().unwrap();
                for e in node.edges.iter_mut() {
                    match &mut e.child {
                        Child::Unvisited => {}
                        Child::Node(c) => *c = new_id[*c],
                        Child::Chance(list) => list.iter_mut().for_each(|(_, c)| *c = new_id[*c]),
                    }
                }
                node
            })
            .collect();
        self.reused_searches += 1;
        self.reused_visits += self.nodes[0].visits as u64;
        true
    }

    /// The searcher's view of `game`: unknowns reshuffled, dice reseeded.
    /// Deals that would hand an opponent a win are rejected: the game is
    /// still on, so the searcher knows nobody has enough points yet.
    fn determinize(&mut self, game: &Game) -> Game {
        for _ in 0..32 {
            let g = self.deal_hidden_cards(game);
            if !g.is_over() {
                return g;
            }
        }
        // Nearly impossible; fall back to the true cards.
        let mut g = game.copy_without_history();
        g.state.rng = Rng::new(self.rng.next_u64());
        g
    }

    fn deal_hidden_cards(&mut self, game: &Game) -> Game {
        let mut g = game.copy_without_history();
        let s = &mut g.state;
        s.rng = Rng::new(self.rng.next_u64());
        let me = self.color;
        // Pool every dev card the searcher can't see, then deal it back.
        let mut pool: Vec<DevCard> = s.dev_deck.iter().copied().collect();
        let mut counts = Vec::new();
        for p in s.players.iter_mut().filter(|p| p.color != me) {
            for d in DEV_CARDS {
                pool.extend(std::iter::repeat(d).take(p.dev_cards[d.idx()] as usize));
            }
            counts.push(p.num_dev_cards());
        }
        self.rng.shuffle(&mut pool);
        let mut k = 0;
        for (p, count) in s.players.iter_mut().filter(|p| p.color != me).zip(counts) {
            p.dev_cards = [0; 5];
            for _ in 0..count {
                p.dev_cards[pool[k].idx()] += 1;
                k += 1;
            }
            p.actual_victory_points = p.victory_points + p.dev_cards[DevCard::VictoryPoint.idx()];
            for d in [DevCard::Knight, DevCard::YearOfPlenty, DevCard::Monopoly, DevCard::RoadBuilding] {
                p.dev_owned_at_start[d.idx()] = p.dev_cards[d.idx()] > 0;
            }
        }
        s.dev_deck = pool[k..].iter().copied().collect();
        g
    }

    fn new_node(&mut self, game: Game) -> usize {
        self.nodes.push(Node { game, edges: Vec::new(), expanded: false, visits: 0, value: 0.0, prior_floor: 0.0 });
        self.nodes.len() - 1
    }

    /// Creates edges from priors; `value` is the network's for the mover.
    fn expand_with(&mut self, id: usize, priors: Vec<f32>, value: Option<f32>) {
        let node = &mut self.nodes[id];
        node.edges = node
            .game
            .playable_actions
            .iter()
            .zip(priors)
            .map(|(&action, prior)| Edge { action, prior, net_prior: prior, visits: 0, value_sum: [0.0; 4], child: Child::Unvisited })
            .collect();
        node.expanded = true;
        node.value = value.unwrap_or(0.0);
        node.prior_floor = 0.0;
        if self.config.prune_mass < 1.0 && node.edges.len() > 1 {
            let mut p: Vec<f32> = node.edges.iter().map(|e| e.net_prior).collect();
            p.sort_unstable_by(|a, b| b.total_cmp(a));
            let mut mass = 0.0;
            for &x in &p {
                mass += x;
                node.prior_floor = x;
                if mass >= self.config.prune_mass {
                    break;
                }
            }
        }
    }

    /// The edge to follow from node `id`: PUCT, with the below-root pruning
    /// (`prune_mass`, `q_prune_margin`) when `id` isn't the search's root.
    fn select(&self, id: usize, is_root: bool) -> usize {
        let node = &self.nodes[id];
        let mover = node.game.state.current_color().idx();
        let sqrt_n = (node.visits.max(1) as f32).sqrt();
        let fpu = node.value - self.config.fpu_reduction;
        let floor = if is_root { 0.0 } else { node.prior_floor };
        let q_cut = if is_root || self.config.q_prune_margin <= 0.0 {
            f32::NEG_INFINITY
        } else {
            node.edges
                .iter()
                .filter(|e| e.visits >= Q_PRUNE_VISITS)
                .map(|e| e.value_sum[mover] / e.visits as f32)
                .fold(f32::NEG_INFINITY, f32::max)
                - self.config.q_prune_margin
        };
        let mut best = (f32::NEG_INFINITY, None);
        for (i, e) in node.edges.iter().enumerate() {
            let q = if e.visits == 0 { fpu } else { e.value_sum[mover] / e.visits as f32 };
            if e.net_prior < floor || (e.visits >= Q_PRUNE_VISITS && q < q_cut) {
                continue;
            }
            let u = self.config.c_puct * e.prior * sqrt_n / (1.0 + e.visits as f32);
            if best.1.is_none() || q + u > best.0 {
                best = (q + u, Some(i));
            }
        }
        best.1.expect("the most likely move and the best-valued move are never pruned")
    }

    /// Child reached by taking edge `e` from node `id` (sampling chance).
    fn child(&mut self, id: usize, e: usize) -> Result<usize, String> {
        let action = self.nodes[id].edges[e].action;
        if !is_stochastic(&action, &self.nodes[id].game) {
            if let Child::Node(c) = self.nodes[id].edges[e].child {
                return Ok(c);
            }
            let mut g = self.nodes[id].game.copy_without_history();
            g.execute_unchecked(action, None)?;
            let c = self.new_node(g);
            self.nodes[id].edges[e].child = Child::Node(c);
            return Ok(c);
        }
        // Progressive widening: with enough outcome children for this many
        // visits, follow one of them instead of drawing a new one.
        if self.config.chance_alpha > 0.0 {
            if let Child::Chance(list) = &self.nodes[id].edges[e].child {
                let visits = self.nodes[id].edges[e].visits.max(1) as f32;
                let allowed = (self.config.chance_k * visits.powf(self.config.chance_alpha)).ceil().max(1.0) as usize;
                if list.len() >= allowed {
                    let weights: Vec<f32> = list.iter().map(|&(_, c)| self.nodes[c].visits as f32 + 1.0).collect();
                    let k = sample_index(&weights, &mut self.rng);
                    return Ok(list[k].1);
                }
            }
        }
        // Sample an outcome with fresh randomness.
        let mut g = self.nodes[id].game.copy_without_history();
        g.state.rng = Rng::new(self.rng.next_u64());
        let mut deck = g.state.dev_deck;
        self.rng.shuffle(&mut deck);
        g.state.dev_deck = deck;
        // Only a roll's total matters, so (3,4) and (4,3) share a child.
        let outcome = match g.execute_unchecked(action, None)?.result {
            ActionResult::Dice(a, b) => ActionResult::Dice(a + b, 0),
            r => r,
        };
        if let Child::Chance(list) = &self.nodes[id].edges[e].child {
            if let Some(&(_, c)) = list.iter().find(|(r, _)| *r == outcome) {
                return Ok(c);
            }
        }
        let c = self.new_node(g);
        match &mut self.nodes[id].edges[e].child {
            Child::Chance(list) => list.push((outcome, c)),
            slot => *slot = Child::Chance(vec![(outcome, c)]),
        }
        Ok(c)
    }

    /// Walks from the root to a leaf, counting the visit on the way (and,
    /// with `vl` > 0, a virtual loss for each mover so that other descents in
    /// the same round spread out). Forced moves are expanded without the
    /// network. `path` gets the (node, edge) steps taken.
    fn descend(&mut self, root: usize, vl: f32, path: &mut Vec<(usize, usize)>) -> Result<Leaf, String> {
        self.descend_from(root, None, vl, path)
    }

    /// `descend`, taking root edge `first` (if given) instead of selecting.
    fn descend_from(&mut self, root: usize, first: Option<usize>, vl: f32, path: &mut Vec<(usize, usize)>) -> Result<Leaf, String> {
        path.clear();
        let mut id = root;
        let mut forced = first;
        loop {
            if !self.nodes[id].expanded {
                let g = &self.nodes[id].game;
                if g.is_over() {
                    let v = terminal_values(g, &self.config.utility);
                    self.nodes[id].visits += 1;
                    return Ok(Leaf::Value(v));
                }
                if g.playable_actions.len() != 1 {
                    self.nodes[id].visits += 1;
                    return Ok(Leaf::Eval(id));
                }
                self.expand_with(id, vec![1.0], None);
            }
            if self.nodes[id].game.is_over() {
                return Ok(Leaf::Value(terminal_values(&self.nodes[id].game, &self.config.utility)));
            }
            let e = forced.take().unwrap_or_else(|| self.select(id, id == root));
            let node = &mut self.nodes[id];
            let mover = node.game.state.current_color().idx();
            node.visits += 1;
            let edge = &mut node.edges[e];
            edge.visits += 1;
            if vl != 0.0 {
                edge.value_sum[mover] -= vl;
            }
            path.push((id, e));
            id = self.child(id, e)?;
        }
    }

    /// Mixes Dirichlet noise into the root's priors.
    fn add_root_noise(&mut self, root: usize) {
        let n = self.nodes[root].edges.len();
        let noise: Vec<f64> = (0..n).map(|_| gamma(&mut self.rng, self.config.dirichlet_alpha as f64)).collect();
        let total: f64 = noise.iter().sum();
        if total <= 0.0 {
            return;
        }
        let eps = self.config.root_noise;
        for (e, x) in self.nodes[root].edges.iter_mut().zip(noise) {
            e.prior = (1.0 - eps) * e.prior + eps * (x / total) as f32;
        }
    }

    /// Adds `v` along `path` and takes back the virtual loss.
    fn backup(&mut self, path: &[(usize, usize)], v: Values, vl: f32) {
        for &(id, e) in path {
            let mover = self.nodes[id].game.state.current_color().idx();
            let edge = &mut self.nodes[id].edges[e];
            if vl != 0.0 {
                edge.value_sum[mover] += vl;
            }
            for (s, x) in edge.value_sum.iter_mut().zip(v) {
                *s += x;
            }
        }
    }

    /// Depth of the last search tree: (max moves ahead, visit-weighted
    /// mean moves ahead, max turns ahead, visit-weighted mean turns ahead,
    /// nodes). "Moves" counts every action, forced ones included.
    pub fn tree_depth_stats(&self) -> (u32, f64, u32, f64, usize) {
        if self.nodes.is_empty() {
            return (0, 0.0, 0, 0.0, 0);
        }
        let root_turn = self.nodes[0].game.state.num_turns;
        let mut stack = vec![(0usize, 0u32)];
        let (mut max_d, mut max_t, mut wd, mut wt, mut w) = (0, 0, 0.0, 0.0, 0.0);
        while let Some((id, d)) = stack.pop() {
            let node = &self.nodes[id];
            let turns = node.game.state.num_turns - root_turn;
            if node.visits > 0 && id != 0 {
                max_d = max_d.max(d);
                max_t = max_t.max(turns);
                wd += d as f64 * node.visits as f64;
                wt += turns as f64 * node.visits as f64;
                w += node.visits as f64;
            }
            for e in &node.edges {
                match &e.child {
                    Child::Unvisited => {}
                    Child::Node(c) => stack.push((*c, d + 1)),
                    Child::Chance(list) => stack.extend(list.iter().map(|&(_, c)| (c, d + 1))),
                }
            }
        }
        (max_d, wd / w.max(1.0), max_t, wt / w.max(1.0), self.nodes.len())
    }

    /// After `search`: per root action (in `playable_actions` order) the
    /// network prior without noise, visits, and mean value for the mover
    /// (None if unvisited); and the network's value of the root.
    pub fn root_stats(&self) -> (Vec<(f32, u32, Option<f32>)>, f32) {
        let root = &self.nodes[0];
        let mover = root.game.state.current_color().idx();
        let edges = root
            .edges
            .iter()
            .map(|e| (e.net_prior, e.visits, (e.visits > 0).then(|| e.value_sum[mover] / e.visits as f32)))
            .collect();
        (edges, root.value)
    }

    /// One simulation from the root, optionally through root edge `first`.
    fn simulate(&mut self, root: usize, first: Option<usize>, path: &mut Vec<(usize, usize)>) -> Result<(), String> {
        let v = match self.descend_from(root, first, 0.0, path)? {
            Leaf::Value(v) => v,
            Leaf::Eval(id) => {
                let (priors, value) = self.eval.evaluate(&self.nodes[id].game);
                self.expand_with(id, priors, Some(value));
                let g = &self.nodes[id].game;
                spread(g, g.state.current_color(), value)
            }
        };
        self.backup(path, v, 0.0);
        Ok(())
    }

    /// Gumbel root search with `sims` simulations; returns the chosen root
    /// edge (see `MctsConfig::gumbel_m`).
    fn gumbel_search(&mut self, root: usize, sims: usize) -> Result<usize, String> {
        let mut path = Vec::new();
        self.simulate(root, None, &mut path)?; // expands the root
        let n = self.nodes[root].edges.len();
        if n == 0 {
            return Err("root has no moves".into());
        }
        let noise: Vec<f32> = (0..n).map(|_| if self.config.gumbel_noise { gumbel(&mut self.rng) } else { 0.0 }).collect();
        let base: Vec<f32> = self.nodes[root].edges.iter().zip(&noise).map(|(e, g)| e.net_prior.max(1e-30).ln() + g).collect();
        let mut alive: Vec<usize> = (0..n).collect();
        alive.sort_by(|&a, &b| base[b].total_cmp(&base[a]));
        alive.truncate((self.config.gumbel_m as usize).min(n));
        let bound = self.config.utility.bound();
        let mover = self.nodes[root].game.state.current_color().idx();
        let score = |me: &Self, e: usize, max_visits: u32| {
            let edge = &me.nodes[root].edges[e];
            let q = if edge.visits > 0 { edge.value_sum[mover] / edge.visits as f32 } else { me.nodes[root].value };
            base[e] + sigma(q, bound, max_visits)
        };
        let phases = (alive.len() as f32).log2().ceil().max(1.0) as usize;
        let mut left = sims.saturating_sub(1);
        while alive.len() > 1 && left > 0 {
            let per = (left / (phases * alive.len())).max(1);
            for &e in &alive.clone() {
                for _ in 0..per {
                    if left == 0 {
                        break;
                    }
                    self.simulate(root, Some(e), &mut path)?;
                    left -= 1;
                }
            }
            let max_visits = self.nodes[root].edges.iter().map(|e| e.visits).max().unwrap_or(0);
            alive.sort_by(|&a, &b| score(self, b, max_visits).total_cmp(&score(self, a, max_visits)));
            alive.truncate(alive.len().div_ceil(2));
        }
        let max_visits = self.nodes[root].edges.iter().map(|e| e.visits).max().unwrap_or(0);
        Ok(*alive.iter().max_by(|&&a, &&b| score(self, a, max_visits).total_cmp(&score(self, b, max_visits))).unwrap())
    }

    /// Shared-tree parallel search (`MctsConfig::tree_threads`) of `game`;
    /// returns visit counts per playable action at the root. Stops after
    /// `simulations` (or `placement_sims`) in total, or at the time budget.
    pub fn search_shared(&mut self, game: &Game) -> Result<Vec<u32>, String> {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        self.nodes.clear();
        let root_game = self.determinize(game);
        let root = self.new_node(root_game);
        let (priors, value) = self.eval.evaluate(&self.nodes[root].game);
        self.expand_with(root, priors, Some(value));
        let sims = match self.config.placement_sims {
            p if p > 0 && game.state.is_initial_build_phase => p,
            _ => self.config.simulations,
        } as usize;
        let deadline = (self.config.time_limit_ms > 0)
            .then(|| std::time::Instant::now() + std::time::Duration::from_millis(self.config.time_limit_ms));
        let tree = Mutex::new(std::mem::take(&mut self.nodes));
        let done = AtomicUsize::new(1);
        let failed = AtomicBool::new(false);
        let error: Mutex<Option<String>> = Mutex::new(None);
        let cfg = self.config;
        let net = self.eval.net.clone();
        let seeds: Vec<u64> = (0..cfg.tree_threads.max(1)).map(|_| self.rng.next_u64()).collect();
        std::thread::scope(|scope| {
            for &seed in &seeds {
                let (tree, done, failed, error, net) = (&tree, &done, &failed, &error, net.clone());
                scope.spawn(move || {
                    let mut eval = Evaluator::new(net);
                    let mut rng = Rng::new(seed);
                    let mut path: Vec<(usize, usize)> = Vec::new();
                    while !failed.load(Ordering::Relaxed) {
                        let more = match deadline {
                            Some(d) => std::time::Instant::now() < d,
                            None => done.load(Ordering::Relaxed) < sims,
                        };
                        if !more {
                            break;
                        }
                        if let Err(e) = shared_simulation(tree, &cfg, &mut eval, &mut rng, &mut path) {
                            *error.lock().unwrap() = Some(e);
                            failed.store(true, Ordering::Relaxed);
                            break;
                        }
                        done.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
        });
        self.nodes = tree.into_inner().unwrap();
        if let Some(e) = error.into_inner().unwrap() {
            return Err(e);
        }
        Ok(self.nodes[root].edges.iter().map(|e| e.visits).collect())
    }

    /// Root-parallel search (`MctsConfig::root_threads`): the move with the
    /// most visits summed over independent searches, ties to the lower index.
    fn parallel_choice(&mut self, game: &Game) -> usize {
        let n = self.config.root_threads as usize;
        let mut single = self.config;
        single.root_threads = 1;
        let net = self.eval.net.clone();
        let color = self.color;
        let seeds: Vec<u64> = (0..n).map(|_| self.rng.next_u64()).collect();
        let all: Vec<Vec<u32>> = std::thread::scope(|scope| {
            let handles: Vec<_> = seeds
                .iter()
                .map(|&seed| {
                    let net = net.clone();
                    scope.spawn(move || {
                        let mut p = MctsPlayer::new(color, net, single, seed);
                        p.search(game).expect("search applies legal actions only")
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("search thread panicked")).collect()
        });
        let mut total = vec![0u64; all[0].len()];
        for v in &all {
            for (t, &x) in total.iter_mut().zip(v) {
                *t += x as u64;
            }
        }
        (0..total.len()).max_by(|&a, &b| total[a].cmp(&total[b]).then(b.cmp(&a))).unwrap()
    }

    /// The move the last Gumbel search chose (index into the root's moves).
    pub fn gumbel_choice(&self) -> Option<usize> {
        self.gumbel_choice
    }

    /// Visit counts per playable action of `game` after searching.
    pub fn search(&mut self, game: &Game) -> Result<Vec<u32>, String> {
        self.gumbel_choice = None;
        if self.config.tree_reuse && (self.config.root_noise > 0.0 || self.config.gumbel_m > 0) {
            return Err("tree reuse needs plain PUCT search without root noise".into());
        }
        let root = match self.config.tree_reuse && self.reuse_tree(game) {
            true => 0,
            false => {
                self.nodes.clear();
                let root_game = self.determinize(game);
                self.new_node(root_game)
            }
        };
        let sims = match self.config.placement_sims {
            p if p > 0 && game.state.is_initial_build_phase => p,
            _ => self.config.simulations,
        } as usize;
        if self.config.gumbel_m > 0 {
            if self.config.time_limit_ms > 0 {
                return Err("Gumbel root search doesn't support a time budget".into());
            }
            if self.config.batch > 1 {
                return Err("Gumbel root search needs batch 1".into());
            }
            self.gumbel_choice = Some(self.gumbel_search(root, sims)?);
            return Ok(self.nodes[root].edges.iter().map(|e| e.visits).collect());
        }
        let k_max = self.config.batch.max(1) as usize;
        let vl = if k_max > 1 { VIRTUAL_LOSS } else { 0.0 };
        let mut paths = vec![Vec::new(); k_max];
        let mut leaves = Vec::with_capacity(k_max);
        let mut done = 0;
        let deadline = (self.config.time_limit_ms > 0)
            .then(|| std::time::Instant::now() + std::time::Duration::from_millis(self.config.time_limit_ms));
        while match deadline {
            Some(d) => done == 0 || std::time::Instant::now() < d,
            None => done < sims,
        } {
            let k = if deadline.is_some() { k_max } else { k_max.min(sims - done) };
            leaves.clear();
            for path in paths[..k].iter_mut() {
                leaves.push(self.descend(root, vl, path)?);
            }
            // Evaluate each distinct leaf once (two descents can meet).
            let mut ids: Vec<usize> = leaves.iter().filter_map(|l| match l {
                Leaf::Eval(id) => Some(*id),
                Leaf::Value(_) => None,
            }).collect();
            ids.sort_unstable();
            ids.dedup();
            let results = match ids.len() {
                0 => Vec::new(),
                1 => vec![self.eval.evaluate(&self.nodes[ids[0]].game)],
                _ => {
                    let games: Vec<&Game> = ids.iter().map(|&id| &self.nodes[id].game).collect();
                    self.eval.evaluate_batch(&games)
                }
            };
            for (&id, (priors, value)) in ids.iter().zip(results) {
                self.expand_with(id, priors, Some(value));
                if id == root && self.config.root_noise > 0.0 {
                    self.add_root_noise(root);
                }
            }
            for (path, leaf) in paths[..k].iter().zip(&leaves) {
                let v = match *leaf {
                    Leaf::Value(v) => v,
                    Leaf::Eval(id) => {
                        let g = &self.nodes[id].game;
                        spread(g, g.state.current_color(), self.nodes[id].value)
                    }
                };
                self.backup(path, v, vl);
            }
            done += k;
        }
        let visits = self.nodes[root].edges.iter().map(|e| e.visits).collect();
        Ok(visits)
    }
}

impl MctsPlayer {
    /// Index of the playable settlement with the best win rate over
    /// `rollout_n` network-only games from each of the top `rollout_k` spots.
    fn rollout_placement(&mut self, game: &Game) -> usize {
        let (priors, _) = self.eval.evaluate(game);
        let mut order: Vec<usize> = (0..priors.len()).collect();
        order.sort_by(|&a, &b| priors[b].total_cmp(&priors[a]));
        let me = self.color;
        let mut best = (order[0], -1.0f32);
        for &i in order.iter().take(self.config.rollout_k.max(1)) {
            let mut wins = 0u32;
            for _ in 0..self.config.rollout_n.max(1) {
                let mut g = game.copy_without_history();
                g.state.rng = Rng::new(self.rng.next_u64());
                self.rng.shuffle(&mut g.state.dev_deck);
                let a = g.playable_actions[i];
                g.execute(a).expect("legal placement");
                while !g.is_over() && g.state.num_turns < 300 {
                    let j = if g.playable_actions.len() == 1 {
                        0
                    } else {
                        let (p, _) = self.eval.evaluate(&g);
                        (0..p.len()).max_by(|&x, &y| p[x].total_cmp(&p[y])).unwrap()
                    };
                    let a = g.playable_actions[j];
                    g.execute(a).expect("legal");
                }
                wins += (g.winning_color() == Some(me)) as u32;
            }
            let rate = wins as f32 / self.config.rollout_n.max(1) as f32;
            if rate > best.1 {
                best = (i, rate);
            }
        }
        best.0
    }
}

impl Player for MctsPlayer {
    fn color(&self) -> Color {
        self.color
    }

    fn decide(&mut self, game: &Game, actions: &[Action]) -> Action {
        if actions.len() == 1 {
            return actions[0];
        }
        // A win now beats any estimate of a win later.
        if let Some(i) = immediate_win(game) {
            return actions[i];
        }
        if self.config.rollout_openings > 0
            && game.state.is_initial_build_phase
            && matches!(actions[0].kind, ActionKind::BuildSettlement(_))
            && (game.state.player(self.color).settlements.len() as u32) < self.config.rollout_openings
        {
            return actions[self.rollout_placement(game)];
        }
        if self.config.lockfree_threads > 1 && self.config.gumbel_m == 0 {
            let visits = self.search_lockfree(game).expect("search applies legal actions only");
            let best = (0..visits.len()).max_by(|&a, &b| visits[a].cmp(&visits[b]).then(b.cmp(&a))).unwrap();
            return actions[best];
        }
        if self.config.tree_threads > 1 && self.config.gumbel_m == 0 {
            let visits = self.search_shared(game).expect("search applies legal actions only");
            let best = (0..visits.len()).max_by(|&a, &b| visits[a].cmp(&visits[b]).then(b.cmp(&a))).unwrap();
            return actions[best];
        }
        if self.config.root_threads > 1 && self.config.gumbel_m == 0 {
            return actions[self.parallel_choice(game)];
        }
        let visits = self.search(game).expect("search applies legal actions only");
        if let Some(choice) = self.gumbel_choice {
            return actions[choice];
        }
        let root = &self.nodes[0];
        // Most visited; ties go to the higher prior.
        let best = (0..visits.len())
            .max_by(|&a, &b| visits[a].cmp(&visits[b]).then(root.edges[a].prior.total_cmp(&root.edges[b].prior)))
            .unwrap();
        // Root edges follow the same order as `playable_actions`.
        actions[best]
    }
}

// ---------------------------------------------------------------------------
// Lock-free shared-tree search (`MctsConfig::lockfree_threads`).
//
// After Enzenberger & Müller (2009) and the Leela/KataGo trees: statistics
// are atomics (visits; value sums in fixed point), a node is expanded once
// by whichever thread wins a compare-and-swap on its state (others wait the
// few microseconds it takes), child links are written once (`OnceLock`),
// chance outcomes take slots in a small fixed table by compare-and-swap,
// and nodes live in an append-only chunked arena so they never move.
// Virtual loss keeps the threads on different paths.
// ---------------------------------------------------------------------------

use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU8, AtomicUsize, Ordering};

/// Fixed-point scale for value sums.
const VSCALE: f64 = 1_000_000.0;
/// Most distinct outcomes of one random event (dice totals 2..=12).
const CHANCE_SLOTS: usize = 11;
const ARENA_CHUNK: usize = 4096;
const ARENA_CHUNKS: usize = 4096;

const UNEXPANDED: u8 = 0;
const EXPANDING: u8 = 1;
const EXPANDED: u8 = 2;

struct LfSlot {
    /// Outcome key (0 = free); see `outcome_key`.
    key: AtomicU32,
    child: OnceLock<usize>,
}

struct LfEdge {
    action: Action,
    prior: f32,
    visits: AtomicU32,
    value_sum: [AtomicI64; 4],
    /// Deterministic child.
    child: OnceLock<usize>,
    /// Children by outcome, for random events.
    chance: Option<Box<[LfSlot; CHANCE_SLOTS]>>,
}

struct LfNode {
    game: Game,
    state: AtomicU8,
    visits: AtomicU32,
    /// Network value for the mover, as f32 bits (first-play urgency).
    value: AtomicU32,
    edges: OnceLock<Box<[LfEdge]>>,
}

impl LfNode {
    fn new(game: Game) -> Self {
        LfNode { game, state: AtomicU8::new(UNEXPANDED), visits: AtomicU32::new(0), value: AtomicU32::new(0), edges: OnceLock::new() }
    }
}

/// Append-only arena: a node keeps its index (and address) for life.
struct LfArena {
    chunks: Box<[OnceLock<Box<[OnceLock<LfNode>]>>]>,
    next: AtomicUsize,
}

impl LfArena {
    fn new() -> Self {
        LfArena { chunks: (0..ARENA_CHUNKS).map(|_| OnceLock::new()).collect(), next: AtomicUsize::new(0) }
    }

    fn len(&self) -> usize {
        self.next.load(Ordering::Relaxed)
    }

    fn push(&self, node: LfNode) -> Result<usize, String> {
        let i = self.next.fetch_add(1, Ordering::Relaxed);
        let chunk = self
            .chunks
            .get(i / ARENA_CHUNK)
            .ok_or("search tree arena is full")?
            .get_or_init(|| (0..ARENA_CHUNK).map(|_| OnceLock::new()).collect());
        if chunk[i % ARENA_CHUNK].set(node).is_err() {
            return Err("arena slot reused".into());
        }
        Ok(i)
    }

    fn get(&self, i: usize) -> &LfNode {
        self.chunks[i / ARENA_CHUNK].get().expect("published chunk")[i % ARENA_CHUNK].get().expect("published node")
    }
}

fn outcome_key(r: &ActionResult) -> u32 {
    match *r {
        ActionResult::Dice(total, _) => 1 + total as u32,
        ActionResult::DevCard(d) => 100 + d.idx() as u32,
        ActionResult::Stolen(Some(res)) => 200 + res.idx() as u32,
        ActionResult::Stolen(None) => 300,
        ActionResult::Discarded(res) => 400 + res.idx() as u32,
        ActionResult::None => 500,
    }
}

fn lf_expand(node: &LfNode, priors: &[f32], value: f32) {
    let edges: Box<[LfEdge]> = node
        .game
        .playable_actions
        .iter()
        .zip(priors)
        .map(|(&action, &prior)| LfEdge {
            action,
            prior,
            visits: AtomicU32::new(0),
            value_sum: Default::default(),
            child: OnceLock::new(),
            chance: is_stochastic(&action, &node.game).then(|| {
                Box::new(std::array::from_fn(|_| LfSlot { key: AtomicU32::new(0), child: OnceLock::new() }))
            }),
        })
        .collect();
    let _ = node.edges.set(edges);
    node.value.store(value.to_bits(), Ordering::Relaxed);
    node.state.store(EXPANDED, Ordering::Release);
}

fn lf_select(node: &LfNode, cfg: &MctsConfig) -> usize {
    let edges = node.edges.get().expect("expanded");
    let mover = node.game.state.current_color().idx();
    let sqrt_n = (node.visits.load(Ordering::Relaxed).max(1) as f32).sqrt();
    let fpu = f32::from_bits(node.value.load(Ordering::Relaxed)) - cfg.fpu_reduction;
    let mut best = (f32::NEG_INFINITY, 0);
    for (i, e) in edges.iter().enumerate() {
        let n = e.visits.load(Ordering::Relaxed);
        let q = if n == 0 { fpu } else { (e.value_sum[mover].load(Ordering::Relaxed) as f64 / VSCALE / n as f64) as f32 };
        let u = cfg.c_puct * e.prior * sqrt_n / (1.0 + n as f32);
        if q + u > best.0 {
            best = (q + u, i);
        }
    }
    best.1
}

/// One simulation of the lock-free search; `path` gets the (node, edge)
/// steps. Returns false when the node cap stopped it before a leaf.
fn lf_simulation(
    arena: &LfArena,
    cfg: &MctsConfig,
    max_nodes: usize,
    eval: &mut Evaluator,
    rng: &mut Rng,
    path: &mut Vec<(usize, usize)>,
) -> Result<bool, String> {
    let vl = (VIRTUAL_LOSS as f64 * VSCALE) as i64;
    path.clear();
    let mut id = 0usize;
    let value: Values = loop {
        let node = arena.get(id);
        if node.game.is_over() {
            break terminal_values(&node.game, &cfg.utility);
        }
        match node.state.load(Ordering::Acquire) {
            EXPANDED => {}
            UNEXPANDED
                if node.state.compare_exchange(UNEXPANDED, EXPANDING, Ordering::AcqRel, Ordering::Acquire).is_ok() =>
            {
                if node.game.playable_actions.len() == 1 {
                    lf_expand(node, &[1.0], 0.0);
                } else {
                    node.visits.fetch_add(1, Ordering::Relaxed);
                    let (priors, v) = eval.evaluate(&node.game);
                    lf_expand(node, &priors, v);
                    break spread(&node.game, node.game.state.current_color(), v);
                }
            }
            _ => {
                // Another thread is expanding it: a network evaluation takes
                // microseconds, so wait for it.
                while node.state.load(Ordering::Acquire) != EXPANDED {
                    std::hint::spin_loop();
                }
            }
        }
        let e = lf_select(node, cfg);
        let edge = &node.edges.get().expect("expanded")[e];
        let mover = node.game.state.current_color().idx();
        node.visits.fetch_add(1, Ordering::Relaxed);
        edge.visits.fetch_add(1, Ordering::Relaxed);
        edge.value_sum[mover].fetch_sub(vl, Ordering::Relaxed);
        path.push((id, e));
        id = match &edge.chance {
            None => match edge.child.get() {
                Some(&c) => c,
                None => {
                    if arena.len() >= max_nodes {
                        lf_undo(arena, path, vl);
                        return Ok(false);
                    }
                    let mut g = node.game.copy_without_history();
                    g.execute_unchecked(edge.action, None)?;
                    let c = arena.push(LfNode::new(g))?;
                    // If another thread linked a child first, use theirs.
                    *edge.child.get_or_init(|| c)
                }
            },
            Some(slots) => {
                let mut g = node.game.copy_without_history();
                g.state.rng = Rng::new(rng.next_u64());
                let mut deck = g.state.dev_deck;
                rng.shuffle(&mut deck);
                g.state.dev_deck = deck;
                let outcome = match g.execute_unchecked(edge.action, None)?.result {
                    ActionResult::Dice(a, b) => ActionResult::Dice(a + b, 0),
                    r => r,
                };
                let key = outcome_key(&outcome);
                let mut found = None;
                let mut new_game = Some(g);
                for slot in slots.iter() {
                    let k = slot.key.load(Ordering::Acquire);
                    let claimed = k == key
                        || (k == 0 && slot.key.compare_exchange(0, key, Ordering::AcqRel, Ordering::Acquire).is_ok())
                        || slot.key.load(Ordering::Acquire) == key;
                    if !claimed {
                        continue;
                    }
                    if let Some(&c) = slot.child.get() {
                        found = Some(c);
                        break;
                    }
                    if arena.len() >= max_nodes {
                        lf_undo(arena, path, vl);
                        return Ok(false);
                    }
                    let c = arena.push(LfNode::new(new_game.take().expect("one new child per simulation")))?;
                    found = Some(*slot.child.get_or_init(|| c));
                    break;
                }
                found.ok_or("more outcomes than chance slots")?
            }
        };
    };
    for &(n, e) in path.iter() {
        let node = arena.get(n);
        let mover = node.game.state.current_color().idx();
        let edge = &node.edges.get().expect("expanded")[e];
        edge.value_sum[mover].fetch_add(vl, Ordering::Relaxed);
        for (s, &x) in edge.value_sum.iter().zip(&value) {
            s.fetch_add((x as f64 * VSCALE) as i64, Ordering::Relaxed);
        }
    }
    Ok(true)
}

/// Takes back the visits and virtual loss of an abandoned descent.
fn lf_undo(arena: &LfArena, path: &[(usize, usize)], vl: i64) {
    for &(n, e) in path {
        let node = arena.get(n);
        let mover = node.game.state.current_color().idx();
        let edge = &node.edges.get().expect("expanded")[e];
        node.visits.fetch_sub(1, Ordering::Relaxed);
        edge.visits.fetch_sub(1, Ordering::Relaxed);
        edge.value_sum[mover].fetch_add(vl, Ordering::Relaxed);
    }
}

impl MctsPlayer {
    /// Lock-free shared-tree search (`MctsConfig::lockfree_threads`) of
    /// `game`; returns visit counts per playable action at the root. Stops
    /// after `simulations` (or `placement_sims`) in total, at the time
    /// budget, or when the tree reaches `LOCKFREE_MAX_NODES` nodes.
    pub fn search_lockfree(&mut self, game: &Game) -> Result<Vec<u32>, String> {
        use std::sync::atomic::AtomicBool;
        let root_game = self.determinize(game);
        let arena = LfArena::new();
        arena.push(LfNode::new(root_game))?;
        {
            let root = arena.get(0);
            root.state.store(EXPANDING, Ordering::Relaxed);
            let (priors, v) = self.eval.evaluate(&root.game);
            lf_expand(root, &priors, v);
        }
        let sims = match self.config.placement_sims {
            p if p > 0 && game.state.is_initial_build_phase => p,
            _ => self.config.simulations,
        } as usize;
        let deadline = (self.config.time_limit_ms > 0)
            .then(|| std::time::Instant::now() + std::time::Duration::from_millis(self.config.time_limit_ms));
        let done = AtomicUsize::new(1);
        let stop = AtomicBool::new(false);
        let error: Mutex<Option<String>> = Mutex::new(None);
        let cfg = self.config;
        let net = self.eval.net.clone();
        let seeds: Vec<u64> = (0..cfg.lockfree_threads.max(1)).map(|_| self.rng.next_u64()).collect();
        std::thread::scope(|scope| {
            for &seed in &seeds {
                let (arena, done, stop, error, net) = (&arena, &done, &stop, &error, net.clone());
                scope.spawn(move || {
                    let mut eval = Evaluator::new(net);
                    let mut rng = Rng::new(seed);
                    let mut path = Vec::new();
                    while !stop.load(Ordering::Relaxed) {
                        let more = match deadline {
                            Some(d) => std::time::Instant::now() < d,
                            None => done.load(Ordering::Relaxed) < sims,
                        };
                        if !more {
                            break;
                        }
                        match lf_simulation(arena, &cfg, LOCKFREE_MAX_NODES, &mut eval, &mut rng, &mut path) {
                            Ok(true) => {
                                done.fetch_add(1, Ordering::Relaxed);
                            }
                            Ok(false) => break, // node cap
                            Err(e) => {
                                *error.lock().unwrap() = Some(e);
                                stop.store(true, Ordering::Relaxed);
                                break;
                            }
                        }
                    }
                });
            }
        });
        if let Some(e) = error.into_inner().unwrap() {
            return Err(e);
        }
        let root = arena.get(0);
        Ok(root.edges.get().expect("expanded").iter().map(|e| e.visits.load(Ordering::Relaxed)).collect())
    }
}

/// Node cap for `search_lockfree` (about 2 KB each: ~6 GB).
pub const LOCKFREE_MAX_NODES: usize = 3_000_000;
