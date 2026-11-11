//! Simple bots. Each owns its RNG, so games are reproducible from seeds.

use crate::game::Game;
use crate::rng::Rng;
use crate::types::*;

pub trait Player: Send {
    fn color(&self) -> Color;
    /// Returns one of `playable_actions` (or a valid trade offer).
    fn decide(&mut self, game: &Game, playable_actions: &[Action]) -> Action;
}

/// Always takes the first playable action.
pub struct SimplePlayer {
    pub color: Color,
}

impl Player for SimplePlayer {
    fn color(&self) -> Color {
        self.color
    }
    fn decide(&mut self, _game: &Game, actions: &[Action]) -> Action {
        actions[0]
    }
}

/// Picks uniformly at random.
pub struct RandomPlayer {
    pub color: Color,
    pub rng: Rng,
}

impl RandomPlayer {
    pub fn new(color: Color, seed: u64) -> Self {
        RandomPlayer { color, rng: Rng::new(seed) }
    }
}

impl Player for RandomPlayer {
    fn color(&self) -> Color {
        self.color
    }
    fn decide(&mut self, _game: &Game, actions: &[Action]) -> Action {
        *self.rng.choose(actions)
    }
}

/// Random, but skewed toward cities > settlements > dev cards.
pub struct WeightedRandomPlayer {
    pub color: Color,
    pub rng: Rng,
}

impl WeightedRandomPlayer {
    pub fn new(color: Color, seed: u64) -> Self {
        WeightedRandomPlayer { color, rng: Rng::new(seed) }
    }

    fn weight(action: &Action) -> u64 {
        match action.kind {
            ActionKind::BuildCity(_) => 10000,
            ActionKind::BuildSettlement(_) => 1000,
            ActionKind::BuyDevelopmentCard => 100,
            _ => 1,
        }
    }
}

impl Player for WeightedRandomPlayer {
    fn color(&self) -> Color {
        self.color
    }
    fn decide(&mut self, _game: &Game, actions: &[Action]) -> Action {
        let total: u64 = actions.iter().map(Self::weight).sum();
        let mut k = self.rng.below(total as usize) as u64;
        for a in actions {
            let w = Self::weight(a);
            if k < w {
                return *a;
            }
            k -= w;
        }
        unreachable!()
    }
}

/// Greedily maximizes its own victory points one action ahead; breaks ties
/// at random.
pub struct VictoryPointPlayer {
    pub color: Color,
    pub rng: Rng,
}

impl VictoryPointPlayer {
    pub fn new(color: Color, seed: u64) -> Self {
        VictoryPointPlayer { color, rng: Rng::new(seed) }
    }
}

impl Player for VictoryPointPlayer {
    fn color(&self) -> Color {
        self.color
    }
    fn decide(&mut self, game: &Game, actions: &[Action]) -> Action {
        if actions.len() == 1 {
            return actions[0];
        }
        let mut best_value = i64::MIN;
        let mut best = Vec::new();
        for &action in actions {
            let mut copy = game.copy_without_history();
            if copy.execute(action).is_err() {
                continue;
            }
            let value = copy.state.player(self.color).actual_victory_points as i64;
            if value > best_value {
                best_value = value;
                best.clear();
            }
            if value == best_value {
                best.push(action);
            }
        }
        *self.rng.choose(&best)
    }
}

/// Builds a bot from a CLI key: R, S, W, VP, F, AB / AB:<depth>,
/// NN:<model.ctnn> (network's top move), NNS:<model.ctnn> (move sampled
/// from the network) or MCTS:<sims>[x<batch>][,c<c_puct>][,f<fpu>][,p<placement sims>][,m<margin weight>][,s<speed weight>/<turns>][,g<gumbel candidates>][,t<milliseconds per move>][,j<parallel searches>][,w<shared-tree threads>][,x<lock-free threads>][,d<chance widening alpha>][,r (keep the tree between moves)][,o<placements>/<K>x<N> (initial settlements by rollouts: best of the top K prior spots over N network-only games)][,k<prior mass kept below the root>][,q<value margin pruned below the root>]:<model.ctnn>
/// (batch = leaves evaluated together per search round, default 1; c_puct
/// and first-play-urgency reduction default to `MctsConfig::default()`).
pub fn from_key(key: &str, color: Color, seed: u64) -> Option<Box<dyn Player>> {
    if let Some(path) = key.strip_prefix("NNS:") {
        let net = crate::mcts::load_net(path).ok()?;
        return Some(Box::new(crate::mcts::NetPlayer::sampled(color, net, seed)));
    }
    if let Some(path) = key.strip_prefix("NN:") {
        let net = crate::mcts::load_net(path).ok()?;
        return Some(Box::new(crate::mcts::NetPlayer::new(color, net)));
    }
    if let Some(rest) = key.strip_prefix("MCTS:") {
        let (spec, path) = rest.split_once(':')?;
        let mut parts = spec.split(',');
        let head = parts.next()?;
        let (sims, batch) = match head.split_once('x') {
            Some((s, b)) => (s, b.parse().ok().filter(|&b| b >= 1)?),
            None => (head, 1),
        };
        let sims: u32 = sims.parse().ok().filter(|&s| s >= 1)?;
        let mut config = crate::mcts::MctsConfig { simulations: sims, batch, ..Default::default() };
        for p in parts {
            match p.split_at_checked(1)? {
                ("c", v) => config.c_puct = v.parse().ok().filter(|&c: &f32| c > 0.0)?,
                ("f", v) => config.fpu_reduction = v.parse().ok()?,
                ("p", v) => config.placement_sims = v.parse().ok()?,
                ("m", v) => config.utility.margin = v.parse().ok().filter(|&m: &f32| m >= 0.0)?,
                ("d", v) => config.chance_alpha = v.parse().ok().filter(|&a: &f32| (0.0..=1.0).contains(&a))?,
                ("x", v) => config.lockfree_threads = v.parse().ok().filter(|&x: &u32| (1..=64).contains(&x))?,
                ("w", v) => config.tree_threads = v.parse().ok().filter(|&w: &u32| (1..=64).contains(&w))?,
                ("j", v) => config.root_threads = v.parse().ok().filter(|&j: &u32| (1..=64).contains(&j))?,
                ("t", v) => config.time_limit_ms = v.parse().ok().filter(|&t: &u64| t >= 1)?,
                ("g", v) => config.gumbel_m = v.parse().ok().filter(|&m: &u32| m >= 2)?,
                ("r", "") => config.tree_reuse = true,
                ("o", v) => {
                    // o<placements>[/<K>x<N>]: initial settlements chosen by rollouts.
                    let (scope, kn) = v.split_once('/').unwrap_or((v, "6x100"));
                    config.rollout_openings = scope.parse().ok().filter(|&o: &u32| (1..=2).contains(&o))?;
                    let (k, n) = kn.split_once('x')?;
                    config.rollout_k = k.parse().ok().filter(|&k: &usize| k >= 1)?;
                    config.rollout_n = n.parse().ok().filter(|&n: &u32| n >= 1)?;
                }
                ("k", v) => config.prune_mass = v.parse().ok().filter(|&k: &f32| k > 0.0 && k <= 1.0)?,
                ("q", v) => config.q_prune_margin = v.parse().ok().filter(|&q: &f32| q > 0.0)?,
                ("s", v) => {
                    let (w, t) = v.split_once('/')?;
                    config.utility.speed = w.parse().ok().filter(|&w: &f32| w >= 0.0)?;
                    config.utility.speed_turns = t.parse().ok().filter(|&t: &f32| t > 0.0)?;
                }
                _ => return None,
            }
        }
        let net = crate::mcts::load_net(path).ok()?;
        return Some(Box::new(crate::mcts::MctsPlayer::new(color, net, config, seed)));
    }
    let (name, arg) = match key.split_once(':') {
        Some((n, a)) => (n, Some(a)),
        None => (key, None),
    };
    Some(match (name, arg) {
        ("R", None) => Box::new(RandomPlayer::new(color, seed)),
        ("S", None) => Box::new(SimplePlayer { color }),
        ("W", None) => Box::new(WeightedRandomPlayer::new(color, seed)),
        ("VP", None) => Box::new(VictoryPointPlayer::new(color, seed)),
        ("F", None) => Box::new(crate::search::ValueFunctionPlayer::new(color)),
        ("AB", depth) => {
            let depth = match depth {
                Some(d) => d.parse().ok().filter(|&d| d >= 1)?,
                None => 2,
            };
            Box::new(crate::search::AlphaBetaPlayer::new(color, depth))
        }
        _ => return None,
    })
}
