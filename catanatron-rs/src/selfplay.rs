//! AlphaZero-style self-play: both seats search with the same network, and
//! fully searched decisions become training examples (observation, legal
//! mask, root visit distribution, final result for the mover).
//!
//! The policy target is Gumbel MuZero's completed-Q target rather than raw
//! visit shares: with a sharp network prior, visits to non-top moves mostly
//! follow noisy value estimates, and training on them flattens the policy.
//!
//! Playout cap randomization (KataGo): only a `p_full` share of decisions
//! get the full search (with root noise) and are recorded; the rest use a
//! cheap search just to play reasonable moves.
//!
//! Two ways of spending more of the budget on decisions that matter:
//! - Close calls: when the cheap search's top two moves have nearly equal
//!   values, the decision is searched again fully and recorded.
//! - Endgame branches: the first position where someone reaches
//!   `endgame_vp` points is saved, and after the game ends it is replayed
//!   `endgame_branches` more times with fresh dice and a reshuffled dev deck
//!   (both hidden from the players, so each branch is an equally likely
//!   continuation), so races to the finish are oversampled.
//! - Opening branches: the position right after the initial placements is
//!   replayed `opening_branches` more times with fresh dice, and the
//!   placement decisions are labeled with the average result, since a
//!   single game's result says little about how good a placement was.
//! - Placement forks: after the game, it is replayed `placement_forks` times
//!   from one of its opening decisions (a settlement or road placement)
//!   with a different choice, drawn from the search's policy target
//!   flattened by `placement_temp` (higher = more unusual openings, still
//!   weighted by how good search found them), or with probability
//!   `placement_uniform` uniformly from the other legal choices (search's
//!   ranking of openings is itself what's being learned, so some forks
//!   ignore it). The forced move is not a training example; everything
//!   after it is. Dice and the dev deck are re-randomized: the fork's
//!   examples are learned from on their own, not compared with the
//!   original's, so independent luck gives more information.
//!
//! Finished games are valued by `utility` (result plus optional point
//! margin and speed terms, see `Utility`), consistently in search, results
//! and luck.
//!
//! Auxiliary targets (`aux`, `AUX_TARGETS` per example, all from the
//! mover's side), lower-noise signals for training the shared layers:
//! final point margin, own and opponent's final points (per point to win),
//! turns left t as t / (t + 50), who ends with Longest Road and Largest Army (+1 own,
//! -1 opponent's, 0 nobody's), and the search value 2 and 8 recorded
//! decisions later in the same game (the final value past the end).
//! Then per-action-type targets (`AUX_KINDS`): for each counted kind
//! (roads, settlements, cities, dev cards bought, knights played, maritime
//! trades, robber moves), how many the mover and the opponent play from this
//! decision to the end of the game (initial placements excluded), and the
//! final settlements, cities and victory point cards of both sides; each
//! count c squashed to c / (c + s) with s the count at which the target is
//! 0.5 (`AUX_SCALES`).
//!
//! Luck-adjusted results (`z_luck`): at every random event (roll, dev card
//! draw, steal) the network values each possible outcome; the actual
//! outcome's value minus the probability-weighted average is that event's
//! luck. A position's `z_luck` is its result minus the luck that came after
//! it. Every luck term has expectation zero whatever the network, so the
//! target's expectation is the true value, with much of the dice noise gone.

use std::sync::Arc;

use crate::encoding::{ActionSpace, ObservationSpace};
use crate::game::{Game, GameConfig};
use crate::mcts::{is_stochastic, terminal_values, Evaluator, MctsConfig, MctsPlayer, Utility};
use crate::nn::PolicyValueNet;
use crate::rng::Rng;
use crate::types::*;

#[derive(Clone, Debug)]
pub struct SelfPlayConfig {
    pub sims_full: u32,
    pub sims_fast: u32,
    /// Simulations for initial placements, which are always searched fully
    /// and recorded (0 = `sims_full`).
    pub sims_placement: u32,
    /// Share of other decisions searched fully and recorded.
    pub p_full: f32,
    /// Spend the full searches where the network is least sure: a decision
    /// is searched fully with probability proportional to its normalized
    /// policy entropy, with the gain steered so the share of full searches
    /// stays about `p_full` but moves to uncertain positions. Off = a coin.
    pub entropy_budget: bool,
    /// KL escalation: every non-placement decision gets the fast search
    /// first, and is searched fully when KL(fast search visits || prior)
    /// exceeds a threshold steered so about `p_full` of decisions escalate
    /// (the search's own surprise picks the positions; replaces the coin).
    pub kl_escalate: bool,
    /// With `kl_escalate`: also record fast-search decisions whose KL is
    /// above this fraction of the escalation threshold (KataGo's admission
    /// of surprising fast positions, with their fast-search targets); 0 = off.
    pub admit_kl: f32,
    /// Focus: a fully searched decision whose chosen move is a road,
    /// settlement (not an initial placement), robber move or bank trade is
    /// searched again with this many simulations and recorded with that
    /// deeper target, tagged (`focus`) for oversampling. 0 = off.
    pub focus_sims: u32,
    /// Each player's first this-many searched decisions are sampled in
    /// proportion to visits (exploration); later ones take the most visited.
    pub temp_moves: u32,
    pub root_noise: f32,
    pub dirichlet_alpha: f32,
    pub c_puct: f32,
    pub fpu_reduction: f32,
    /// A cheap search whose two most visited moves differ in value by less
    /// than this is redone fully and recorded (0 = off).
    pub escalate_gap: f32,
    /// Branch the game at the first position where a player has at least
    /// this many points (0 = off).
    pub endgame_vp: u32,
    /// Extra continuations played from that position.
    pub endgame_branches: u32,
    /// Extra continuations played from the position after the initial
    /// placements; placement decisions get the average result.
    pub opening_branches: u32,
    /// Replays per game from one of its opening settlement or road
    /// placements, with a different choice.
    pub placement_forks: u32,
    /// Temperature for choosing the forced placement (1 = the search's
    /// policy target as is).
    pub placement_temp: f32,
    /// Share of forks whose placement is drawn uniformly from the other
    /// legal choices instead of from the flattened search target.
    pub placement_uniform: f32,
    /// Placement forks only at settlement decisions (not roads), each
    /// settlement of the game forked once before any is forked twice.
    pub fork_settlements_only: bool,
    /// With a positive margin, a settlement fork's alternative is drawn
    /// uniformly from the other legal spots within this many pips of the best
    /// legal spot (the plausible ones) instead of from the flattened search
    /// target. Such forks, like uniform ones, are not sibling labels.
    pub placement_margin: f32,
    /// Root-prior flattening at settlement decisions (`MctsConfig::placement_flatten`),
    /// over the spots within `placement_margin` pips of the best (0 = off).
    pub placement_flatten: f32,
    /// Placement targets from short-horizon playouts instead of the PUCT
    /// search (0 = off): at every initial settlement decision (roads keep the
    /// search), each candidate spot within `placement_margin` pips of the best
    /// gets `placement_playouts` network-only playouts from the position
    /// after it until the game's turn counter reaches this value (or the
    /// game ends); the deciding player's value there, averaged, ranks the
    /// candidates. The policy target is softmax(value / `placement_horizon_tau`)
    /// over the candidates and the row's search value is the target-weighted
    /// mean. Measured on the 100 expert boards: the turn-5 value ranks first
    /// picks at Spearman 0.71 vs 0.63 for the one-ply value and 0.66 for
    /// game outcomes.
    pub placement_horizon_turns: u32,
    pub placement_playouts: u32,
    pub placement_horizon_tau: f32,
    /// With `placement_horizon_turns`: also take the policy target and the move
    /// from the playouts' softmax (the first on4 arm). Off = the playouts supply
    /// only the row's value (the horizon value label); the policy target and the
    /// move come from the placement search as before. Decoupled because the
    /// 12-playout softmax is a noisy policy target (on4: policy regret 6.6-8.6
    /// pts vs 6 with the search target) while the value head gained from the
    /// label; the search's values come from the opening network, so a better
    /// value head improves the search target too (the KataGo / MuZero split:
    /// policy from the strongest search, value from the least noisy estimate).
    pub placement_horizon_policy: bool,
    /// With `fork_settlements_only`: relative weight of each settlement decision
    /// (0..3 in placement order) for choosing which one a placement fork replays
    /// (all 1 = cycle through them; the learner sets these from the forks'
    /// regret, so more forks go where the network is wrong most).
    pub placement_fork_weights: [f32; 4],
    /// Move forks: after the game, it is replayed this many times from a
    /// fully searched mid-game road or robber decision of the main game,
    /// with a different choice (drawn like a placement fork, with
    /// `placement_temp` and `placement_uniform`): outcome data for the
    /// alternatives at the decisions the network gets wrong most.
    pub move_forks: u32,
    /// Temperature for a move fork's alternative (1 = the search target as
    /// is: mostly runner-up moves) and the share drawn uniformly instead.
    pub move_fork_temp: f32,
    pub move_fork_uniform: f32,
    /// Sibling-averaged value labels: a main-game position's result is the
    /// mean over every continuation that shares it (the main game, opening
    /// branches, and placement/move forks drawn from the search target),
    /// luck-adjusted each; de-noises the targets most, where forks branch.
    /// Uniformly drawn forks are exploration and are not averaged in.
    pub sibling_labels: bool,
    /// With a board pool (`GameConfig::board_pool`), the share of games
    /// played on a pool board; the rest get a random board as usual.
    pub pool_board_share: f32,
    /// The pool boards' seeds (same order), so their games report them.
    pub pool_seeds: Vec<u64>,
    /// Share of game slots played as a seat-swapped pair against a noisy
    /// opponent: the current network, playing a uniformly random legal
    /// move with probability `random_eps` (1 = a random bot) instead of
    /// searching. Its moves aren't recorded. Teaches finishing games against
    /// opponents unlike itself.
    pub random_share: f32,
    pub random_eps: f32,
    /// How finished games are valued, see `Utility`.
    pub utility: Utility,
    /// Gumbel root search over this many candidates, with Gumbel noise for
    /// exploration in place of Dirichlet noise and temperature (0 = PUCT).
    pub gumbel_m: u32,
    /// Rules (points to win, discard limit, robber, dice); seed and history
    /// settings are overridden per game.
    pub game: GameConfig,
}

/// Examples from a set of games, position-major.
#[derive(Clone, Debug, Default)]
pub struct SelfPlayData {
    /// `[n][obs_size]`
    pub obs: Vec<f32>,
    /// `[n][action_size]`, 1 = legal.
    pub mask: Vec<u8>,
    /// `[n][action_size]`: root visit share per action slot.
    pub pi: Vec<f32>,
    /// `[n]`: +1 if the mover went on to win, -1 if it lost, 0 if no winner.
    pub z: Vec<f32>,
    /// `[n]`: `z` minus the luck (for the mover) of random events after the
    /// position.
    pub z_luck: Vec<f32>,
    /// `[n][AUX_TARGETS]`, see the module docs.
    pub aux: Vec<f32>,
    /// `[n]`: the search's value of the position for the mover (visit-
    /// weighted mean over root actions).
    pub q: Vec<f32>,
    /// `[n]`: KL(policy target || network prior) over the root's moves, how
    /// much the search taught on this example.
    pub surprise: Vec<f32>,
    /// `[n]`: 1 if a player had at least `endgame_vp` points.
    pub endgame: Vec<u8>,
    /// `[n]`: 1 for an opening decision (initial settlement or road).
    pub opening: Vec<u8>,
    /// `[n]`: 1 for a focus decision (see `SelfPlayConfig::focus_sims`).
    pub focus: Vec<u8>,
    /// Games against the noisy opponent (`random_share`).
    pub random_games: u64,
    /// Games won by the first player to move, by the second, and with no winner.
    pub results: [u64; 3],
    /// Decisions made in all games (forced ones included).
    pub decisions: u64,
    /// Cheap searches redone fully because they were close calls.
    pub escalated: u64,
    /// Fast-search decisions recorded for their surprise (`admit_kl`).
    pub admitted: u64,
    /// Decisions searched again at `focus_sims`.
    pub focused: u64,
    /// Endgame and opening branches played (not counted in `results`).
    pub branches: u64,
    /// Games against a past network (see `self_play_with_pool`).
    pub pool_games: u64,
    /// Placement forks played (not counted in `results`).
    pub forks: u64,
    /// Per main game: (board seed, opening value error = mean |luck-adjusted
    /// result - search value| over its placement examples), for board
    /// prioritisation by the learner.
    pub boards: Vec<(u64, f32)>,
    /// `[n]`: the game each example comes from (one id per `play_game`:
    /// the main line with its branches and forks), for per-game weighting.
    pub game: Vec<u32>,
    /// Per fork: (mover's result in the main line, in the fork, the search
    /// target's probability of the forced alternative, kind: 0 = placement
    /// from the target, 1 = mid-game from the target, 2 / 3 = the same drawn
    /// uniformly): what the forks teach, read without a model.
    pub fork_pairs: Vec<(f32, f32, f32, u8)>,
    /// Sum over random events of |luck| (for the first seat), and the
    /// number of events.
    pub luck_abs: f64,
    pub luck_events: u64,
    /// Search simulations run (every search, forced moves excluded) and
    /// network evaluations made (search, luck accounting and entropy
    /// probes), for benchmarking.
    pub sims: u64,
    pub evals: u64,
}

impl SelfPlayData {
    pub fn len(&self) -> usize {
        self.z.len()
    }

    pub fn is_empty(&self) -> bool {
        self.z.is_empty()
    }

    fn append(&mut self, other: SelfPlayData) {
        self.obs.extend(other.obs);
        self.mask.extend(other.mask);
        self.pi.extend(other.pi);
        self.z.extend(other.z);
        self.z_luck.extend(other.z_luck);
        self.aux.extend(other.aux);
        self.q.extend(other.q);
        self.surprise.extend(other.surprise);
        self.endgame.extend(other.endgame);
        self.opening.extend(other.opening);
        self.focus.extend(other.focus);
        self.random_games += other.random_games;
        for (a, b) in self.results.iter_mut().zip(other.results) {
            *a += b;
        }
        self.decisions += other.decisions;
        self.escalated += other.escalated;
        self.admitted += other.admitted;
        self.focused += other.focused;
        self.branches += other.branches;
        self.pool_games += other.pool_games;
        self.forks += other.forks;
        self.boards.extend(other.boards);
        self.fork_pairs.extend(other.fork_pairs);
        self.game.extend(other.game);
        self.luck_abs += other.luck_abs;
        self.luck_events += other.luck_events;
        self.sims += other.sims;
        self.evals += other.evals;
    }
}

/// Gumbel MuZero's policy target (Danihelka et al. 2022): softmax of the
/// network's log prior plus a bonus growing with each action's value, where
/// unvisited actions get the mixed value estimate. Values in [-1, 1] map to
/// [0, 1] with a fixed scale rather than min-max over actions, since a
/// once-visited move's value is mostly dice luck.
/// Auxiliary targets per example: the 8 game-level ones, then the
/// per-action-type ones (see the module docs).
pub const AUX_TARGETS: usize = 8 + 2 * AUX_KINDS + 6;
/// Action kinds counted for the auxiliary targets (`aux_kind`).
pub const AUX_KINDS: usize = 7;
/// Squashing scales: the count at which a target is 0.5, per counted kind
/// (future counts), then for the final settlements, cities and victory point cards.
const AUX_SCALES: [f32; AUX_KINDS + 3] = [4.0, 2.0, 2.0, 3.0, 2.0, 4.0, 2.0, 3.0, 2.0, 1.0];

/// A count squashed into [0, 1): `c / (c + scale)`.
fn squash(c: f32, scale: f32) -> f32 {
    c / (c + scale)
}

/// Index of an action's kind among the counted ones, if it is counted.
fn aux_kind(kind: &ActionKind) -> Option<usize> {
    Some(match kind {
        ActionKind::BuildRoad(_) => 0,
        ActionKind::BuildSettlement(_) => 1,
        ActionKind::BuildCity(_) => 2,
        ActionKind::BuyDevelopmentCard => 3,
        ActionKind::PlayKnightCard => 4,
        ActionKind::MaritimeTrade { .. } => 5,
        ActionKind::MoveRobber { .. } => 6,
        _ => return None,
    })
}

/// Executes `action` with luck accounting and, outside the initial
/// placements, logs its kind for the per-action-type targets
/// (`k` = recorded examples so far: the action comes after examples `< k`).
fn execute_logged(
    game: &mut Game,
    action: Action,
    ctx: &mut Ctx,
    log: &mut Vec<(usize, usize, usize)>,
    k: usize,
) -> Result<Option<f32>, String> {
    if !game.state.is_initial_build_phase {
        if let Some(kind) = aux_kind(&action.kind) {
            log.push((game.state.seat(action.color), kind, k));
        }
    }
    execute_with_luck(game, action, &mut ctx.eval, ctx.reference, &ctx.cfg.utility)
}

/// Short-horizon playout targets for a placement decision (see
/// `SelfPlayConfig::placement_horizon_turns`). Returns the target over
/// `actions`, the target-weighted candidate value, the network's prior over
/// `actions` and the forwards spent.
fn horizon_targets(ctx: &mut Ctx, game: &Game, actions: &[Action], rng: &mut Rng) -> (Vec<f32>, f32, Vec<f32>, u64) {
    let cfg = &ctx.cfg;
    let me = game.state.current_color();
    let n_players = game.state.colors.len() as f32;
    let (prior, _) = ctx.eval.evaluate(game);
    let map = &game.state.board.map;
    let pips = |a: &Action| match a.kind {
        ActionKind::BuildSettlement(n) => Some(map.node_production[n as usize].iter().sum::<f64>() as f32 * 36.0),
        _ => None,
    };
    let best = actions.iter().filter_map(pips).fold(f32::MIN, f32::max);
    let candidate: Vec<bool> = actions
        .iter()
        .map(|a| match pips(a) {
            Some(p) => cfg.placement_margin <= 0.0 || p >= best - cfg.placement_margin,
            None => true,
        })
        .collect();
    let mut forwards = 1u64;
    let mut values = vec![f32::NAN; actions.len()];
    for (i, a) in actions.iter().enumerate() {
        if !candidate[i] {
            continue;
        }
        let mut sum = 0.0f32;
        for _ in 0..cfg.placement_playouts {
            let mut g = game.copy_without_history();
            g.state.rng = Rng::new(rng.next_u64());
            g.execute(*a).expect("candidate is legal");
            while !g.is_over() && g.state.num_turns < cfg.placement_horizon_turns {
                let legal = g.playable_actions.clone();
                let pick = if legal.len() == 1 {
                    legal[0]
                } else {
                    let (p, _) = ctx.eval.evaluate(&g);
                    forwards += 1;
                    legal[(0..p.len()).max_by(|&x, &y| p[x].total_cmp(&p[y])).unwrap()]
                };
                g.execute(pick).expect("legal");
            }
            sum += if g.is_over() {
                if g.winning_color() == Some(me) { 1.0 } else { -1.0 / (n_players - 1.0) }
            } else {
                let (_, v) = ctx.eval.evaluate(&g);
                forwards += 1;
                if g.state.current_color() == me { v } else { -v / (n_players - 1.0) }
            };
        }
        values[i] = sum / cfg.placement_playouts as f32;
    }
    let top = values.iter().copied().filter(|v| !v.is_nan()).fold(f32::MIN, f32::max);
    let mut target: Vec<f32> = values.iter().map(|v| if v.is_nan() { 0.0 } else { ((v - top) / cfg.placement_horizon_tau).exp() }).collect();
    let z: f32 = target.iter().sum();
    for t in target.iter_mut() {
        *t /= z;
    }
    let q = target.iter().zip(&values).filter(|(_, v)| !v.is_nan()).map(|(t, v)| t * v).sum();
    (target, q, prior, forwards)
}

fn completed_q_target(stats: &[(f32, u32, Option<f32>)], root_value: f32, bound: f32) -> Vec<f32> {
    const C_VISIT: f32 = 50.0;
    const C_SCALE: f32 = 0.1;
    // The paper's bonus grows with the top action's raw visit count, so the
    // target sharpens as simulations increase (doubling sims roughly doubles
    // it, amplifying noisy values). Scale by the top action's visit share
    // instead, at this many simulations, so sharpness doesn't depend on the
    // simulation count (identical to the paper's at 800).
    const REF_SIMS: f32 = 800.0;
    let n_sum: u32 = stats.iter().map(|s| s.1).sum();
    let max_n = stats.iter().map(|s| s.1).max().unwrap_or(0);
    let (mut pq, mut p) = (0.0f32, 0.0f32);
    for &(prior, _, q) in stats {
        if let Some(q) = q {
            pq += prior * q;
            p += prior;
        }
    }
    let v_mix = if p > 0.0 { (root_value + n_sum as f32 * pq / p) / (1.0 + n_sum as f32) } else { root_value };
    let top_share = if n_sum > 0 { max_n as f32 / n_sum as f32 } else { 0.0 };
    let scale = (C_VISIT + REF_SIMS * top_share) * C_SCALE;
    let logits: Vec<f32> = stats
        .iter()
        .map(|&(prior, _, q)| prior.max(1e-30).ln() + scale * (q.unwrap_or(v_mix).clamp(-bound, bound) + bound) / (2.0 * bound))
        .collect();
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.into_iter().map(|e| e / sum).collect()
}

fn uniform(rng: &mut Rng) -> f32 {
    (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32
}

/// Plays `games` 1v1 self-play games on `threads` threads. Game `i` depends
/// only on `seed` and `i`.
pub fn self_play(
    net: Arc<PolicyValueNet>,
    games: u64,
    seed: u64,
    threads: usize,
    cfg: &SelfPlayConfig,
) -> Result<SelfPlayData, String> {
    self_play_with_pool(net, &[], 0.0, games, seed, threads, cfg)
}

/// `self_play`, with a `pool_share` of game slots played as a pair against a
/// past network from `pool` on the same board, the current network taking
/// each seat once; only the current network's decisions are recorded, so
/// past networks are opponents, never teachers.
pub fn self_play_with_pool(
    net: Arc<PolicyValueNet>,
    pool: &[Arc<PolicyValueNet>],
    pool_share: f32,
    games: u64,
    seed: u64,
    threads: usize,
    cfg: &SelfPlayConfig,
) -> Result<SelfPlayData, String> {
    if let Some(p) = pool.iter().find(|p| p.obs_size != net.obs_size) {
        return Err(format!("pool network has {} inputs, the current one {}", p.obs_size, net.obs_size));
    }
    let parts = crate::parallel::par_map(games, threads, |i| {
        play_one(net.clone(), pool, pool_share, seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ i.wrapping_add(1), cfg)
    });
    let mut out = SelfPlayData::default();
    for part in parts {
        out.append(part?);
    }
    Ok(out)
}

/// Whether a search's two most visited moves have values within `gap`.
fn close_call(stats: &[(f32, u32, Option<f32>)], gap: f32) -> bool {
    let (mut first, mut second): (Option<(u32, f32)>, Option<(u32, f32)>) = (None, None);
    for &(_, n, q) in stats {
        let Some(q) = q else { continue };
        if first.is_none_or(|(m, _)| n > m) {
            second = first;
            first = Some((n, q));
        } else if second.is_none_or(|(m, _)| n > m) {
            second = Some((n, q));
        }
    }
    match (first, second) {
        (Some((_, a)), Some((_, b))) => (a - b).abs() < gap,
        _ => false,
    }
}

fn max_points(game: &Game) -> u32 {
    game.state.players.iter().map(|p| p.actual_victory_points).max().unwrap_or(0)
}

/// Value of `game` for `reference`: the result if it is over, else the
/// network's value for the player to move, negated for the other player.
fn value_for(game: &Game, reference: Color, network_value: f32, utility: &Utility) -> f32 {
    if game.is_over() {
        return terminal_values(game, utility)[reference.idx()];
    }
    if game.state.current_color() == reference { network_value } else { -network_value }
}

/// The possible outcomes of random `action` in `game`, as (forced result,
/// probability). Rolls are grouped by total, with one representative pair
/// per total (the observation only counts totals).
fn outcomes(game: &Game, action: &Action) -> Result<Vec<(ActionResult, f32)>, String> {
    let s = &game.state;
    let mut out: Vec<(ActionResult, f32)> = Vec::new();
    match action.kind {
        ActionKind::Roll => match &s.dice_deck {
            // One representative pair per total from the deck being drawn
            // from, with the deck's odds for this roller.
            Some(deck) => {
                let p = deck.probabilities(s.seat(action.color), s.colors.len());
                let next = deck.next_deck();
                for (a, b) in next.remaining() {
                    let t = (a + b) as usize;
                    if p[t] > 0.0 && !out.iter().any(|(r, _)| matches!(r, ActionResult::Dice(x, y) if (x + y) as usize == t)) {
                        out.push((ActionResult::Dice(a, b), p[t]));
                    }
                }
            }
            None => {
                for (a, b) in (1..=6).flat_map(|a| (1..=6).map(move |b| (a, b))) {
                    match out.iter_mut().find(|(r, _)| matches!(r, ActionResult::Dice(x, y) if x + y == a + b)) {
                        Some(o) => o.1 += 1.0 / 36.0,
                        None => out.push((ActionResult::Dice(a, b), 1.0 / 36.0)),
                    }
                }
            }
        },
        ActionKind::BuyDevelopmentCard => {
            let n = s.dev_deck.len() as f32;
            for d in DEV_CARDS {
                let k = s.dev_deck.iter().filter(|&&c| c == d).count();
                if k > 0 {
                    out.push((ActionResult::DevCard(d), k as f32 / n));
                }
            }
        }
        ActionKind::MoveRobber { victim: Some(v), .. } => {
            let hand = s.player(v).resources;
            let n: u32 = hand.iter().sum();
            for r in RESOURCES {
                if hand[r.idx()] > 0 {
                    out.push((ActionResult::Stolen(Some(r)), hand[r.idx()] as f32 / n as f32));
                }
            }
        }
        _ => return Err(format!("{action} is not a random event")),
    }
    Ok(out)
}

/// Whether forced result `r` and actual result `actual` are the same
/// outcome (rolls compare by total).
fn same_outcome(r: &ActionResult, actual: &ActionResult) -> bool {
    match (r, actual) {
        (ActionResult::Dice(a, b), ActionResult::Dice(c, d)) => a + b == c + d,
        _ => r == actual,
    }
}

/// KL(search visit distribution || network prior) at the root, over the
/// visited moves (how much the search disagreed with the network).
fn root_kl(stats: &[(f32, u32, Option<f32>)]) -> f32 {
    let total: f32 = stats.iter().map(|s| s.1 as f32).sum();
    let prior_sum: f32 = stats.iter().map(|s| s.0).sum();
    if total <= 0.0 || prior_sum <= 0.0 {
        return 0.0;
    }
    stats
        .iter()
        .filter(|s| s.1 > 0)
        .map(|&(p, n, _)| {
            let q = n as f32 / total;
            q * (q / (p / prior_sum).max(1e-30)).ln()
        })
        .sum::<f32>()
        .max(0.0)
}

/// Executes `action`; for a random event, also returns its luck for
/// `reference`: the actual outcome's value minus the expected value.
fn execute_with_luck(game: &mut Game, action: Action, eval: &mut Evaluator, reference: Color, utility: &Utility) -> Result<Option<f32>, String> {
    if !is_stochastic(&action, game) {
        game.execute(action)?;
        return Ok(None);
    }
    let options = outcomes(game, &action)?;
    let mut games = Vec::with_capacity(options.len());
    for (r, _) in &options {
        let mut g = game.copy_without_history();
        g.execute_unchecked(action, Some(*r))?;
        games.push(g);
    }
    let refs: Vec<&Game> = games.iter().collect();
    let values: Vec<f32> = eval
        .evaluate_batch(&refs)
        .into_iter()
        .zip(&games)
        .map(|((_, v), g)| value_for(g, reference, v, utility))
        .collect();
    let expected: f32 = values.iter().zip(&options).map(|(v, (_, p))| v * p).sum();
    let actual = game.execute(action)?.result;
    let k = options
        .iter()
        .position(|(r, _)| same_outcome(r, &actual))
        .ok_or_else(|| format!("outcome {actual:?} of {action} was not enumerated"))?;
    Ok(Some(values[k] - expected))
}

/// A copy of `snap` with fresh dice and a reshuffled dev deck, both hidden
/// from the players.
fn rerandomize(snap: &Game, rng: &mut Rng) -> Game {
    let mut g = snap.copy_without_history();
    g.state.rng = Rng::new(rng.next_u64());
    rng.shuffle(&mut g.state.dev_deck);
    g
}

/// One self-play game; or, for a `pool_share` of them, a pair against a
/// past network on the same board (same seats, same dice to start with),
/// the current network taking each seat once.
fn play_one(
    net: Arc<PolicyValueNet>,
    pool: &[Arc<PolicyValueNet>],
    pool_share: f32,
    seed: u64,
    cfg: &SelfPlayConfig,
) -> Result<SelfPlayData, String> {
    // A separate stream, so self-play without a pool is unchanged.
    let mut pick = Rng::new(seed ^ 0x9001_9001_9001_9001);
    if cfg.random_share > 0.0 {
        // Its own stream too, so runs without it are unchanged.
        let mut r = Rng::new(seed ^ 0x7A4D_0A1E_7A4D_0A1E);
        if uniform(&mut r) < cfg.random_share {
            let mut out = play_game(net.clone(), Some((COLORS[0], net.clone())), seed, seed, cfg, cfg.random_eps)?;
            out.append(play_game(net.clone(), Some((COLORS[1], net)), seed, seed ^ 0x5EA7_5EA7, cfg, cfg.random_eps)?);
            out.random_games += 2;
            out.pool_games = 0;
            return Ok(out);
        }
    }
    if pool.is_empty() || uniform(&mut pick) >= pool_share {
        return play_game(net, None, seed, seed, cfg, 0.0);
    }
    let past = pool[pick.below(pool.len())].clone();
    let mut out = play_game(net.clone(), Some((COLORS[0], past.clone())), seed, seed, cfg, 0.0)?;
    out.append(play_game(net, Some((COLORS[1], past)), seed, seed ^ 0x5EA7_5EA7, cfg, 0.0)?);
    Ok(out)
}

/// One game on board `game_seed`, with `past` (if any) playing one color,
/// a random legal move instead of a searched one with probability
/// `past_noise`; `search_seed` drives the players' randomness.
fn play_game(
    net: Arc<PolicyValueNet>,
    past: Option<(Color, Arc<PolicyValueNet>)>,
    game_seed: u64,
    search_seed: u64,
    cfg: &SelfPlayConfig,
    past_noise: f32,
) -> Result<SelfPlayData, String> {
    let colors = &COLORS[..2];
    let mut game_cfg = GameConfig { seed: Some(game_seed), record_history: false, ..cfg.game.clone() };
    let mut board_seed = game_seed;
    if let Some(pool) = game_cfg.board_pool.take() {
        // A pool board for `pool_board_share` of games (reported under its own
        // seed, so its error keeps being measured), else a random board.
        let mut r = Rng::new(game_seed ^ 0xB0A2_D5EE_D000_0001);
        if uniform(&mut r) < cfg.pool_board_share {
            let idx = r.below(pool.len());
            game_cfg.catan_map = Some(Arc::new(pool[idx].clone()));
            board_seed = cfg.pool_seeds.get(idx).copied().unwrap_or(game_seed);
        }
    }
    let game = Game::new(colors, &game_cfg);
    let mut rng = Rng::new(search_seed ^ 0x5E1F_D1A7_C0FF_EE01);
    let search = MctsConfig {
        root_noise: cfg.root_noise,
        dirichlet_alpha: cfg.dirichlet_alpha,
        c_puct: cfg.c_puct,
        fpu_reduction: cfg.fpu_reduction,
        utility: cfg.utility,
        gumbel_m: cfg.gumbel_m,
        gumbel_noise: cfg.gumbel_m > 0,
        placement_flatten: cfg.placement_flatten,
        placement_flatten_margin: cfg.placement_margin,
        ..Default::default()
    };
    // Indexed by `Color::idx` (seat order is shuffled per game).
    let mut players: Vec<MctsPlayer> = colors
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let n = match &past {
                Some((pc, pn)) if *pc == c => pn.clone(),
                _ => net.clone(),
            };
            MctsPlayer::new(c, n, search, rng.next_u64() ^ i as u64)
        })
        .collect();
    let map = &game.state.board.map;
    let spaces = (ObservationSpace::for_network(map, net.obs_size), ActionSpace::new(map));
    let mut ctx = Ctx { cfg, spaces, eval: Evaluator::new(net.clone()), reference: colors[0], past: past.as_ref().map(|(c, _)| *c), past_noise };

    let mut out = SelfPlayData { pool_games: past.is_some() as u64, ..Default::default() };
    let mut searched = [0u32; 2];
    let main = play_out(game, &mut players, &mut rng, &mut ctx, &mut searched, &mut out, true)?;
    out.results[main.seat] += 1;
    let main_len = out.z.len();
    // Continuations sharing a prefix of the main game, for sibling-averaged
    // labels: (last shared example index, final values, main luck at the
    // branch, luck inside the branch).
    let mut siblings: Vec<(usize, [f32; 4], f32, f32)> = Vec::new();

    if let Some(snap) = &main.endgame {
        for _ in 0..cfg.endgame_branches {
            play_out(rerandomize(snap, &mut rng), &mut players, &mut rng, &mut ctx, &mut searched, &mut out, false)?;
            out.branches += 1;
        }
    }
    if let (Some(opening), true) = (&main.opening, cfg.opening_branches > 0) {
        // Each continuation from the opening position (the main game first):
        // every player's final value, and the reference player's luck since
        // the opening.
        let mut ends = vec![(main.values, main.luck - opening.luck)];
        for _ in 0..cfg.opening_branches {
            let b = play_out(rerandomize(&opening.game, &mut rng), &mut players, &mut rng, &mut ctx, &mut searched, &mut out, false)?;
            ends.push((b.values, b.luck));
            if !opening.records.is_empty() {
                siblings.push((opening.records.len() - 1, b.values, opening.luck, b.luck));
            }
            out.branches += 1;
        }
        let n = ends.len() as f32;
        // The main game's examples come first in `out`; each gets its mover's
        // average result, raw and luck-adjusted.
        for (k, &(mover, before)) in opening.records.iter().enumerate() {
            let sign = if mover == ctx.reference { 1.0 } else { -1.0 };
            let raw = ends.iter().map(|(v, _)| v[mover.idx()]).sum::<f32>() / n;
            let adjusted = ends.iter().map(|(v, l)| v[mover.idx()] - sign * l).sum::<f32>() / n;
            out.z[k] = raw;
            out.z_luck[k] = adjusted - sign * (opening.luck - before);
        }
    }
    let is_settlement = |f: &ForkPoint| matches!(f.game.playable_actions[f.pick].kind, ActionKind::BuildSettlement(_));
    let placement_points: Vec<&ForkPoint> = main.forks.iter().filter(|f| !f.mid_game && (!cfg.fork_settlements_only || is_settlement(f))).collect();
    let move_points: Vec<&ForkPoint> = main.forks.iter().filter(|f| f.mid_game).collect();
    let plan = std::iter::repeat(&placement_points).take(cfg.placement_forks as usize).chain(std::iter::repeat(&move_points).take(cfg.move_forks as usize));
    // Placement forks walk the placement decisions from a random start, so each is forked once before any twice.
    let placement_start = rng.below(placement_points.len().max(1));
    for (fi, points) in plan.enumerate() {
        if points.is_empty() {
            continue;
        }
        let point = if fi < cfg.placement_forks as usize {
            let w = &cfg.placement_fork_weights;
            if cfg.fork_settlements_only && w.iter().any(|&x| (x - 1.0).abs() > 1e-6) {
                // Regret-weighted: more forks at the settlement decisions the network gets wrong most.
                let weights: Vec<f32> = points.iter().map(|p| if (p.settlement as usize) < 4 { w[p.settlement as usize].max(0.0) } else { 1.0 }).collect();
                points[crate::mcts::sample_index(&weights, &mut rng)]
            } else {
                points[(placement_start + fi) % points.len()]
            }
        } else {
            points[rng.below(points.len())]
        };
        let (temp, p_uniform) = if point.mid_game { (cfg.move_fork_temp, cfg.move_fork_uniform) } else { (cfg.placement_temp, cfg.placement_uniform) };
        let mut uniform_pick = uniform(&mut rng) < p_uniform;
        let mut weights: Vec<f32> = Vec::new();
        if !uniform_pick && !point.mid_game && cfg.placement_margin > 0.0 && is_settlement(point) {
            // The plausible alternatives: legal spots within the margin of the best one, by pips.
            let map = &point.game.state.board.map;
            let pips = |a: &Action| match a.kind {
                ActionKind::BuildSettlement(n) => map.node_production[n as usize].iter().sum::<f64>() as f32 * 36.0,
                _ => 0.0,
            };
            let best = point.game.playable_actions.iter().map(pips).fold(0.0, f32::max);
            weights = point.game.playable_actions.iter().enumerate().map(|(k, a)| (k != point.pick && pips(a) >= best - cfg.placement_margin) as u8 as f32).collect();
            if weights.iter().sum::<f32>() > 0.0 {
                uniform_pick = true; // not a sibling label, counted with the uniform forks
            } else {
                weights.clear();
            }
        }
        if weights.is_empty() {
            weights = point
                .target
                .iter()
                .enumerate()
                .map(|(k, &t)| match (k == point.pick, uniform_pick) {
                    (true, _) => 0.0,
                    (false, true) => 1.0,
                    (false, false) => t.max(1e-30).powf(1.0 / temp),
                })
                .collect();
        }
        let alt = crate::mcts::sample_index(&weights, &mut rng);
        let mut g = rerandomize(&point.game, &mut rng);
        let action = g.playable_actions[alt];
        g.execute(action)?;
        let mut fork_searched = point.searched;
        let b = play_out(g, &mut players, &mut rng, &mut ctx, &mut fork_searched, &mut out, false)?;
        if !uniform_pick {
            siblings.push((point.record_index, b.values, point.luck_before, b.luck));
        }
        let fork_mover = main.records[point.record_index].0;
        // kind: bit 0 mid-game, bit 1 uniform/plausible draw, bits 2.. = settlement ordinal + 1 (0 = not a settlement).
        let settlement_tag = if point.settlement < 4 { 4 * (point.settlement + 1) } else { 0 };
        out.fork_pairs.push((main.values[fork_mover.idx()], b.values[fork_mover.idx()], point.target[alt], point.mid_game as u8 + 2 * uniform_pick as u8 + settlement_tag));
        out.forks += 1;
    }
    // Opening value error of this board, for prioritised board sampling.
    let (mut err, mut n_open) = (0.0f32, 0u32);
    for i in 0..out.z.len() {
        if out.opening[i] == 1 {
            err += (out.z_luck[i] - out.q[i]).abs();
            n_open += 1;
        }
    }
    if n_open > 0 {
        out.boards.push((board_seed, err / n_open as f32));
    }
    out.game.resize(out.z.len(), (game_seed ^ (game_seed >> 32)) as u32 ^ (search_seed as u32).rotate_left(16));
    out.evals += players.iter().map(|p| p.evaluations()).sum::<u64>() + ctx.eval.evaluations();
    if cfg.sibling_labels {
        for k in 0..main_len {
            let (mover, before) = main.records[k];
            let sign = if mover == ctx.reference { 1.0 } else { -1.0 };
            let (mut raw, mut adjusted, mut n) = (main.values[mover.idx()], main.values[mover.idx()] - sign * (main.luck - before), 1.0f32);
            for &(last_shared, values, luck_at_branch, luck_in_branch) in &siblings {
                if last_shared >= k {
                    raw += values[mover.idx()];
                    adjusted += values[mover.idx()] - sign * ((luck_at_branch - before) + luck_in_branch);
                    n += 1.0;
                }
            }
            out.z[k] = raw / n;
            out.z_luck[k] = adjusted / n;
        }
    }
    Ok(out)
}

/// An opening settlement or road placement in the main game: the position, the search's
/// policy target over its playable actions, the move played, and each
/// seat's searched-decision count (for temperature) at that point.
struct ForkPoint {
    game: Game,
    target: Vec<f32>,
    pick: usize,
    searched: [u32; 2],
    /// A mid-game road/robber decision (`move_forks`) rather than a placement.
    mid_game: bool,
    /// Which settlement of the game this decision placed (0..3), 255 otherwise.
    settlement: u8,
    /// Index of this decision's example in the main game, and the luck before it.
    record_index: usize,
    luck_before: f32,
}

struct Ctx<'a> {
    cfg: &'a SelfPlayConfig,
    spaces: (ObservationSpace, ActionSpace),
    /// Values random events' outcomes.
    eval: Evaluator,
    /// Luck and results are tracked from this player's point of view.
    reference: Color,
    /// The seat a past network plays (its decisions aren't recorded).
    past: Option<Color>,
    /// Probability that seat plays a random legal move instead of searching.
    past_noise: f32,
}

/// The position after the initial placements, the luck before it, and the
/// examples recorded before it (mover, luck at the time).
struct Opening {
    game: Game,
    luck: f32,
    records: Vec<(Color, f32)>,
}

struct PlayOut {
    /// Winner's seat index (2 = no winner).
    seat: usize,
    /// Per recorded example: mover and the luck before it.
    records: Vec<(Color, f32)>,
    /// Each player's value of the finished game (`terminal_values`).
    values: [f32; 4],
    /// Total luck for `Ctx::reference`.
    luck: f32,
    endgame: Option<Game>,
    opening: Option<Opening>,
    forks: Vec<ForkPoint>,
}

/// Plays `game` to the end, appending its recorded examples (with their
/// results) to `out`. In the main game, saves the endgame and opening
/// positions for branching.
fn play_out(
    mut game: Game,
    players: &mut [MctsPlayer],
    rng: &mut Rng,
    ctx: &mut Ctx,
    searched: &mut [u32; 2],
    out: &mut SelfPlayData,
    main: bool,
) -> Result<PlayOut, String> {
    let cfg = ctx.cfg;
    let (o, a) = (ctx.spaces.0.len(), ctx.spaces.1.len());
    // Per recorded example: mover and the luck accumulated before it.
    let mut records: Vec<(Color, f32)> = Vec::new();
    // Per recorded example: turn number (for turns left).
    let mut turns: Vec<u32> = Vec::new();
    // Counted actions: (seat, kind, recorded examples before it).
    let mut action_log: Vec<(usize, usize, usize)> = Vec::new();
    let q_base = out.q.len();
    let mut luck = 0.0f32;
    // `entropy_budget`: full-search probability per unit of normalized
    // entropy, steered to average `p_full` (typical entropy is around 0.3).
    let mut entropy_gain = cfg.p_full / 0.3;
    // `kl_escalate`: the KL above which a fast search escalates, steered so
    // about `p_full` of decisions do.
    let mut kl_thresh = 0.1f32;
    let mut endgame_snap = None;
    let mut opening = None;
    let mut forks = Vec::new();
    let mut settlements_placed: u8 = 0;
    let mut legal = vec![false; a];
    while !game.is_over() {
        let endgame = cfg.endgame_vp > 0 && max_points(&game) >= cfg.endgame_vp;
        if main && endgame && endgame_snap.is_none() {
            endgame_snap = Some(game.copy_without_history());
        }
        if main && opening.is_none() && !game.state.is_initial_build_phase {
            opening = Some(Opening { game: game.copy_without_history(), luck, records: records.clone() });
        }
        out.decisions += 1;
        let actions = game.playable_actions.clone();
        let mover = game.state.current_color();
        let seat = game.state.seat(mover);
        if actions.len() == 1 {
            if let Some(l) = execute_logged(&mut game, actions[0], ctx, &mut action_log, records.len())? {
                luck += l;
                out.luck_abs += l.abs() as f64;
                out.luck_events += 1;
            }
            continue;
        }
        // The noisy opponent: a random legal move (not recorded).
        if ctx.past == Some(mover) && ctx.past_noise > 0.0 && uniform(rng) < ctx.past_noise {
            let a = actions[rng.below(actions.len())];
            if let Some(l) = execute_logged(&mut game, a, ctx, &mut action_log, records.len())? {
                luck += l;
                out.luck_abs += l.abs() as f64;
                out.luck_events += 1;
            }
            continue;
        }
        // Take an outright win without searching (search could prefer an
        // inflated estimate of playing on); the decision is not recorded.
        if let Some(i) = crate::mcts::immediate_win(&game) {
            if let Some(l) = execute_logged(&mut game, actions[i], ctx, &mut action_log, records.len())? {
                luck += l;
                out.luck_abs += l.abs() as f64;
                out.luck_events += 1;
            }
            continue;
        }
        let placing = game.state.is_initial_build_phase;
        let mut full = placing
            || if cfg.kl_escalate {
                false // decided after the fast search, from its surprise
            } else if cfg.entropy_budget {
                // Normalized entropy of the network's policy over the legal moves.
                let (priors, _) = ctx.eval.evaluate(&game);
                let h: f32 = -priors.iter().filter(|&&p| p > 0.0).map(|&p| p * p.ln()).sum::<f32>();
                let h_norm = h / (actions.len() as f32).ln().max(1e-6);
                // The gain is steered so the probabilities average `p_full`
                // (a fixed ratio to the mean entropy would fall short, since
                // the most uncertain positions are capped at 1).
                let p = (entropy_gain * h_norm).clamp(0.0, 1.0);
                let full = uniform(rng) < p;
                // Steered on what happened, so the cap's lost mass is made up.
                entropy_gain = (entropy_gain + 0.2 * (cfg.p_full - full as u8 as f32)).max(0.0);
                full
            } else {
                uniform(rng) < cfg.p_full
            };
        let player = &mut players[mover.idx()];
        player.config.simulations = match (placing, full) {
            (true, _) if cfg.sims_placement > 0 => cfg.sims_placement,
            (_, true) => cfg.sims_full,
            _ => cfg.sims_fast,
        };
        player.config.root_noise = if full && cfg.gumbel_m == 0 { cfg.root_noise } else { 0.0 };
        // Settlement decisions of the opening by short-horizon playouts (`placement_horizon_turns`): no tree
        // search. The opening roads keep the search: a road's horizon values differ by less than the playout
        // noise, so their softmax came out nearly uniform (top probability 0.46 over 3 roads) and taught
        // arbitrary roads, which cost 3 points against Ham within 10 iterations on Nov 11.
        let horizon = (placing && cfg.placement_horizon_turns > 0 && matches!(actions[0].kind, ActionKind::BuildSettlement(_)))
            .then(|| horizon_targets(ctx, &game, &actions, rng));
        if let Some((_, _, _, forwards)) = &horizon {
            out.sims += forwards;
        }
        // The policy target and the move come from the playouts only with `placement_horizon_policy`;
        // otherwise the search runs as usual and the playouts supply the row's value alone.
        let policy_from_horizon = horizon.is_some() && cfg.placement_horizon_policy;
        let mut visits = match &horizon {
            Some((target, _, _, _)) if policy_from_horizon => target.iter().map(|t| (t * 1000.0).round() as u32).collect(),
            _ => {
                let v = player.search(&game)?;
                out.sims += player.config.simulations as u64;
                v
            }
        };
        // Surprise of the fast search: KL(visit distribution || network prior).
        let mut fast_kl = 0.0f32;
        if !full && (cfg.kl_escalate || cfg.admit_kl > 0.0) {
            fast_kl = root_kl(&player.root_stats().0);
        }
        let mut admitted = false;
        if !full && cfg.kl_escalate {
            if fast_kl > kl_thresh {
                full = true;
                out.escalated += 1;
                kl_thresh += 0.05 * (1.0 - cfg.p_full);
                player.config.simulations = cfg.sims_full;
                player.config.root_noise = if cfg.gumbel_m == 0 { cfg.root_noise } else { 0.0 };
                visits = player.search(&game)?;
                out.sims += player.config.simulations as u64;
            } else {
                kl_thresh = (kl_thresh - 0.05 * cfg.p_full).max(0.0);
                admitted = cfg.admit_kl > 0.0 && fast_kl > cfg.admit_kl * kl_thresh;
            }
        }
        if !full && cfg.escalate_gap > 0.0 && close_call(&player.root_stats().0, cfg.escalate_gap) {
            full = true;
            out.escalated += 1;
            player.config.simulations = cfg.sims_full;
            player.config.root_noise = if cfg.gumbel_m == 0 { cfg.root_noise } else { 0.0 };
            visits = player.search(&game)?;
            out.sims += player.config.simulations as u64;
        }
        // Focus: the decisions the network gets wrong most (roads, robber,
        // settlements, trades) get a deeper target when searched fully.
        let mut focused = false;
        if full && !placing && cfg.focus_sims > player.config.simulations {
            let best = (0..visits.len()).max_by_key(|&j| visits[j]).unwrap_or(0);
            let is_focus = matches!(
                actions[best].kind,
                ActionKind::BuildRoad(_) | ActionKind::BuildSettlement(_) | ActionKind::MoveRobber { .. } | ActionKind::MaritimeTrade { .. }
            );
            if is_focus {
                player.config.simulations = cfg.focus_sims;
                visits = player.search(&game)?;
                out.sims += player.config.simulations as u64;
                focused = true;
                out.focused += 1;
            }
        }
        let total: u32 = visits.iter().sum();
        let mut fork_target = None;

        if (full || admitted) && total > 0 && ctx.past != Some(mover) {
            out.admitted += admitted as u64;
            let (obs_space, act_space) = &ctx.spaces;
            let start = out.obs.len();
            out.obs.resize(start + o, 0.0);
            obs_space.encode_into(&game.state, mover, &mut out.obs[start..]);
            act_space.fill_mask(&game.state, &actions, &mut legal);
            out.mask.extend(legal.iter().map(|&m| m as u8));
            let (target, mut q_value, priors): (Vec<f32>, f32, Vec<f32>) = match &horizon {
                Some((target, q, prior, _)) if policy_from_horizon => (target.clone(), *q, prior.clone()),
                _ => {
                    let (stats, root_value) = player.root_stats();
                    let target = completed_q_target(&stats, root_value, cfg.utility.bound());
                    let visited: u32 = stats.iter().map(|s| s.1).sum();
                    let q_sum: f32 = stats.iter().filter_map(|&(_, n, q)| q.map(|q| q * n as f32)).sum();
                    (target, if visited > 0 { q_sum / visited as f32 } else { root_value }, stats.iter().map(|s| s.0).collect())
                }
            };
            if let Some((_, q, _, _)) = &horizon {
                q_value = *q; // the horizon value label, whichever head chose the policy target
            }
            out.q.push(q_value);
            let prior_sum: f32 = priors.iter().sum();
            out.surprise.push(
                priors
                    .iter()
                    .zip(&target)
                    .filter(|&(_, &t)| t > 0.0)
                    .map(|(&p, &t)| t * (t / (p / prior_sum).max(1e-30)).ln())
                    .sum::<f32>()
                    .max(0.0),
            );
            out.endgame.push(endgame as u8);
            out.opening.push(placing as u8);
            out.focus.push(focused as u8);
            let start = out.pi.len();
            out.pi.resize(start + a, 0.0);
            for (action, &t) in actions.iter().zip(&target) {
                let slot = act_space.encode(&game.state, action).ok_or("unencodable root action")?;
                out.pi[start + slot] += t;
            }
            records.push((mover, luck));
            turns.push(game.state.num_turns);
            let opening = placing
                && actions.len() > 1
                && matches!(actions[0].kind, ActionKind::BuildSettlement(_) | ActionKind::BuildRoad(_));
            if main && cfg.placement_forks > 0 && opening {
                fork_target = Some((target, false));
            } else if main && cfg.move_forks > 0 && full && !placing {
                let best = (0..visits.len()).max_by_key(|&j| visits[j]).unwrap_or(0);
                if matches!(actions[best].kind, ActionKind::BuildRoad(_) | ActionKind::MoveRobber { .. }) {
                    fork_target = Some((target, true));
                }
            }
        }

        let pick = if let Some(choice) = player.gumbel_choice().filter(|_| !policy_from_horizon) {
            choice
        } else if searched[seat] < cfg.temp_moves && total > 0 {
            let weights: Vec<f32> = visits.iter().map(|&v| v as f32).collect();
            crate::mcts::sample_index(&weights, rng)
        } else {
            (0..visits.len()).max_by_key(|&k| visits[k]).unwrap()
        };
        let settles = placing && matches!(actions[pick].kind, ActionKind::BuildSettlement(_));
        if let Some((target, mid_game)) = fork_target {
            let settlement = if !mid_game && settles { settlements_placed } else { 255 };
            forks.push(ForkPoint { game: game.copy_without_history(), target, pick, searched: *searched, mid_game, settlement, record_index: records.len() - 1, luck_before: luck });
        }
        if settles {
            settlements_placed = settlements_placed.saturating_add(1);
        }
        searched[seat] += 1;
        if let Some(l) = execute_logged(&mut game, actions[pick], ctx, &mut action_log, records.len())? {
            luck += l;
            out.luck_abs += l.abs() as f64;
            out.luck_events += 1;
        }
    }

    let winner = game.winning_color();
    // Each mover's own value: not zero-sum when the game times out.
    let values = terminal_values(&game, &cfg.utility);
    for &(mover, before) in &records {
        let sign = if mover == ctx.reference { 1.0 } else { -1.0 };
        out.z.push(values[mover.idx()]);
        out.z_luck.push(values[mover.idx()] - sign * (luck - before));
    }
    let s = &game.state;
    let vps = game.vps_to_win as f32;
    let army = s.largest_army().map(|(c, _)| c);
    let road = s.longest_road_color();
    let holder = |c: Option<Color>, mover: Color| match c {
        Some(c) if c == mover => 1.0,
        Some(_) => -1.0,
        None => 0.0,
    };
    // Counted actions after each recorded example: suffix sums over the log.
    let mut future = vec![[[0u16; AUX_KINDS]; 2]; records.len() + 1];
    for &(seat, kind, k) in &action_log {
        future[k.min(records.len())][seat][kind] += 1;
    }
    for k in (0..records.len()).rev() {
        for seat in 0..2 {
            for kind in 0..AUX_KINDS {
                future[k][seat][kind] += future[k + 1][seat][kind];
            }
        }
    }
    let finals = |c: Color| {
        let p = s.player(c);
        let vp = DevCard::VictoryPoint as usize;
        [p.settlements.len() as f32, p.cities.len() as f32, (p.dev_cards[vp] + p.played_dev_cards[vp]) as f32]
    };
    for (k, &(mover, _)) in records.iter().enumerate() {
        let own = s.player(mover).actual_victory_points as f32;
        let opp = s.players.iter().filter(|p| p.color != mover).map(|p| p.actual_victory_points).max().unwrap_or(0) as f32;
        let later = |d: usize| match records.get(k + d) {
            Some(&(m, _)) => out.q[q_base + k + d] * if m == mover { 1.0 } else { -1.0 },
            None => values[mover.idx()],
        };
        out.aux.extend_from_slice(&[
            (own - opp) / vps,
            own / vps,
            opp / vps,
            {
                let left = (s.num_turns - turns[k]) as f32;
                left / (left + 50.0)
            },
            holder(road, mover),
            holder(army, mover),
            later(2),
            later(8),
        ]);
        let own_seat = s.seat(mover);
        let opp_color = s.players.iter().map(|p| p.color).find(|&c| c != mover).ok_or("no opponent")?;
        // The example's own action and everything after it: logged with k + 1.
        for kind in 0..AUX_KINDS {
            out.aux.push(squash(future[k + 1][own_seat][kind] as f32, AUX_SCALES[kind]));
            out.aux.push(squash(future[k + 1][1 - own_seat][kind] as f32, AUX_SCALES[kind]));
        }
        let (fo, fp) = (finals(mover), finals(opp_color));
        for i in 0..3 {
            out.aux.push(squash(fo[i], AUX_SCALES[AUX_KINDS + i]));
            out.aux.push(squash(fp[i], AUX_SCALES[AUX_KINDS + i]));
        }
    }
    Ok(PlayOut { seat: winner.map_or(2, |w| game.state.seat(w)), records, values, luck, endgame: endgame_snap, opening, forks })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::players::from_key;
    use crate::state::DiceMode;

    /// Every random event's listed outcomes sum to probability 1 and
    /// include what actually happens (with every dice mode).
    #[test]
    fn random_event_outcomes_are_complete() {
        for dice in [DiceMode::Fair, DiceMode::Deck, DiceMode::OpenHex] {
            let mut events = 0;
            for seed in 0..10 {
                let mut g = Game::new(&COLORS[..2], &GameConfig { seed: Some(seed), dice, ..Default::default() });
                let mut bots: Vec<_> = COLORS[..2].iter().map(|&c| from_key("R", c, seed).unwrap()).collect();
                while !g.is_over() {
                    let mover = g.state.current_color();
                    let actions = g.playable_actions.clone();
                    let action = bots[mover.idx()].decide(&g, &actions);
                    if is_stochastic(&action, &g) {
                        let options = outcomes(&g, &action).unwrap();
                        let total: f32 = options.iter().map(|o| o.1).sum();
                        assert!((total - 1.0).abs() < 1e-5, "{action}: probabilities sum to {total}");
                        let actual = g.execute(action).unwrap().result;
                        assert!(options.iter().any(|(r, _)| same_outcome(r, &actual)), "{actual:?} not listed");
                        events += 1;
                    } else {
                        g.execute(action).unwrap();
                    }
                }
            }
            assert!(events > 100);
        }
    }
    /// Finished-game values: zero-sum in 1v1, the winner's above 1 by its
    /// margin and speed terms, the speed bonus shrinking with game length.
    #[test]
    fn utility_values_of_finished_games() {
        let u = Utility { margin: 0.1, speed: 0.2, speed_turns: 1000.0 };
        let mut checked = 0;
        for seed in 0..20 {
            // Random bots need longer than the turn limit to finish most games.
            let config = GameConfig { seed: Some(seed), turns_limit: None, ..Default::default() };
            let mut g = Game::new(&COLORS[..2], &config);
            let mut bots: Vec<_> = COLORS[..2].iter().map(|&c| from_key("R", c, seed).unwrap()).collect();
            g.play(&mut bots).unwrap();
            let Some(w) = g.winning_color() else { continue };
            let l = COLORS[..2].iter().copied().find(|&c| c != w).unwrap();
            let v = terminal_values(&g, &u);
            assert!((v[w.idx()] + v[l.idx()]).abs() < 1e-6, "zero-sum");
            let s = &g.state;
            let margin = (s.player(w).actual_victory_points as f32 - s.player(l).actual_victory_points as f32) / g.vps_to_win as f32;
            let quick = 0.2 * (-(s.num_turns as f32) / 1000.0).exp();
            assert!((v[w.idx()] - (1.0 + quick + 0.1 * margin)).abs() < 1e-5);
            let slower = Utility { speed_turns: s.num_turns as f32 / 3.0, ..u };
            let bonus = terminal_values(&g, &slower)[w.idx()] - (1.0 + 0.1 * margin);
            assert!(bonus > 0.0 && bonus < quick, "a later win keeps a smaller bonus");
            checked += 1;
        }
        assert!(checked > 10);
    }
}
