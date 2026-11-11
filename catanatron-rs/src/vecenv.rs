//! Batched environment for reinforcement learning.
//!
//! Runs `num_envs` games. In each, one *learner* seat is controlled from
//! outside through action indices (see `encoding::ActionSpace`). Other seats
//! are either Rust bots, which act inside `step`, or *external* seats
//! (opponent key `"SELF"`), which the caller also controls — used for
//! self-play against frozen copies of the learner.
//!
//! With external seats, one learner step is:
//!   1. `act(actions)`: apply learner actions, let bots play.
//!   2. While `pending_opponents()` is non-empty: encode those positions
//!      with `observe_envs`, pick moves, `act_opponents(envs, actions)`.
//!   3. `finish()`: rewards and episode ends; finished games reset.
//!   4. Resolve pending opponents again (a new game may start with them).
//!   5. `observe()`: learner observations for the next decision.
//!
//! `step` does all of this when there are no external seats.

use std::sync::Arc;

use rayon::prelude::*;

use crate::encoding::{ActionSpace, ObservationSpace};
use crate::nn::{PolicyValueNet, Scratch};
use crate::game::{Game, GameConfig, TURNS_LIMIT};
use crate::map::{build_map, MapType, NumberPlacement};
use crate::players::{from_key, Player};
use crate::rng::Rng;
use crate::types::*;

const LEARNER: Color = Color::Red;
/// Opponent key for a seat controlled by the caller.
pub const EXTERNAL: &str = "SELF";

#[derive(Clone, Debug)]
pub struct VecEnvConfig {
    pub num_envs: usize,
    /// Keys for the other seats (1 to 3 of them): bot keys like `"F"`, or
    /// `"SELF"` for a caller-controlled seat.
    pub opponents: Vec<String>,
    pub seed: u64,
    pub threads: usize,
    pub map_type: MapType,
    pub vps_to_win: u32,
    pub discard_limit: u32,
    pub friendly_robber: crate::state::FriendlyRobber,
    pub dice: crate::state::DiceMode,
    /// How number tokens are laid out (see `map::NumberPlacement`).
    pub number_placement: NumberPlacement,
    /// Learner observations include opponents' hands by resource
    /// (`encoding::HAND_FEATURES`).
    pub opponent_hands: bool,
    /// Learner observations include development-card tracking
    /// (`encoding::DEV_FEATURES`).
    pub opponent_devs: bool,
    /// Learner observations include tile numbers, first player and
    /// opponents' this-turn dev activity (`ObsFeatures::extras`).
    pub extra_info: bool,
    /// Learner observations include settlement spots and trade rates
    /// (`ObsFeatures::spots`).
    pub spot_features: bool,
    /// Learner observations include reach, contention and affordability
    /// (`ObsFeatures::reach`).
    pub reach_features: bool,
    /// Spot-structure block (see `ObsFeatures::structure`).
    pub structure_features: bool,
    /// Extra reward per victory point the learner gains (0 = win/loss only).
    pub vp_reward: f32,
}

impl Default for VecEnvConfig {
    fn default() -> Self {
        VecEnvConfig {
            num_envs: 64,
            opponents: vec!["F".into()],
            seed: 0,
            threads: crate::parallel::default_threads(),
            map_type: MapType::Base,
            vps_to_win: 10,
            discard_limit: 7,
            friendly_robber: crate::state::FriendlyRobber::Off,
            dice: crate::state::DiceMode::Fair,
            number_placement: NumberPlacement::OfficialSpiral,
            opponent_hands: false,
            opponent_devs: false,
            extra_info: false,
            spot_features: false,
            reach_features: false,
            structure_features: false,
            vp_reward: 0.0,
        }
    }
}

/// Observations and masks for a set of envs, row-major.
#[derive(Clone, Debug, Default)]
pub struct Observation {
    /// `rows * obs_size`
    pub obs: Vec<f32>,
    /// `rows * action_size`, 1 = legal.
    pub mask: Vec<u8>,
}

/// Per-env outcome of one learner step.
#[derive(Clone, Debug, Default)]
pub struct StepResult {
    pub reward: Vec<f32>,
    /// 1 if the episode ended on this step (the next obs is a new game's).
    pub done: Vec<u8>,
    /// For ended episodes: 1 if the learner won.
    pub won: Vec<u8>,
    /// For ended episodes: 1 if it hit the turn limit with no winner.
    pub truncated: Vec<u8>,
}

/// `step`'s combined output.
#[derive(Clone, Debug, Default)]
pub struct StepOutput {
    pub obs: Vec<f32>,
    pub mask: Vec<u8>,
    pub reward: Vec<f32>,
    pub done: Vec<u8>,
    pub won: Vec<u8>,
    pub truncated: Vec<u8>,
}

struct Env {
    game: Game,
    /// Seat bots indexed like `COLORS[1..]`; `None` = external seat.
    bots: Vec<Option<Box<dyn Player>>>,
    /// Learner's actual VPs at the last `finish`, for VP reward.
    last_vps: u32,
    /// Games played in this slot: with the slot index and base seed it
    /// determines every game, independent of threading.
    episode: u64,
    /// Samples the learner's moves in `rollout`.
    rng: Rng,
}

fn episode_seed(base: u64, index: usize, episode: u64) -> u64 {
    let mut x = base
        ^ (index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ episode.wrapping_mul(0xD1B5_4A32_D192_ED03);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Starts game number `episode` in slot `index` and plays bots up to the
/// first decision of the learner or an external seat.
fn make_episode(config: &VecEnvConfig, index: usize, episode: u64) -> Result<Env, String> {
    let seed = episode_seed(config.seed, index, episode);
    let n = config.opponents.len() + 1;
    let game = Game::new(
        &COLORS[..n],
        &GameConfig {
            seed: Some(seed),
            map_type: config.map_type,
            vps_to_win: config.vps_to_win,
            discard_limit: config.discard_limit,
            friendly_robber: config.friendly_robber,
            dice: config.dice,
            number_placement: config.number_placement,
            record_history: false,
            ..Default::default()
        },
    );
    let bots = config
        .opponents
        .iter()
        .zip(&COLORS[1..])
        .enumerate()
        .map(|(i, (k, &c))| {
            if k == EXTERNAL {
                return Ok(None);
            }
            from_key(k, c, seed.wrapping_mul(31).wrapping_add(i as u64 + 1))
                .map(Some)
                .ok_or_else(|| format!("can't create bot {k}"))
        })
        .collect::<Result<_, String>>()?;
    let mut env = Env { game, bots, last_vps: 0, episode, rng: Rng::new(seed ^ 0x5EED) };
    env.advance()?;
    Ok(env)
}

/// Reward since the last call, and (won, truncated) if the game is over.
fn settle(env: &mut Env, vp_reward: f32) -> (f32, Option<(bool, bool)>) {
    let vps = env.game.state.player(LEARNER).actual_victory_points;
    let mut reward = vp_reward * (vps as f32 - env.last_vps as f32);
    env.last_vps = vps;
    if let Some(w) = env.game.winning_color() {
        let won = w == LEARNER;
        reward += if won { 1.0 } else { -1.0 };
        return (reward, Some((won, false)));
    }
    if env.game.state.num_turns >= TURNS_LIMIT {
        return (reward, Some((false, true)));
    }
    (reward, None)
}

impl Env {
    /// Lets bots play until the learner or an external seat must decide,
    /// or the game ends.
    fn advance(&mut self) -> Result<(), String> {
        while !self.game.is_over() {
            let c = self.game.state.current_color();
            if c == LEARNER {
                return Ok(());
            }
            match &mut self.bots[c.idx() - 1] {
                Some(bot) => {
                    let a = bot.decide(&self.game, &self.game.playable_actions);
                    self.game.execute(a)?;
                }
                None => return Ok(()),
            }
        }
        Ok(())
    }

    fn waiting_on_external(&self) -> bool {
        !self.game.is_over() && self.game.state.current_color() != LEARNER
    }

    fn apply(&mut self, space: &ActionSpace, action: usize) -> Result<(), String> {
        if action >= space.len() {
            return Err(format!("action {action} out of range"));
        }
        let a = space.decode(&self.game.state, action);
        self.game.execute(a).map_err(|e| format!("illegal action {action} ({a}): {e}"))?;
        self.advance()
    }
}

pub struct VecEnv {
    config: VecEnvConfig,
    envs: Vec<Env>,
    obs_space: ObservationSpace,
    action_space: ActionSpace,
    /// Learner network used by `rollout`.
    policy: Option<Arc<PolicyValueNet>>,
    /// Persistent rayon thread pool for rollout/act parallelism.
    pool: Arc<rayon::ThreadPool>,
}

/// Learner experience from `rollout`, env-major: index `[env][step]`.
#[derive(Clone, Debug, Default)]
pub struct Rollout {
    pub num_envs: usize,
    pub steps: usize,
    /// `[env][step][obs_size]`
    pub obs: Vec<f32>,
    /// `[env][step][action_size]`, 1 = legal.
    pub mask: Vec<u8>,
    pub action: Vec<u32>,
    /// Log-probability of the chosen action under the rollout policy.
    pub logp: Vec<f32>,
    pub value: Vec<f32>,
    pub reward: Vec<f32>,
    /// 1 if the episode ended after this step.
    pub done: Vec<u8>,
    pub won: Vec<u8>,
    pub truncated: Vec<u8>,
    /// `[env]`: value of the position after the last step (for bootstrapping).
    pub last_value: Vec<f32>,
}

/// Every board of a map type shares a topology, so one encoder serves all.
fn spaces(config: &VecEnvConfig) -> (ObservationSpace, ActionSpace) {
    let map = build_map(config.map_type, NumberPlacement::OfficialSpiral, &mut Rng::new(0));
    let features = crate::encoding::ObsFeatures {
        dice: config.dice != crate::state::DiceMode::Fair,
        dice_history: config.dice == crate::state::DiceMode::OpenHex,
        hands: config.opponent_hands,
        devs: config.opponent_devs,
        extras: config.extra_info,
        spots: config.spot_features,
        reach: config.reach_features,
        structure: config.structure_features,
    };
    (ObservationSpace::with_features(&map, features), ActionSpace::new(&map))
}

fn check_opponents(opponents: &[String]) -> Result<(), String> {
    if !(1..=3).contains(&opponents.len()) {
        return Err("need 1 to 3 opponents".into());
    }
    for k in opponents {
        if k != EXTERNAL {
            from_key(k, Color::Blue, 0).ok_or_else(|| format!("unknown bot key {k}"))?;
        }
    }
    Ok(())
}

/// Runs `f` on each env (with its index) using the given rayon pool.
fn par_envs<T: Send>(
    pool: &rayon::ThreadPool,
    envs: &mut [Env],
    f: impl Fn(usize, &mut Env) -> T + Sync,
) -> Vec<T> {
    pool.install(|| {
        envs.par_iter_mut()
            .enumerate()
            .map(|(i, env)| f(i, env))
            .collect()
    })
}

impl VecEnv {
    pub fn new(config: VecEnvConfig) -> Result<VecEnv, String> {
        check_opponents(&config.opponents)?;
        let (obs_space, action_space) = spaces(&config);
        let envs = (0..config.num_envs).map(|i| make_episode(&config, i, 0)).collect::<Result<_, _>>()?;
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(config.threads.max(1))
                .build()
                .map_err(|e| format!("rayon pool: {e}"))?,
        );
        Ok(VecEnv { envs, obs_space, action_space, policy: None, pool, config })
    }

    pub fn obs_size(&self) -> usize {
        self.obs_space.len()
    }

    pub fn action_size(&self) -> usize {
        self.action_space.len()
    }

    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    pub fn observation_names(&self) -> Vec<String> {
        self.obs_space.names(&self.envs[0].game.state)
    }

    /// Swap opponents; takes effect as each game resets.
    pub fn set_opponents(&mut self, opponents: Vec<String>) -> Result<(), String> {
        if opponents.len() != self.config.opponents.len() {
            return Err("number of opponents can't change".into());
        }
        check_opponents(&opponents)?;
        self.config.opponents = opponents;
        Ok(())
    }

    /// Learner observations and masks for every env.
    pub fn observe(&self) -> Observation {
        let all: Vec<usize> = (0..self.envs.len()).collect();
        self.observe_envs(&all)
    }

    /// Observations for the given envs, from the view of whoever must
    /// decide there (the learner, or an external seat).
    pub fn observe_envs(&self, idxs: &[usize]) -> Observation {
        let (o, a) = (self.obs_size(), self.action_size());
        let mut out = Observation { obs: vec![0.0; idxs.len() * o], mask: vec![0; idxs.len() * a] };
        let mut mask = vec![false; a];
        for (row, &i) in idxs.iter().enumerate() {
            let s = &self.envs[i].game.state;
            self.obs_space.encode_into(s, s.current_color(), &mut out.obs[row * o..(row + 1) * o]);
            self.action_space.fill_mask(s, &self.envs[i].game.playable_actions, &mut mask);
            for (dst, &m) in out.mask[row * a..(row + 1) * a].iter_mut().zip(&mask) {
                *dst = m as u8;
            }
        }
        out
    }

    /// Envs waiting on an external seat.
    pub fn pending_opponents(&self) -> Vec<usize> {
        (0..self.envs.len()).filter(|&i| self.envs[i].waiting_on_external()).collect()
    }

    /// Applies one learner action per env, then lets bots play.
    pub fn act(&mut self, actions: &[usize]) -> Result<(), String> {
        if actions.len() != self.envs.len() {
            return Err(format!("expected {} actions, got {}", self.envs.len(), actions.len()));
        }
        if let Some(i) = self.pending_opponents().first() {
            return Err(format!("env {i} is waiting on an external opponent"));
        }
        let space = &self.action_space;
        par_envs(&self.pool, &mut self.envs, |i, env| {
            if env.game.is_over() {
                return Ok(()); // finish() hasn't reset it yet
            }
            env.apply(space, actions[i]).map_err(|e| format!("env {i}: {e}"))
        })
        .into_iter()
        .collect()
    }

    /// Applies external-seat actions to the given (pending) envs.
    pub fn act_opponents(&mut self, idxs: &[usize], actions: &[usize]) -> Result<(), String> {
        if idxs.len() != actions.len() {
            return Err("envs and actions differ in length".into());
        }
        let mut per_env = vec![None; self.envs.len()];
        for (&i, &a) in idxs.iter().zip(actions) {
            if i >= self.envs.len() || !self.envs[i].waiting_on_external() {
                return Err(format!("env {i} is not waiting on an external opponent"));
            }
            per_env[i] = Some(a);
        }
        let space = &self.action_space;
        par_envs(&self.pool, &mut self.envs, |i, env| match per_env[i] {
            Some(a) => env.apply(space, a).map_err(|e| format!("env {i}: {e}")),
            None => Ok(()),
        })
        .into_iter()
        .collect()
    }

    /// Rewards and episode ends since the last `finish`; resets ended games.
    pub fn finish(&mut self) -> Result<StepResult, String> {
        if let Some(i) = self.pending_opponents().first() {
            return Err(format!("env {i} is waiting on an external opponent"));
        }
        let n = self.envs.len();
        let mut r = StepResult { reward: vec![0.0; n], done: vec![0; n], won: vec![0; n], truncated: vec![0; n] };
        for i in 0..n {
            let (reward, ended) = settle(&mut self.envs[i], self.config.vp_reward);
            r.reward[i] = reward;
            if let Some((won, truncated)) = ended {
                r.done[i] = 1;
                r.won[i] = won as u8;
                r.truncated[i] = truncated as u8;
                let next = self.envs[i].episode + 1;
                self.envs[i] = make_episode(&self.config, i, next)?;
            }
        }
        Ok(r)
    }

    /// One learner step when there are no external seats.
    pub fn step(&mut self, actions: &[usize]) -> Result<StepOutput, String> {
        if self.config.opponents.iter().any(|k| k == EXTERNAL) {
            return Err("step() can't drive external seats; use act/act_opponents/finish".into());
        }
        self.act(actions)?;
        let r = self.finish()?;
        let o = self.observe();
        Ok(StepOutput { obs: o.obs, mask: o.mask, reward: r.reward, done: r.done, won: r.won, truncated: r.truncated })
    }
}

impl VecEnv {
    /// Sets the learner network that `rollout` samples moves from.
    pub fn set_policy(&mut self, net: Arc<PolicyValueNet>) -> Result<(), String> {
        if net.obs_size != self.obs_size() || net.action_size != self.action_size() {
            return Err(format!(
                "network is {}x{}, environment needs {}x{}",
                net.obs_size, net.action_size, self.obs_size(), self.action_size()
            ));
        }
        self.policy = Some(net);
        Ok(())
    }

    /// Plays `steps` learner decisions in every env, sampling moves from the
    /// policy set with `set_policy`, into newly allocated buffers.
    pub fn rollout(&mut self, steps: usize) -> Result<Rollout, String> {
        let (n, o, a) = (self.envs.len(), self.obs_size(), self.action_size());
        let mut r = Rollout {
            num_envs: n,
            steps,
            obs: vec![0.0; n * steps * o],
            mask: vec![0; n * steps * a],
            action: vec![0; n * steps],
            logp: vec![0.0; n * steps],
            value: vec![0.0; n * steps],
            reward: vec![0.0; n * steps],
            done: vec![0; n * steps],
            won: vec![0; n * steps],
            truncated: vec![0; n * steps],
            last_value: vec![0.0; n],
        };
        self.rollout_into(
            steps,
            RolloutBuffers {
                obs: &mut r.obs,
                mask: &mut r.mask,
                action: &mut r.action,
                logp: &mut r.logp,
                value: &mut r.value,
                reward: &mut r.reward,
                done: &mut r.done,
                won: &mut r.won,
                truncated: &mut r.truncated,
                last_value: &mut r.last_value,
            },
        )?;
        Ok(r)
    }

    /// Like `rollout`, but writes into caller-owned buffers (laid out like
    /// `Rollout`), so nothing is allocated or copied after play. Bots
    /// (including network bots) play the other seats. Each env runs on one
    /// thread start to finish and writes only its own rows. Finished games
    /// reset automatically.
    pub fn rollout_into(&mut self, steps: usize, out: RolloutBuffers) -> Result<(), String> {
        if self.config.opponents.iter().any(|k| k == EXTERNAL) {
            return Err("rollout() can't drive external seats".into());
        }
        let net = self.policy.clone().ok_or("call set_policy before rollout")?;
        let (n, o, a) = (self.envs.len(), self.obs_size(), self.action_size());
        let expect = |name: &str, len: usize, want: usize| {
            if len == want { Ok(()) } else { Err(format!("{name} buffer has {len} elements, need {want}")) }
        };
        expect("obs", out.obs.len(), n * steps * o)?;
        expect("mask", out.mask.len(), n * steps * a)?;
        expect("action", out.action.len(), n * steps)?;
        expect("logp", out.logp.len(), n * steps)?;
        expect("value", out.value.len(), n * steps)?;
        expect("reward", out.reward.len(), n * steps)?;
        expect("done", out.done.len(), n * steps)?;
        expect("won", out.won.len(), n * steps)?;
        expect("truncated", out.truncated.len(), n * steps)?;
        expect("last_value", out.last_value.len(), n)?;
        if steps == 0 || n == 0 {
            return Err("rollout needs at least one env and one step".into());
        }

        let rows: Vec<EnvRows> = out
            .obs
            .chunks_mut(steps * o)
            .zip(out.mask.chunks_mut(steps * a))
            .zip(out.action.chunks_mut(steps))
            .zip(out.logp.chunks_mut(steps))
            .zip(out.value.chunks_mut(steps))
            .zip(out.reward.chunks_mut(steps))
            .zip(out.done.chunks_mut(steps))
            .zip(out.won.chunks_mut(steps))
            .zip(out.truncated.chunks_mut(steps))
            .zip(out.last_value.iter_mut())
            .map(|(((((((((obs, mask), action), logp), value), reward), done), won), truncated), last_value)| EnvRows {
                obs, mask, action, logp, value, reward, done, won, truncated, last_value,
            })
            .collect();

        let config = &self.config;
        let (obs_space, act_space) = (&self.obs_space, &self.action_space);
        let envs = &mut self.envs;
        let results: Vec<Result<(), String>> = self.pool.install(|| {
            envs.par_iter_mut()
                .zip(rows)
                .enumerate()
                .map(|(i, (env, r))| {
                    let mut scratch = Scratch::default();
                    let mut logits = Vec::new();
                    let mut legal = vec![false; a];
                    let mut probs: Vec<(usize, f32)> = Vec::with_capacity(64);
                    let mut weights: Vec<f32> = Vec::with_capacity(64);
                    for t in 0..steps {
                        debug_assert_eq!(env.game.state.current_color(), LEARNER);
                        let obs = &mut r.obs[t * o..(t + 1) * o];
                        obs_space.encode_into(&env.game.state, LEARNER, obs);
                        act_space.fill_mask(&env.game.state, &env.game.playable_actions, &mut legal);
                        for (dst, &m) in r.mask[t * a..(t + 1) * a].iter_mut().zip(&legal) {
                            *dst = m as u8;
                        }
                        r.value[t] = net.forward(obs, &mut scratch, &mut logits);

                        // Sample from the softmax over legal slots.
                        probs.clear();
                        probs.extend(legal.iter().enumerate().filter(|(_, &m)| m).map(|(k, _)| (k, logits[k])));
                        if probs.is_empty() {
                            return Err(format!("env {i}: no legal action"));
                        }
                        let max = probs.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);
                        let mut sum = 0.0;
                        for p in probs.iter_mut() {
                            p.1 = (p.1 - max).exp();
                            sum += p.1;
                        }
                        weights.clear();
                        weights.extend(probs.iter().map(|p| p.1));
                        let pick = crate::mcts::sample_index(&weights, &mut env.rng);
                        let (slot, w) = probs[pick];
                        r.action[t] = slot as u32;
                        r.logp[t] = (w / sum).ln();

                        env.apply(act_space, slot).map_err(|e| format!("env {i}: {e}"))?;
                        let (reward, ended) = settle(env, config.vp_reward);
                        r.reward[t] = reward;
                        r.done[t] = 0;
                        r.won[t] = 0;
                        r.truncated[t] = 0;
                        if let Some((won, truncated)) = ended {
                            r.done[t] = 1;
                            r.won[t] = won as u8;
                            r.truncated[t] = truncated as u8;
                            *env = make_episode(config, i, env.episode + 1).map_err(|e| format!("env {i}: {e}"))?;
                        }
                    }
                    let mut obs = vec![0.0; o];
                    obs_space.encode_into(&env.game.state, LEARNER, &mut obs);
                    *r.last_value = net.forward(&obs, &mut scratch, &mut logits);
                    Ok(())
                })
                .collect()
        });
        results.into_iter().collect()
    }
}

/// Caller-owned output of `rollout_into`, laid out like `Rollout`.
pub struct RolloutBuffers<'a> {
    pub obs: &'a mut [f32],
    pub mask: &'a mut [u8],
    pub action: &'a mut [u32],
    pub logp: &'a mut [f32],
    pub value: &'a mut [f32],
    pub reward: &'a mut [f32],
    pub done: &'a mut [u8],
    pub won: &'a mut [u8],
    pub truncated: &'a mut [u8],
    pub last_value: &'a mut [f32],
}

/// One env's rows of a `RolloutBuffers`.
struct EnvRows<'a> {
    obs: &'a mut [f32],
    mask: &'a mut [u8],
    action: &'a mut [u32],
    logp: &'a mut [f32],
    value: &'a mut [f32],
    reward: &'a mut [f32],
    done: &'a mut [u8],
    won: &'a mut [u8],
    truncated: &'a mut [u8],
    last_value: &'a mut f32,
}
