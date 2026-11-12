//! Python module `catanatron_rs`: exposes `VecEnv` and a match runner.
//! Arrays cross the boundary as raw little-endian bytes; the `catanatron_env`
//! Python helper turns them into numpy arrays without copying per element.
//! `rollout`, whose output is large, instead writes into caller buffers.

use pyo3::buffer::{Element, PyBuffer};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use catanatron::map::MapType;

mod mirror;
use catanatron::vecenv::{RolloutBuffers, StepOutput, VecEnv, VecEnvConfig};

/// Game rules from Python arguments; friendly_robber is "off", "actual"
/// (catanatron: points incl. hidden VP cards) or "public" (Colonist); dice
/// is "fair", "deck" (each combination once per 36 rolls) or "openhex"
/// (OpenHex's balanced dice); numbers is "spiral" (fixed corner),
/// "spiral-random-corner" or "random".
fn game_rules(map: &str, vps_to_win: u32, discard_limit: u32, friendly_robber: &str, dice: &str, numbers: &str) -> PyResult<catanatron::GameConfig> {
    use catanatron::FriendlyRobber;
    let friendly_robber = match friendly_robber {
        "off" => FriendlyRobber::Off,
        "actual" => FriendlyRobber::ActualPoints,
        "public" => FriendlyRobber::PublicPoints,
        other => return Err(PyValueError::new_err(format!("friendly_robber must be off/actual/public, not {other}"))),
    };
    use catanatron::state::DiceMode;
    let dice = match dice {
        "fair" => DiceMode::Fair,
        "deck" => DiceMode::Deck,
        "openhex" => DiceMode::OpenHex,
        other => return Err(PyValueError::new_err(format!("dice must be fair/deck/openhex, not {other}"))),
    };
    use catanatron::map::NumberPlacement;
    let number_placement = match numbers {
        "spiral" => NumberPlacement::OfficialSpiral,
        "spiral-random-corner" => NumberPlacement::SpiralRandomCorner,
        "random" => NumberPlacement::Random,
        other => return Err(PyValueError::new_err(format!("numbers must be spiral/spiral-random-corner/random, not {other}"))),
    };
    Ok(catanatron::GameConfig { map_type: map_type(map)?, vps_to_win, discard_limit, friendly_robber, dice, number_placement, ..Default::default() })
}

/// `rules` with the board for every game, if given: `seed:<n>` fixes the
/// board a game with seed n draws under `rules` (as the board screener
/// numbers them); `seeds:<n>,<m>,...` plays each game on one of those boards,
/// drawn uniformly; `group:<resource>` draws random boards whose scarcest
/// resource is that one (`catanatron::map::scarcest_resource`); anything else
/// is an OpenHex board in `mirror::openhex_map_spec` form, fixed.
fn with_board(board: Option<&str>, rules: catanatron::GameConfig) -> PyResult<catanatron::GameConfig> {
    let Some(board) = board else { return Ok(rules) };
    // The same draw as `Game::new` with that seed.
    let seed_map = |n: &str| -> PyResult<std::sync::Arc<catanatron::map::CatanMap>> {
        let seed: u64 = n.trim().parse().map_err(|_| PyValueError::new_err(format!("bad board seed {n:?} in {board:.40}")))?;
        let mut rng = catanatron::rng::Rng::new(seed);
        Ok(catanatron::map::build_map(rules.map_type, rules.number_placement, &mut rng))
    };
    if let Some(n) = board.strip_prefix("seed:") {
        return Ok(catanatron::GameConfig { catan_map: Some(seed_map(n)?), ..rules });
    }
    if let Some(list) = board.strip_prefix("seeds:") {
        let maps = list.split(',').map(|n| seed_map(n).map(|m| (*m).clone())).collect::<PyResult<Vec<_>>>()?;
        return Ok(catanatron::GameConfig { board_pool: Some(maps.into()), ..rules });
    }
    if let Some(r) = board.strip_prefix("group:") {
        return Ok(catanatron::GameConfig { board_group: Some(mirror::resource(r)?), ..rules });
    }
    Ok(catanatron::GameConfig { catan_map: Some(mirror::openhex_map_spec(board)?), ..rules })
}

/// Shares of games on selected real boards (see `catanatron::map::BoardMix`).
fn board_mix(scarce_ows: f32, port_start: f32) -> PyResult<catanatron::map::BoardMix> {
    if scarce_ows < 0.0 || port_start < 0.0 || scarce_ows + port_start > 1.0 {
        return Err(PyValueError::new_err("board shares must be >= 0 and sum to at most 1"));
    }
    Ok(catanatron::map::BoardMix { scarce_ows, port_start })
}

fn map_type(name: &str) -> PyResult<MapType> {
    match name.to_uppercase().as_str() {
        "BASE" => Ok(MapType::Base),
        "MINI" => Ok(MapType::Mini),
        "TOURNAMENT" => Ok(MapType::Tournament),
        other => Err(PyValueError::new_err(format!("unknown map type {other}"))),
    }
}

/// Raw bytes of a numeric slice, copied once. Python reads them with
/// little-endian dtypes, matching every platform this targets.
fn raw_bytes<'py, T: Copy>(py: Python<'py>, v: &[T]) -> Bound<'py, PyBytes> {
    #[cfg(target_endian = "big")]
    compile_error!("raw_bytes assumes a little-endian target");
    // SAFETY: plain numeric data (f32/u32/u8), no padding; length in bytes
    // is len * size_of::<T>().
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    PyBytes::new(py, bytes)
}

fn f32_bytes<'py>(py: Python<'py>, v: &[f32]) -> Bound<'py, PyBytes> {
    raw_bytes(py, v)
}

type StepTuple<'py> = (
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
);

fn to_tuple<'py>(py: Python<'py>, o: &StepOutput) -> StepTuple<'py> {
    (
        f32_bytes(py, &o.obs),
        PyBytes::new(py, &o.mask),
        f32_bytes(py, &o.reward),
        PyBytes::new(py, &o.done),
        PyBytes::new(py, &o.won),
        PyBytes::new(py, &o.truncated),
    )
}

/// Batched games: one learner seat per game, bots in the other seats.
#[pyclass(name = "VecEnv", unsendable)]
struct PyVecEnv {
    inner: VecEnv,
}

#[pymethods]
impl PyVecEnv {
    #[new]
    #[pyo3(signature = (num_envs, opponents, seed=0, threads=0, map_type="BASE", vps_to_win=10, vp_reward=0.0,
                        discard_limit=7, friendly_robber="off", dice="fair", numbers="spiral", opponent_hands=false, opponent_devs=false, extra_info=false, spot_features=false, reach_features=false, structure_features=false, relation_features=false, award_features=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_envs: usize,
        opponents: Vec<String>,
        seed: u64,
        threads: usize,
        map_type: &str,
        vps_to_win: u32,
        vp_reward: f32,
        discard_limit: u32,
        friendly_robber: &str,
        dice: &str,
        numbers: &str,
        opponent_hands: bool,
        opponent_devs: bool,
        extra_info: bool,
        spot_features: bool,
        reach_features: bool,
        structure_features: bool,
        relation_features: bool,
        award_features: bool,
    ) -> PyResult<Self> {
        let rules = game_rules(map_type, vps_to_win, discard_limit, friendly_robber, dice, numbers)?;
        let config = VecEnvConfig {
            num_envs,
            opponents,
            seed,
            threads: if threads == 0 { catanatron::parallel::default_threads() } else { threads },
            map_type: self::map_type(map_type)?,
            vps_to_win,
            vp_reward,
            discard_limit,
            friendly_robber: rules.friendly_robber,
            dice: rules.dice,
            number_placement: rules.number_placement,
            opponent_hands,
            opponent_devs,
            extra_info,
            spot_features,
            reach_features,
            structure_features,
            relation_features,
            award_features,
        };
        Ok(PyVecEnv { inner: VecEnv::new(config).map_err(PyValueError::new_err)? })
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.inner.num_envs()
    }

    #[getter]
    fn obs_size(&self) -> usize {
        self.inner.obs_size()
    }

    #[getter]
    fn action_size(&self) -> usize {
        self.inner.action_size()
    }

    fn observation_names(&self) -> Vec<String> {
        self.inner.observation_names()
    }

    fn set_opponents(&mut self, opponents: Vec<String>) -> PyResult<()> {
        self.inner.set_opponents(opponents).map_err(PyValueError::new_err)
    }

    /// (obs, mask) of the learner in every env.
    fn observe<'py>(&self, py: Python<'py>) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>) {
        let o = self.inner.observe();
        (f32_bytes(py, &o.obs), PyBytes::new(py, &o.mask))
    }

    /// (obs, mask) for the given envs, from the view of whoever decides there.
    fn observe_envs<'py>(&self, py: Python<'py>, envs: Vec<usize>) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        if envs.iter().any(|&i| i >= self.inner.num_envs()) {
            return Err(PyValueError::new_err("env index out of range"));
        }
        let o = self.inner.observe_envs(&envs);
        Ok((f32_bytes(py, &o.obs), PyBytes::new(py, &o.mask)))
    }

    /// Envs waiting on an external ("SELF") seat.
    fn pending_opponents(&self) -> Vec<usize> {
        self.inner.pending_opponents()
    }

    /// Applies one learner action per env, then lets bots play.
    fn act(&mut self, actions: Vec<i64>) -> PyResult<()> {
        let actions = to_usize(actions)?;
        self.inner.act(&actions).map_err(PyValueError::new_err)
    }

    /// Applies external-seat actions to the given pending envs.
    fn act_opponents(&mut self, envs: Vec<usize>, actions: Vec<i64>) -> PyResult<()> {
        let actions = to_usize(actions)?;
        self.inner.act_opponents(&envs, &actions).map_err(PyValueError::new_err)
    }

    /// (reward, done, won, truncated); resets ended games.
    fn finish<'py>(&mut self, py: Python<'py>) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let r = self.inner.finish().map_err(PyValueError::new_err)?;
        Ok((f32_bytes(py, &r.reward), PyBytes::new(py, &r.done), PyBytes::new(py, &r.won), PyBytes::new(py, &r.truncated)))
    }

    /// Sets the learner network (CTNN bytes, see export.py) for `rollout`.
    fn set_policy(&mut self, net: &[u8]) -> PyResult<()> {
        let net = catanatron::nn::PolicyValueNet::from_bytes(net).map_err(PyValueError::new_err)?;
        self.inner.set_policy(std::sync::Arc::new(net)).map_err(PyValueError::new_err)
    }

    /// Plays `steps` learner decisions per env in Rust, writing into the
    /// given C-contiguous, writable, non-overlapping buffers (e.g. numpy
    /// arrays), env-major ([env][step]...): obs (f32), mask (u8), action
    /// (u32), logp, value, reward (f32), done, won, truncated (u8),
    /// last_value (f32, [env]). Releases the GIL while playing; the buffers
    /// must not be touched from Python until it returns.
    #[allow(clippy::too_many_arguments)]
    fn rollout(
        &mut self,
        py: Python<'_>,
        steps: usize,
        obs: PyBuffer<f32>,
        mask: PyBuffer<u8>,
        action: PyBuffer<u32>,
        logp: PyBuffer<f32>,
        value: PyBuffer<f32>,
        reward: PyBuffer<f32>,
        done: PyBuffer<u8>,
        won: PyBuffer<u8>,
        truncated: PyBuffer<u8>,
        last_value: PyBuffer<f32>,
    ) -> PyResult<()> {
        let spans = [
            span(&obs, "obs")?,
            span(&mask, "mask")?,
            span(&action, "action")?,
            span(&logp, "logp")?,
            span(&value, "value")?,
            span(&reward, "reward")?,
            span(&done, "done")?,
            span(&won, "won")?,
            span(&truncated, "truncated")?,
            span(&last_value, "last_value")?,
        ];
        let mut sorted = spans;
        sorted.sort_unstable_by_key(|s| s.0);
        if sorted.windows(2).any(|w| w[0].0 + w[0].1 > w[1].0) {
            return Err(PyValueError::new_err("rollout buffers overlap"));
        }
        // SAFETY: each buffer is writable and C-contiguous (checked by
        // `span`), none overlap, and the PyBuffers keep them alive until
        // after the call.
        let out = unsafe {
            RolloutBuffers {
                obs: as_mut_slice(&obs),
                mask: as_mut_slice(&mask),
                action: as_mut_slice(&action),
                logp: as_mut_slice(&logp),
                value: as_mut_slice(&value),
                reward: as_mut_slice(&reward),
                done: as_mut_slice(&done),
                won: as_mut_slice(&won),
                truncated: as_mut_slice(&truncated),
                last_value: as_mut_slice(&last_value),
            }
        };
        let inner = &mut self.inner;
        py.detach(|| inner.rollout_into(steps, out)).map_err(PyValueError::new_err)
    }

    /// act + finish + observe, for envs without external seats:
    /// (obs, mask, reward, done, won, truncated).
    fn step<'py>(&mut self, py: Python<'py>, actions: Vec<i64>) -> PyResult<StepTuple<'py>> {
        let actions = to_usize(actions)?;
        let out = self.inner.step(&actions).map_err(PyValueError::new_err)?;
        Ok(to_tuple(py, &out))
    }
}

/// (address, byte length) of a buffer Rust may write into.
fn span<T: Element>(buf: &PyBuffer<T>, name: &str) -> PyResult<(usize, usize)> {
    if buf.readonly() {
        return Err(PyValueError::new_err(format!("{name} buffer is read-only")));
    }
    if !buf.is_c_contiguous() {
        return Err(PyValueError::new_err(format!("{name} buffer is not C-contiguous")));
    }
    Ok((buf.buf_ptr() as usize, buf.len_bytes()))
}

/// # Safety
/// `buf` must be writable and C-contiguous, with no other live reference
/// to its memory while the slice is used.
unsafe fn as_mut_slice<'a, T: Element>(buf: &'a PyBuffer<T>) -> &'a mut [T] {
    std::slice::from_raw_parts_mut(buf.buf_ptr() as *mut T, buf.item_count())
}

fn to_usize(actions: Vec<i64>) -> PyResult<Vec<usize>> {
    actions
        .into_iter()
        .map(|a| usize::try_from(a).map_err(|_| PyValueError::new_err("negative action")))
        .collect()
}

/// Plays `num_games` bot-vs-bot games in parallel; returns wins per seat
/// (in the order of `players`) and the number of games with no winner.
#[pyfunction]
#[pyo3(signature = (players, num_games, seed=0, threads=0, map_type="BASE", vps_to_win=10, discard_limit=7, friendly_robber="off", dice="fair", numbers="spiral", scarce_ows_share=0.0, port_start_share=0.0, board=None))]
#[allow(clippy::too_many_arguments)]
fn play_bots(
    players: Vec<String>,
    num_games: u64,
    seed: u64,
    threads: usize,
    map_type: &str,
    vps_to_win: u32,
    discard_limit: u32,
    friendly_robber: &str,
    dice: &str,
    numbers: &str,
    scarce_ows_share: f32,
    port_start_share: f32,
    board: Option<&str>,
) -> PyResult<(Vec<u64>, u64)> {
    let mix = board_mix(scarce_ows_share, port_start_share)?;
    let games = run_bots(&players, num_games, seed, threads, map_type, vps_to_win, discard_limit, friendly_robber, dice, numbers, mix, board)?;
    let mut wins = vec![0; players.len()];
    let mut none = 0;
    for (winner, _) in games {
        match winner {
            Some(p) => wins[p] += 1,
            None => none += 1,
        }
    }
    Ok((wins, none))
}

/// Like `play_bots`, split by seat: returns (games, wins), each indexed
/// [player][seat] (seat 0 moves first).
#[pyfunction]
#[pyo3(signature = (players, num_games, seed=0, threads=0, map_type="BASE", vps_to_win=10, discard_limit=7, friendly_robber="off", dice="fair", numbers="spiral"))]
#[allow(clippy::too_many_arguments)]
fn play_bots_by_seat(
    players: Vec<String>,
    num_games: u64,
    seed: u64,
    threads: usize,
    map_type: &str,
    vps_to_win: u32,
    discard_limit: u32,
    friendly_robber: &str,
    dice: &str,
    numbers: &str,
) -> PyResult<(Vec<Vec<u64>>, Vec<Vec<u64>>)> {
    let n = players.len();
    let games = run_bots(&players, num_games, seed, threads, map_type, vps_to_win, discard_limit, friendly_robber, dice, numbers, Default::default(), None)?;
    let (mut played, mut wins) = (vec![vec![0; n]; n], vec![vec![0; n]; n]);
    for (winner, seats) in games {
        for (seat, &p) in seats.iter().enumerate() {
            played[p][seat] += 1;
            if winner == Some(p) {
                wins[p][seat] += 1;
            }
        }
    }
    Ok((played, wins))
}

/// Plays the games for `play_bots*`: per game, the winning player's index
/// (in `players`) and the players' indices in seat order.
#[allow(clippy::too_many_arguments)]
fn run_bots(
    players: &[String],
    num_games: u64,
    seed: u64,
    threads: usize,
    map_type: &str,
    vps_to_win: u32,
    discard_limit: u32,
    friendly_robber: &str,
    dice: &str,
    numbers: &str,
    mix: catanatron::map::BoardMix,
    board: Option<&str>,
) -> PyResult<Vec<(Option<usize>, Vec<usize>)>> {
    let base = game_rules(map_type, vps_to_win, discard_limit, friendly_robber, dice, numbers)?;
    let rules = catanatron::GameConfig { board_mix: mix, ..with_board(board, base)? };
    use catanatron::players::from_key;
    use catanatron::{Color, Game, GameConfig, COLORS};
    if !(2..=4).contains(&players.len()) {
        return Err(PyValueError::new_err("need 2 to 4 players"));
    }
    for k in players {
        from_key(k, COLORS[0], 0).ok_or_else(|| PyValueError::new_err(format!("unknown bot {k}")))?;
    }
    let colors = &COLORS[..players.len()];
    let threads = if threads == 0 { catanatron::parallel::default_threads() } else { threads };
    let results = catanatron::parallel::par_map(num_games, threads, |i| {
        let s = seed.wrapping_add(i);
        let mut bots: Vec<_> = players
            .iter()
            .zip(colors)
            .enumerate()
            .map(|(j, (k, &c))| from_key(k, c, s.wrapping_mul(31).wrapping_add(j as u64 + 1)).unwrap())
            .collect();
        let mut g = Game::new(colors, &GameConfig { seed: Some(s), record_history: false, ..rules.clone() });
        let winner = g.play(&mut bots)?;
        let index = |c: Color| colors.iter().position(|&x| x == c).unwrap();
        Ok::<_, String>((winner.map(index), g.state.colors.iter().map(|&c| index(c)).collect()))
    });
    results.into_iter().map(|r| r.map_err(PyValueError::new_err)).collect()
}

/// Runs an exported network on one observation (float32 bytes):
/// returns (logits as float32 bytes, value). For checking exports.
#[pyfunction]
fn net_forward<'py>(py: Python<'py>, path: &str, obs: &[u8]) -> PyResult<(Bound<'py, PyBytes>, f32)> {
    let net = catanatron::mcts::load_net(path).map_err(PyValueError::new_err)?;
    let obs: Vec<f32> = obs.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    if obs.len() != net.obs_size {
        return Err(PyValueError::new_err("observation size mismatch"));
    }
    let mut logits = Vec::new();
    let v = net.forward(&obs, &mut catanatron::nn::Scratch::default(), &mut logits);
    Ok((f32_bytes(py, &logits), v))
}

/// AlphaZero-style self-play games with network `net` (CTNN bytes). Returns
/// a dict: obs (f32 [n][obs_size]), mask (u8 [n][action_size]), pi (f32
/// [n][action_size], policy targets), z (f32 [n], result for the mover), q (f32 [n], search value for the mover),
/// z_luck (f32 [n], z minus later luck), aux (f32 [n][aux_size], auxiliary targets), aux_size, surprise (f32 [n], KL(target || prior)), endgame (u8 [n]), opening (u8 [n], initial placements), n, results
/// ([first player wins, second player wins, no winner]; endgame branches not
/// counted), decisions, escalated, branches, luck_abs, luck_events. Releases the GIL while playing.
#[pyfunction]
#[pyo3(signature = (net, games, seed, threads=0, sims_full=200, sims_fast=32, p_full=0.25, sims_placement=0,
                    temp_moves=15, root_noise=0.25, dirichlet_alpha=0.3, c_puct=1.5, fpu_reduction=0.1,
                    escalate_gap=0.0, entropy_budget=false, kl_escalate=false, admit_kl=0.0, focus_sims=0, endgame_vp=0, endgame_branches=0, opening_branches=0, placement_forks=0, placement_temp=1.0, placement_uniform=0.0, fork_settlements_only=false, placement_margin=0.0, placement_flatten=0.0, placement_horizon_turns=0, placement_playouts=12, placement_horizon_tau=0.2, placement_horizon_policy=false, midgame_horizon_turns=0, midgame_playouts=4, placement_fork_weights=vec![1.0, 1.0, 1.0, 1.0], move_forks=0, move_fork_temp=1.0, move_fork_uniform=0.0, sibling_labels=false, pool_board_share=1.0, random_share=0.0, random_eps=1.0, scarce_ows_share=0.0, port_start_share=0.0, margin_weight=0.0, speed_weight=0.0, speed_turns=100.0, gumbel_m=0, pool=Vec::new(), pool_share=0.0, opening_net=None, map_type="BASE", vps_to_win=10, discard_limit=7, friendly_robber="off", dice="fair", numbers="spiral", board=None))]
#[allow(clippy::too_many_arguments)]
fn self_play<'py>(
    py: Python<'py>,
    net: &[u8],
    games: u64,
    seed: u64,
    threads: usize,
    sims_full: u32,
    sims_fast: u32,
    p_full: f32,
    sims_placement: u32,
    temp_moves: u32,
    root_noise: f32,
    dirichlet_alpha: f32,
    c_puct: f32,
    fpu_reduction: f32,
    escalate_gap: f32,
    entropy_budget: bool,
    kl_escalate: bool,
    admit_kl: f32,
    focus_sims: u32,
    endgame_vp: u32,
    endgame_branches: u32,
    opening_branches: u32,
    placement_forks: u32,
    placement_temp: f32,
    placement_uniform: f32,
    fork_settlements_only: bool,
    placement_margin: f32,
    placement_flatten: f32,
    placement_horizon_turns: u32,
    placement_playouts: u32,
    placement_horizon_tau: f32,
    placement_horizon_policy: bool,
    midgame_horizon_turns: u32,
    midgame_playouts: u32,
    placement_fork_weights: Vec<f32>,
    move_forks: u32,
    move_fork_temp: f32,
    move_fork_uniform: f32,
    sibling_labels: bool,
    pool_board_share: f32,
    random_share: f32,
    random_eps: f32,
    scarce_ows_share: f32,
    port_start_share: f32,
    margin_weight: f32,
    speed_weight: f32,
    speed_turns: f32,
    gumbel_m: u32,
    pool: Vec<Vec<u8>>,
    pool_share: f32,
    opening_net: Option<&[u8]>,
    map_type: &str,
    vps_to_win: u32,
    discard_limit: u32,
    friendly_robber: &str,
    dice: &str,
    numbers: &str,
    board: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let base = game_rules(map_type, vps_to_win, discard_limit, friendly_robber, dice, numbers)?;
    let game = catanatron::GameConfig {
        board_mix: board_mix(scarce_ows_share, port_start_share)?,
        ..with_board(board, base)?
    };
    // A `seeds:` pool's seeds in order, so pool games report their board.
    let pool_seeds: Vec<u64> = board
        .and_then(|b| b.strip_prefix("seeds:"))
        .map(|l| l.split(',').filter_map(|n| n.trim().parse().ok()).collect())
        .unwrap_or_default();
    use catanatron::selfplay::SelfPlayConfig;
    if sims_full < 1 || sims_fast < 1 || !(0.0..=1.0).contains(&p_full) || placement_temp <= 0.0 || !(0.0..=1.0).contains(&placement_uniform)
        || !(0.0..=1.0).contains(&random_share) || !(0.0..=1.0).contains(&random_eps) {
        return Err(PyValueError::new_err("need sims >= 1, 0 <= p_full <= 1, placement_temp > 0 and placement_uniform, random_share, random_eps in [0, 1]"));
    }
    if margin_weight < 0.0 || speed_weight < 0.0 || speed_turns <= 0.0 {
        return Err(PyValueError::new_err("need margin_weight, speed_weight >= 0 and speed_turns > 0"));
    }
    let utility = catanatron::mcts::Utility { margin: margin_weight, speed: speed_weight, speed_turns };
    let mut main = catanatron::nn::PolicyValueNet::from_bytes(net).map_err(PyValueError::new_err)?;
    if let Some(o) = opening_net {
        // A separate network for the initial placements (see PolicyValueNet::opening).
        main = main.with_opening(std::sync::Arc::new(catanatron::nn::PolicyValueNet::from_bytes(o).map_err(PyValueError::new_err)?));
    }
    let net = std::sync::Arc::new(main);
    if !(0.0..=1.0).contains(&pool_share) {
        return Err(PyValueError::new_err("pool_share must be in [0, 1]"));
    }
    let pool_nets = pool
        .iter()
        .map(|b| catanatron::nn::PolicyValueNet::from_bytes(b).map(std::sync::Arc::new))
        .collect::<Result<Vec<_>, _>>()
        .map_err(PyValueError::new_err)?;
    let threads = if threads == 0 { catanatron::parallel::default_threads() } else { threads };
    if placement_fork_weights.len() != 4 {
        return Err(PyValueError::new_err("placement_fork_weights needs 4 entries (one per settlement decision)"));
    }
    let placement_fork_weights = [placement_fork_weights[0], placement_fork_weights[1], placement_fork_weights[2], placement_fork_weights[3]];
    let cfg = SelfPlayConfig { sims_full, sims_fast, sims_placement, p_full, temp_moves, root_noise, dirichlet_alpha, c_puct, fpu_reduction, escalate_gap, entropy_budget, kl_escalate, admit_kl, focus_sims, endgame_vp, endgame_branches, opening_branches, placement_forks, placement_temp, placement_uniform, fork_settlements_only, placement_margin, placement_flatten, placement_horizon_turns, placement_playouts, placement_horizon_tau, placement_horizon_policy, midgame_horizon_turns, midgame_playouts, placement_fork_weights, move_forks, move_fork_temp, move_fork_uniform, sibling_labels, pool_board_share, pool_seeds, random_share, random_eps, utility, gumbel_m, game };
    let data = py
        .detach(|| catanatron::selfplay::self_play_with_pool(net, &pool_nets, pool_share, games, seed, threads, &cfg))
        .map_err(PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("obs", raw_bytes(py, &data.obs))?;
    d.set_item("mask", raw_bytes(py, &data.mask))?;
    d.set_item("pi", raw_bytes(py, &data.pi))?;
    d.set_item("z", raw_bytes(py, &data.z))?;
    d.set_item("q", raw_bytes(py, &data.q))?;
    d.set_item("z_luck", raw_bytes(py, &data.z_luck))?;
    d.set_item("aux", raw_bytes(py, &data.aux))?;
    d.set_item("aux_size", catanatron::selfplay::AUX_TARGETS)?;
    d.set_item("luck_abs", data.luck_abs)?;
    d.set_item("luck_events", data.luck_events)?;
    d.set_item("surprise", raw_bytes(py, &data.surprise))?;
    d.set_item("endgame", raw_bytes(py, &data.endgame))?;
    d.set_item("opening", raw_bytes(py, &data.opening))?;
    d.set_item("focus", raw_bytes(py, &data.focus))?;
    d.set_item("kind", raw_bytes(py, &data.kind))?;
    d.set_item("horizon", f32_bytes(py, &data.horizon))?;
    d.set_item("kind_names", catanatron::selfplay::KIND_NAMES.to_vec())?;
    d.set_item("game", raw_bytes(py, &data.game))?;
    d.set_item("focused", data.focused)?;
    d.set_item("random_games", data.random_games)?;
    d.set_item("escalated", data.escalated)?;
    d.set_item("admitted", data.admitted)?;
    d.set_item("branches", data.branches)?;
    d.set_item("forks", data.forks)?;
    d.set_item("boards", data.boards.clone())?;
    d.set_item("fork_pairs", data.fork_pairs.clone())?;
    d.set_item("pool_games", data.pool_games)?;
    d.set_item("n", data.len())?;
    d.set_item("results", data.results.to_vec())?;
    d.set_item("decisions", data.decisions)?;
    Ok(d)
}

/// Board topology in observation order, for structured networks: per land
/// node, its land tiles (positions in the observation's tile order); per
/// land edge (observation and action-space order), its two nodes.
#[pyfunction]
#[pyo3(signature = (map_type="BASE"))]
fn board_topology(map_type: &str) -> PyResult<(Vec<Vec<usize>>, Vec<(u32, u32)>)> {
    use catanatron::map::{build_map, NumberPlacement};
    let map = build_map(self::map_type(map_type)?, NumberPlacement::OfficialSpiral, &mut catanatron::rng::Rng::new(0));
    let mut land_pos = vec![usize::MAX; map.tiles.len()];
    for (t, &ti) in map.land_tiles.iter().enumerate() {
        land_pos[ti] = t;
    }
    let num_land_nodes = map.land_nodes.iter().filter(|&&x| x).count();
    let node_tiles = (0..num_land_nodes)
        .map(|n| map.adjacent_tiles[n].iter().map(|&ti| land_pos[ti]).filter(|&t| t != usize::MAX).collect())
        .collect();
    let edges = (0..map.edges.len())
        .filter(|&i| map.land_edge[i])
        .map(|i| (map.edges[i].0 as u32, map.edges[i].1 as u32))
        .collect();
    Ok((node_tiles, edges))
}

/// The settlement action slot of every land node (node order), for the spot head's tables.
#[pyfunction]
#[pyo3(signature = (map_type="BASE"))]
fn settlement_slots(map_type: &str) -> PyResult<Vec<u32>> {
    use catanatron::map::{build_map, NumberPlacement};
    let map = build_map(self::map_type(map_type)?, NumberPlacement::OfficialSpiral, &mut catanatron::rng::Rng::new(0));
    Ok(catanatron::encoding::ActionSpace::new(&map).settlement_slots())
}

/// Board-structured action slots for the board heads' tables: (settlement [node], city [node],
/// road [land edge in `board_topology` order], robber [land tile][victim 0..4]).
#[pyfunction]
#[pyo3(signature = (map_type="BASE"))]
#[allow(clippy::type_complexity)]
fn board_slots(map_type: &str) -> PyResult<(Vec<u32>, Vec<u32>, Vec<u32>, Vec<Vec<u32>>)> {
    use catanatron::map::{build_map, NumberPlacement};
    let map = build_map(self::map_type(map_type)?, NumberPlacement::OfficialSpiral, &mut catanatron::rng::Rng::new(0));
    Ok(catanatron::encoding::ActionSpace::new(&map).board_slots())
}

/// One of our boards (`board` as `with_board` takes it, e.g. "seed:N") in the
/// OpenHex frame `Mirror` takes: land tiles ((x, y, z), terrain, number) and
/// ports (vertex a, vertex b, resource or None for 3:1).
#[pyfunction]
#[pyo3(signature = (board, map_type="BASE", numbers="spiral-random-corner"))]
#[allow(clippy::type_complexity)]
fn board_layout(
    board: &str,
    map_type: &str,
    numbers: &str,
) -> PyResult<(Vec<((i8, i8, i8), String, Option<u8>)>, Vec<((i8, i8, i8), (i8, i8, i8), Option<String>)>)> {
    use catanatron::map::{port_noderefs, TileKind};
    use catanatron::openhex::{turn_hex, turn_vertex, Geometry};
    use catanatron::types::Resource;
    // `numbers` must match the rules the board's networks trained under
    // (OpenHex 1v1: spiral from a random corner), or the seed gives another board.
    let rules = game_rules(map_type, 15, 9, "public", "openhex", numbers)?;
    let map = with_board(Some(board), rules)?
        .catan_map
        .ok_or_else(|| PyValueError::new_err("board_layout needs a single board (seed:N or a board spec)"))?;
    let geo = Geometry::new(&map).map_err(PyValueError::new_err)?;
    let terrain = |r: Option<Resource>| {
        match r {
            None => "desert",
            Some(Resource::Wood) => "forest",
            Some(Resource::Brick) => "hills",
            Some(Resource::Sheep) => "pasture",
            Some(Resource::Wheat) => "fields",
            Some(Resource::Ore) => "mountains",
        }
        .to_string()
    };
    let (mut tiles, mut ports) = (Vec::new(), Vec::new());
    for t in &map.tiles {
        match t.kind {
            TileKind::Land { resource, number, .. } => tiles.push((turn_hex(t.coordinate), terrain(resource), number)),
            TileKind::Port { resource, direction, .. } => {
                let (a, b) = port_noderefs(direction);
                let v = |n: catanatron::types::NodeId| turn_vertex(geo.node_vertex[&n]);
                ports.push((v(t.nodes[a]), v(t.nodes[b]), resource.map(|r| mirror::resource_name(r).to_string())));
            }
            TileKind::Water => {}
        }
    }
    Ok((tiles, ports))
}

/// The board's symmetries for a network with `obs_size` inputs, as
/// (observation permutation, action permutation) pairs: entry i of a
/// transformed vector is entry perm[i] of the original. Identity first.
#[pyfunction]
#[pyo3(signature = (obs_size, map_type="BASE"))]
fn board_symmetries(obs_size: usize, map_type: &str) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
    use catanatron::encoding::{symmetries, ActionSpace, ObservationSpace};
    use catanatron::map::{build_map, NumberPlacement};
    let map = build_map(self::map_type(map_type)?, NumberPlacement::OfficialSpiral, &mut catanatron::rng::Rng::new(0));
    let obs = ObservationSpace::for_network(&map, obs_size);
    if obs.len() != obs_size {
        return Err(PyValueError::new_err(format!("no observation layout has {obs_size} inputs")));
    }
    Ok(symmetries(&map, &obs, &ActionSpace::new(&map)).into_iter().map(|s| (s.obs, s.actions)).collect())
}

#[pymodule]
fn catanatron_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVecEnv>()?;
    m.add_class::<mirror::PyMirror>()?;
    m.add_function(wrap_pyfunction!(net_forward, m)?)?;
    m.add_function(wrap_pyfunction!(play_bots, m)?)?;
    m.add_function(wrap_pyfunction!(board_layout, m)?)?;
    m.add_function(wrap_pyfunction!(play_bots_by_seat, m)?)?;
    m.add_function(wrap_pyfunction!(self_play, m)?)?;
    m.add_function(wrap_pyfunction!(board_topology, m)?)?;
    m.add_function(wrap_pyfunction!(settlement_slots, m)?)?;
    m.add_function(wrap_pyfunction!(board_slots, m)?)?;
    m.add_function(wrap_pyfunction!(board_symmetries, m)?)?;
    Ok(())
}
