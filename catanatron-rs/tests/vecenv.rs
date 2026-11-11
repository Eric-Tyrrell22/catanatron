use catanatron::rng::Rng;
use catanatron::vecenv::{VecEnv, VecEnvConfig};

/// Picks a random legal action per env from the masks.
fn random_actions(out_mask: &[u8], n: usize, a: usize, rng: &mut Rng) -> Vec<usize> {
    (0..n)
        .map(|i| {
            let legal: Vec<usize> = (0..a).filter(|&k| out_mask[i * a + k] == 1).collect();
            *rng.choose(&legal)
        })
        .collect()
}

fn run(config: VecEnvConfig, steps: usize) -> (u32, u32, u32, f64) {
    let mut env = VecEnv::new(config).unwrap();
    let (n, a) = (env.num_envs(), env.action_size());
    let o = env.observe();
    let mut out = catanatron::vecenv::StepOutput { obs: o.obs, mask: o.mask, ..Default::default() };
    let mut rng = Rng::new(5);
    let (mut episodes, mut wins, mut truncated, mut checksum) = (0, 0, 0, 0.0);
    for _ in 0..steps {
        assert!(out.mask.chunks(a).all(|m| m.contains(&1)), "an env has no legal action");
        let acts = random_actions(&out.mask, n, a, &mut rng);
        out = env.step(&acts).unwrap();
        for i in 0..n {
            if out.done[i] == 1 {
                episodes += 1;
                wins += out.won[i] as u32;
                truncated += out.truncated[i] as u32;
                let r = out.reward[i];
                assert!(if out.won[i] == 1 { r > 0.0 } else { r <= 0.0 }, "reward {r}");
            } else {
                assert_eq!(out.reward[i], 0.0);
            }
        }
        checksum += out.obs.iter().map(|&x| x as f64).sum::<f64>();
    }
    (episodes, wins, truncated, checksum)
}

#[test]
fn random_learner_loses_to_value_bot() {
    let (episodes, wins, _, _) = run(VecEnvConfig { num_envs: 16, opponents: vec!["F".into()], threads: 4, ..Default::default() }, 3000);
    assert!(episodes > 20, "only {episodes} episodes");
    assert!(wins * 10 < episodes, "random learner won {wins}/{episodes}");
}

#[test]
fn results_do_not_depend_on_threads() {
    let cfg = |threads| VecEnvConfig { num_envs: 12, opponents: vec!["W".into(), "R".into()], threads, seed: 3, ..Default::default() };
    assert_eq!(run(cfg(1), 400), run(cfg(5), 400));
}

#[test]
fn rejects_bad_input() {
    assert!(VecEnv::new(VecEnvConfig { opponents: vec![], ..Default::default() }).is_err());
    assert!(VecEnv::new(VecEnvConfig { opponents: vec!["NOPE".into()], ..Default::default() }).is_err());
    let mut env = VecEnv::new(VecEnvConfig { num_envs: 2, ..Default::default() }).unwrap();
    let out = env.observe();
    let a = env.action_size();
    let illegal = (0..a).find(|&k| out.mask[k] == 0).unwrap();
    let legal = (0..a).find(|&k| out.mask[a + k] == 1).unwrap();
    assert!(env.step(&[illegal, legal]).is_err());
    assert!(env.step(&[0]).is_err());
}

/// Self-play driver: learner and external seats both pick random legal moves.
fn run_external(seed: u64, threads: usize, steps: usize) -> (u32, u32, f64) {
    let cfg = VecEnvConfig { num_envs: 10, opponents: vec!["SELF".into(), "W".into()], seed, threads, ..Default::default() };
    let mut env = VecEnv::new(cfg).unwrap();
    let (n, a) = (env.num_envs(), env.action_size());
    let mut rng = Rng::new(9);
    let resolve = |env: &mut VecEnv, rng: &mut Rng| loop {
        let idxs = env.pending_opponents();
        if idxs.is_empty() {
            break;
        }
        let o = env.observe_envs(&idxs);
        let acts = random_actions(&o.mask, idxs.len(), a, rng);
        env.act_opponents(&idxs, &acts).unwrap();
    };
    resolve(&mut env, &mut rng);
    let (mut episodes, mut wins, mut checksum) = (0, 0, 0.0);
    for _ in 0..steps {
        let o = env.observe();
        assert!(o.mask.chunks(a).all(|m| m.contains(&1)));
        let acts = random_actions(&o.mask, n, a, &mut rng);
        env.act(&acts).unwrap();
        resolve(&mut env, &mut rng);
        let r = env.finish().unwrap();
        resolve(&mut env, &mut rng);
        for i in 0..n {
            if r.done[i] == 1 {
                episodes += 1;
                wins += r.won[i] as u32;
            }
        }
        checksum += o.obs.iter().map(|&x| x as f64).sum::<f64>();
    }
    (episodes, wins, checksum)
}

#[test]
fn external_seats_play_full_games() {
    let (episodes, wins, _) = run_external(1, 4, 2500);
    assert!(episodes >= 10, "only {episodes} episodes");
    // Random learner vs random external seat + weighted bot: should win sometimes.
    assert!(wins > 0 && wins < episodes);
}

#[test]
fn external_seats_are_thread_independent() {
    assert_eq!(run_external(2, 1, 300), run_external(2, 6, 300));
}

#[test]
fn step_rejects_external_seats() {
    let mut env = VecEnv::new(VecEnvConfig { num_envs: 2, opponents: vec!["SELF".into()], ..Default::default() }).unwrap();
    assert!(env.step(&[0, 0]).is_err());
}

const MODEL: &str = "python/runs/selfplay-smoke/model.ctnn";

fn rollout_env(threads: usize, opponents: &[&str]) -> Option<VecEnv> {
    if !std::path::Path::new(MODEL).exists() {
        eprintln!("skipping: no exported model at {MODEL}");
        return None;
    }
    let cfg = VecEnvConfig {
        num_envs: 12,
        opponents: opponents.iter().map(|s| s.to_string()).collect(),
        threads,
        seed: 11,
        vp_reward: 0.05,
        ..Default::default()
    };
    let mut env = VecEnv::new(cfg).unwrap();
    env.set_policy(catanatron::mcts::load_net(MODEL).unwrap()).unwrap();
    Some(env)
}

#[test]
fn rollout_is_consistent_and_thread_independent() {
    let Some(mut a) = rollout_env(1, &["F"]) else { return };
    let mut b = rollout_env(5, &["F"]).unwrap();
    let (ra, rb) = (a.rollout(300).unwrap(), b.rollout(300).unwrap());
    assert_eq!(ra.action, rb.action);
    assert_eq!(ra.reward, rb.reward);
    assert_eq!(ra.obs, rb.obs);

    let (n, t, act) = (ra.num_envs, ra.steps, a.action_size());
    assert_eq!(ra.obs.len(), n * t * a.obs_size());
    let mut episodes = 0;
    for k in 0..n * t {
        // The chosen action is legal and its probability is valid.
        assert_eq!(ra.mask[k * act + ra.action[k] as usize], 1);
        assert!(ra.logp[k] <= 1e-6 && ra.logp[k].is_finite());
        if ra.done[k] == 1 {
            episodes += 1;
            assert!(ra.reward[k].abs() >= 0.9, "terminal reward {}", ra.reward[k]);
        }
    }
    assert!(episodes > 0);
    // Continuing a rollout picks up where the last one stopped.
    let rc = a.rollout(10).unwrap();
    assert_eq!(rc.num_envs, n);
}

#[test]
fn rollout_against_network_opponents() {
    let key = format!("NNS:{MODEL}");
    let Some(mut env) = rollout_env(4, &[&key]) else { return };
    let r = env.rollout(400).unwrap();
    let games: u32 = r.done.iter().map(|&d| d as u32).sum();
    let wins: u32 = r.won.iter().map(|&d| d as u32).sum();
    assert!(games > 5);
    // Same network on both sides: neither should dominate completely.
    assert!(wins > 0 && wins < games, "{wins}/{games}");
}

#[test]
fn rollout_needs_policy() {
    let mut env = VecEnv::new(VecEnvConfig { num_envs: 2, ..Default::default() }).unwrap();
    assert!(env.rollout(5).is_err());
}
