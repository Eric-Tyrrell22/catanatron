use catanatron::mcts::load_net;
use catanatron::selfplay::{self_play, SelfPlayConfig};

const MODEL: &str = "python/runs/selfplay-smoke/model.ctnn";

#[test]
fn self_play_examples_are_consistent_and_thread_independent() {
    if !std::path::Path::new(MODEL).exists() {
        eprintln!("skipping: no exported model at {MODEL}");
        return;
    }
    let net = load_net(MODEL).unwrap();
    let cfg = SelfPlayConfig {
        sims_full: 16,
        sims_fast: 4,
        sims_placement: 24,
        p_full: 0.5,
        entropy_budget: false, kl_escalate: false, admit_kl: 0.0, focus_sims: 0,
        temp_moves: 10,
        root_noise: 0.25,
        dirichlet_alpha: 0.3,
        c_puct: 1.5,
        fpu_reduction: 0.1,
        escalate_gap: 0.1,
        endgame_vp: 7,
        endgame_branches: 2,
        opening_branches: 1,
        placement_forks: 1,
        placement_temp: 2.0,
        placement_uniform: 0.0,
        move_forks: 0,
        move_fork_temp: 1.0,
        move_fork_uniform: 0.0,
        sibling_labels: false,
        pool_board_share: 1.0,
        pool_seeds: Vec::new(),
        random_share: 0.0,
        random_eps: 1.0,
        utility: catanatron::mcts::Utility { margin: 0.1, speed: 0.1, speed_turns: 200.0 },
        gumbel_m: 0,
        game: Default::default(),
    };
    let a = self_play(net.clone(), 4, 7, 1, &cfg).unwrap();
    let b = self_play(net.clone(), 4, 7, 3, &cfg).unwrap();
    assert_eq!((&a.obs, &a.mask, &a.pi, &a.z, a.results), (&b.obs, &b.mask, &b.pi, &b.z, b.results));
    assert_eq!((&a.surprise, &a.endgame, a.escalated, a.branches), (&b.surprise, &b.endgame, b.escalated, b.branches));
    assert_eq!((&a.z_luck, a.forks), (&b.z_luck, b.forks));
    assert_eq!(a.forks, 4, "one placement fork per game");
    assert!(a.escalated > 0, "some close calls are re-searched");
    // Two endgame branches per game reaching 7 points, one opening branch per game.
    assert!(a.branches >= 4 && (a.branches - 4) % 2 == 0 && a.branches <= 12, "branches: {}", a.branches);
    assert!(a.luck_events > 0 && a.luck_abs > 0.0);

    // Uniform forks: still one per game and thread-independent, but they
    // open differently than forks drawn from the search target.
    let uniform_cfg = SelfPlayConfig { placement_uniform: 1.0, ..cfg.clone() };
    let u = self_play(net.clone(), 4, 7, 1, &uniform_cfg).unwrap();
    let v = self_play(net.clone(), 4, 7, 3, &uniform_cfg).unwrap();
    assert_eq!((&u.obs, &u.pi, &u.z, u.forks), (&v.obs, &v.pi, &v.z, v.forks));
    assert_eq!(u.forks, 4, "one placement fork per game");
    assert_ne!(u.obs, a.obs, "uniform forks change the forked games");

    // Games against a random opponent: counted, thread independent, and
    // with fewer recorded decisions per game (the opponent's aren't).
    let random_cfg = SelfPlayConfig { random_share: 1.0, random_eps: 1.0, placement_forks: 0, endgame_branches: 0, opening_branches: 0, ..cfg.clone() };
    let r1 = self_play(net.clone(), 4, 7, 1, &random_cfg).unwrap();
    let r2 = self_play(net.clone(), 4, 7, 3, &random_cfg).unwrap();
    assert_eq!((&r1.obs, &r1.z, r1.random_games), (&r2.obs, &r2.z, r2.random_games));
    assert_eq!(r1.random_games, 8, "every slot is a pair against the random opponent");

    let n = a.len();
    assert!(n > 0);
    assert_eq!(a.q.len(), n);
    assert_eq!(a.surprise.len(), n);
    assert_eq!(a.endgame.len(), n);
    assert_eq!(a.opening.len(), n);
    // Each game records its 4 initial settlements and 4 roads (forks and
    // opening branches add more); the rest of the game is not opening.
    let openings = a.opening.iter().filter(|&&o| o == 1).count();
    assert!(openings >= 4 * 8 && openings < n / 4, "opening examples: {openings} of {n}");
    assert_eq!(a.z_luck.len(), n);
    assert!(a.z_luck.iter().all(|x| x.is_finite() && x.abs() < 10.0));
    // Results plus at most 0.1 each for margin and speed (averaged for openings).
    assert!(a.z.iter().all(|&z| z.abs() <= 1.2 + 1e-6));
    assert!(a.z.iter().any(|&z| z.abs() > 1.0 + 1e-6), "margin and speed count");
    assert_eq!(a.aux.len(), n * catanatron::selfplay::AUX_TARGETS);
    assert!(a.aux.iter().all(|x| x.is_finite() && x.abs() <= 2.0));
    assert_eq!(&a.aux, &b.aux);
    assert!(a.z_luck.iter().zip(&a.z).any(|(l, z)| l != z), "luck adjusts some targets");
    assert!(a.surprise.iter().all(|&x| x.is_finite() && x >= 0.0));
    assert!(a.endgame.iter().any(|&e| e == 1));
    assert_eq!(a.results.iter().sum::<u64>(), 4);
    assert_eq!(a.obs.len() % n, 0);
    let actions = a.mask.len() / n;
    assert_eq!(a.pi.len(), n * actions);
    for k in 0..n {
        let pi = &a.pi[k * actions..(k + 1) * actions];
        let mask = &a.mask[k * actions..(k + 1) * actions];
        assert!((pi.iter().sum::<f32>() - 1.0).abs() < 1e-4, "pi sums to 1");
        assert!(pi.iter().zip(mask).all(|(&p, &m)| p == 0.0 || m == 1), "pi only on legal moves");
        assert!(a.z[k].abs() <= 1.2 + 1e-6);
        assert!(a.q[k].is_finite());
    }
}

/// Gumbel self-play: same invariants, thread-independent, and every recorded
/// target is a distribution over legal moves.
#[test]
fn gumbel_self_play_is_consistent() {
    if !std::path::Path::new(MODEL).exists() {
        eprintln!("skipping: no exported model at {MODEL}");
        return;
    }
    let net = load_net(MODEL).unwrap();
    let cfg = SelfPlayConfig {
        sims_full: 32,
        sims_fast: 8,
        sims_placement: 48,
        p_full: 0.5,
        entropy_budget: false, kl_escalate: false, admit_kl: 0.0, focus_sims: 0,
        temp_moves: 10,
        root_noise: 0.25,
        dirichlet_alpha: 0.3,
        c_puct: 1.5,
        fpu_reduction: 0.1,
        escalate_gap: 0.0,
        endgame_vp: 0,
        endgame_branches: 0,
        opening_branches: 0,
        placement_forks: 0,
        placement_temp: 1.0,
        placement_uniform: 0.0,
        move_forks: 0,
        move_fork_temp: 1.0,
        move_fork_uniform: 0.0,
        sibling_labels: false,
        pool_board_share: 1.0,
        pool_seeds: Vec::new(),
        random_share: 0.0,
        random_eps: 1.0,
        utility: Default::default(),
        gumbel_m: 8,
        game: Default::default(),
    };
    let a = self_play(net.clone(), 4, 9, 1, &cfg).unwrap();
    let b = self_play(net.clone(), 4, 9, 3, &cfg).unwrap();
    assert_eq!((&a.obs, &a.pi, &a.z, a.results), (&b.obs, &b.pi, &b.z, b.results));
    let n = a.len();
    assert!(n > 0);
    assert_eq!(a.results.iter().sum::<u64>(), 4);
    let actions = a.mask.len() / n;
    for k in 0..n {
        let pi = &a.pi[k * actions..(k + 1) * actions];
        let mask = &a.mask[k * actions..(k + 1) * actions];
        assert!((pi.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!(pi.iter().zip(mask).all(|(&p, &m)| p == 0.0 || m == 1));
    }
}

/// Past-network games: at share 1 every slot is a seat-swapped pair on one
/// board, and only the current network's seat is recorded.
#[test]
fn pool_games_record_only_the_current_network() {
    if !std::path::Path::new(MODEL).exists() {
        eprintln!("skipping: no exported model at {MODEL}");
        return;
    }
    let net = load_net(MODEL).unwrap();
    let cfg = SelfPlayConfig {
        sims_full: 8,
        sims_fast: 4,
        sims_placement: 8,
        p_full: 1.0,
        entropy_budget: false, kl_escalate: false, admit_kl: 0.0, focus_sims: 0,
        temp_moves: 10,
        root_noise: 0.25,
        dirichlet_alpha: 0.3,
        c_puct: 1.5,
        fpu_reduction: 0.1,
        escalate_gap: 0.0,
        endgame_vp: 0,
        endgame_branches: 0,
        opening_branches: 0,
        placement_forks: 0,
        placement_temp: 1.0,
        placement_uniform: 0.0,
        move_forks: 0,
        move_fork_temp: 1.0,
        move_fork_uniform: 0.0,
        sibling_labels: false,
        pool_board_share: 1.0,
        pool_seeds: Vec::new(),
        random_share: 0.0,
        random_eps: 1.0,
        utility: Default::default(),
        gumbel_m: 0,
        game: Default::default(),
    };
    let plain = self_play(net.clone(), 6, 11, 2, &cfg).unwrap();
    let pooled = catanatron::selfplay::self_play_with_pool(net.clone(), &[net.clone()], 1.0, 6, 11, 2, &cfg).unwrap();
    assert_eq!(plain.pool_games, 0);
    // Each slot becomes a pair (current network first, then second) with
    // one seat recorded per game: about as many examples as plain play.
    assert_eq!(pooled.pool_games, 12);
    assert_eq!(pooled.results.iter().sum::<u64>(), 12);
    let ratio = pooled.len() as f64 / plain.len() as f64;
    assert!((0.6..1.4).contains(&ratio), "one seat recorded: {} vs {}", pooled.len(), plain.len());
    let again = catanatron::selfplay::self_play_with_pool(net.clone(), &[net.clone()], 1.0, 6, 11, 3, &cfg).unwrap();
    assert_eq!((&again.obs, &again.z), (&pooled.obs, &pooled.z), "thread-independent");
}
