use catanatron::mcts::{load_net, MctsConfig, MctsPlayer};
use catanatron::players::{from_key, Player};
use catanatron::*;

const MODEL: &str = "python/runs/selfplay-smoke/model.ctnn";

#[test]
fn search_returns_visits_for_every_root_move() {
    if !std::path::Path::new(MODEL).exists() {
        eprintln!("skipping: no exported model at {MODEL}");
        return;
    }
    let net = load_net(MODEL).unwrap();
    for seed in 0..20 {
        let mut g = Game::new(&COLORS[..2], &GameConfig { seed: Some(seed), ..Default::default() });
        let mut f = from_key("F", Color::Blue, seed).unwrap();
        let mut m = MctsPlayer::new(Color::Red, net.clone(), MctsConfig { simulations: 30, ..Default::default() }, seed);
        while !g.is_over() {
            let c = g.state.current_color();
            let a = if c == Color::Red {
                if g.playable_actions.len() > 1 {
                    let v = m.search(&g).unwrap();
                    assert_eq!(v.len(), g.playable_actions.len(), "seed {seed} prompt {:?}", g.state.current_prompt);
                }
                m.decide(&g, &g.playable_actions.clone())
            } else {
                f.decide(&g, &g.playable_actions.clone())
            };
            g.execute(a).unwrap();
        }
    }
}

#[test]
fn determinized_root_is_never_already_won() {
    // Late-game positions where opponents hold hidden VP cards.
    if !std::path::Path::new(MODEL).exists() {
        return;
    }
    let net = load_net(MODEL).unwrap();
    for seed in 0..40 {
        let mut g = Game::new(&COLORS[..2], &GameConfig { seed: Some(seed), ..Default::default() });
        let mut bots: Vec<Box<dyn Player>> = vec![from_key("F", Color::Red, seed).unwrap(), from_key("F", Color::Blue, seed + 1).unwrap()];
        let mut m = MctsPlayer::new(Color::Red, net.clone(), MctsConfig { simulations: 8, ..Default::default() }, seed);
        while !g.is_over() {
            if g.state.current_color() == Color::Red && g.playable_actions.len() > 1 {
                let v = m.search(&g).unwrap();
                assert_eq!(v.len(), g.playable_actions.len());
            }
            g.play_tick(&mut bots).unwrap();
        }
    }
}

#[test]
fn batched_forward_matches_single() {
    use catanatron::encoding::ObservationSpace;
    use catanatron::nn::{BatchScratch, Scratch, MAX_BATCH};
    if !std::path::Path::new(MODEL).exists() {
        eprintln!("skipping: no exported model at {MODEL}");
        return;
    }
    let net = load_net(MODEL).unwrap();
    // Positions from a few games between F bots.
    let mut positions = Vec::new();
    for seed in 0..3 {
        let mut g = Game::new(&COLORS[..2], &GameConfig { seed: Some(seed), ..Default::default() });
        let mut bots: Vec<_> = (0..2).map(|i| from_key("F", COLORS[i], seed + i as u64).unwrap()).collect();
        let space = ObservationSpace::new(&g.state.board.map);
        while !g.is_over() {
            let mut nz = Vec::new();
            let n = space.encode_sparse_into(&g.state, g.state.current_color(), &mut nz);
            nz.truncate(n);
            positions.push(nz);
            g.play_tick(&mut bots).unwrap();
        }
    }
    let (mut scratch, mut batch) = (Scratch::default(), BatchScratch::default());
    let mut single_logits = Vec::new();
    let mut logits = vec![Vec::new(); MAX_BATCH];
    let mut values = [0.0; MAX_BATCH];
    for size in 1..=MAX_BATCH {
        for chunk in positions.chunks(size) {
            let xs: Vec<&[(u32, f32)]> = chunk.iter().map(|p| &p[..]).collect();
            net.forward_batch(&xs, &mut batch, &mut logits, &mut values);
            for (b, x) in xs.iter().enumerate() {
                let v = net.forward_sparse(x, &mut scratch, &mut single_logits);
                assert_eq!(values[b], v, "value, batch size {size}");
                assert_eq!(logits[b], single_logits, "logits, batch size {size}");
            }
        }
    }
}

/// Gumbel root search: stays within the budget, plays a legal move that was
/// searched, and is deterministic for a seed.
#[test]
fn gumbel_search_is_bounded_and_deterministic() {
    use catanatron::mcts::{MctsConfig, MctsPlayer};
    let path = "python/runs/selfplay-smoke/model.ctnn";
    if !std::path::Path::new(path).exists() {
        eprintln!("skipping: no exported model at {path}");
        return;
    }
    let net = catanatron::mcts::load_net(path).unwrap();
    let config = MctsConfig { simulations: 64, gumbel_m: 8, gumbel_noise: true, ..Default::default() };
    let mut g = catanatron::Game::new(&catanatron::COLORS[..2], &catanatron::GameConfig { seed: Some(2), ..Default::default() });
    let mut bots: Vec<Box<dyn catanatron::Player>> = catanatron::COLORS[..2].iter().map(|&c| catanatron::players::from_key("R", c, 2).unwrap()).collect();
    let mut checked = 0;
    while !g.is_over() && checked < 30 {
        let actions = g.playable_actions.clone();
        if actions.len() > 1 {
            let color = g.state.current_color();
            let run = |seed| {
                let mut p = MctsPlayer::new(color, net.clone(), config, seed);
                let visits = p.search(&g).unwrap();
                (visits, p.gumbel_choice().unwrap())
            };
            let (visits, choice) = run(5);
            assert_eq!(run(5), (visits.clone(), choice), "deterministic for a seed");
            assert!(visits.iter().sum::<u32>() <= 64);
            assert!(choice < actions.len());
            assert!(visits[choice] > 0 || actions.len() == 1, "the chosen move was searched");
            checked += 1;
        }
        g.play_tick(&mut bots).unwrap();
    }
    assert!(checked >= 20);
}

/// A time-budgeted search runs about as long as asked, and more simulations
/// with more time.
#[test]
fn time_budget_controls_search_length() {
    use catanatron::mcts::{MctsConfig, MctsPlayer};
    let path = "python/runs/selfplay-smoke/model.ctnn";
    if !std::path::Path::new(path).exists() {
        eprintln!("skipping: no exported model at {path}");
        return;
    }
    let net = catanatron::mcts::load_net(path).unwrap();
    let g = catanatron::Game::new(&catanatron::COLORS[..2], &catanatron::GameConfig { seed: Some(3), ..Default::default() });
    let mut sims = Vec::new();
    for ms in [100u64, 400] {
        let mut p = MctsPlayer::new(g.state.current_color(), net.clone(), MctsConfig { time_limit_ms: ms, ..Default::default() }, 1);
        let t = std::time::Instant::now();
        let visits = p.search(&g).unwrap();
        let took = t.elapsed().as_millis() as u64;
        assert!(took >= ms && took < ms + 150, "{ms} ms budget took {took} ms");
        sims.push(visits.iter().sum::<u32>());
    }
    assert!(sims[1] > 2 * sims[0], "more time, more simulations: {sims:?}");
}

/// Root-parallel play: a legal move, and with a time budget every search
/// runs at once (one budget, not one per thread).
#[test]
fn root_parallel_search_plays_legal_moves_within_the_budget() {
    use catanatron::players::from_key;
    let path = "python/runs/selfplay-smoke/model.ctnn";
    if !std::path::Path::new(path).exists() {
        eprintln!("skipping: no exported model at {path}");
        return;
    }
    let g = catanatron::Game::new(&catanatron::COLORS[..2], &catanatron::GameConfig { seed: Some(4), ..Default::default() });
    let c = g.state.current_color();
    let mut p = from_key(&format!("MCTS:100,t200,j4:{path}"), c, 1).unwrap();
    let actions = g.playable_actions.clone();
    let t = std::time::Instant::now();
    let a = p.decide(&g, &actions);
    let took = t.elapsed().as_millis();
    assert!(actions.contains(&a));
    assert!((200..450).contains(&took), "4 parallel 200 ms searches took {took} ms");
}

/// Lock-free shared-tree search: total root visits match the simulations
/// run (every visit backed up, no virtual loss left behind), the chosen
/// move is legal, and it runs through whole games without errors.
#[test]
fn lockfree_search_is_consistent() {
    use catanatron::mcts::{MctsConfig, MctsPlayer};
    use catanatron::players::from_key;
    let path = "python/runs/selfplay-smoke/model.ctnn";
    if !std::path::Path::new(path).exists() {
        eprintln!("skipping: no exported model at {path}");
        return;
    }
    let net = catanatron::mcts::load_net(path).unwrap();
    let mut g = catanatron::Game::new(&catanatron::COLORS[..2], &catanatron::GameConfig { seed: Some(8), ..Default::default() });
    let mut bots: Vec<Box<dyn catanatron::Player>> =
        catanatron::COLORS[..2].iter().map(|&c| from_key(&format!("MCTS:300,x6:{path}"), c, 5).unwrap()).collect();
    let mut checked = 0;
    while !g.is_over() && checked < 25 {
        if g.playable_actions.len() > 1 {
            let config = MctsConfig { simulations: 2000, lockfree_threads: 6, ..Default::default() };
            let mut p = MctsPlayer::new(g.state.current_color(), net.clone(), config, 3);
            let visits = p.search_lockfree(&g).unwrap();
            let total: u32 = visits.iter().sum();
            // Simulations stop at 2000 overall; threads already running may finish a few more.
            assert!((1999..=2000 + 6).contains(&total), "root visits {total}");
            checked += 1;
        }
        g.play_tick(&mut bots).unwrap();
    }
    assert!(checked >= 15);
}

/// With a winning move available, search plays one, however highly the
/// network rates playing on (its values are clamped to what a finished game
/// can be worth, and an outright win is taken without searching).
#[test]
fn search_takes_an_outright_win() {
    use catanatron::mcts::{immediate_win, load_net, MctsConfig, MctsPlayer, Utility};
    use catanatron::players::{from_key, Player};
    let model = "python/runs/selfplay-smoke/model.ctnn";
    if !std::path::Path::new(model).exists() {
        eprintln!("skipping: no exported model at {model}");
        return;
    }
    let net = load_net(model).unwrap();
    let config = MctsConfig { simulations: 50, utility: Utility { margin: 0.1, speed: 0.1, speed_turns: 400.0 }, ..Default::default() };
    let mut found = 0;
    for seed in 0..40 {
        let mut g = Game::new(&COLORS[..2], &GameConfig { seed: Some(seed), ..Default::default() });
        let mut bots: Vec<Box<dyn Player>> = (0..2).map(|i| from_key("R", COLORS[i], seed * 3 + i as u64).unwrap()).collect();
        while !g.is_over() {
            if let Some(i) = immediate_win(&g) {
                let me = g.state.current_color();
                let mut m = MctsPlayer::new(me, net.clone(), config, seed);
                let actions = g.playable_actions.clone();
                let a = m.decide(&g, &actions);
                let mut after = g.copy_without_history();
                after.execute(a).unwrap();
                assert_eq!(after.winning_color(), Some(me), "search passed up {:?} for {:?}", actions[i], a);
                found += 1;
            }
            g.play_tick(&mut bots).unwrap();
        }
    }
    assert!(found > 10, "only {found} winning positions seen");
}
