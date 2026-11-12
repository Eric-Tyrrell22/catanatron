//! Single-threaded cost of one network evaluation on real observations.
use catanatron::encoding::ObservationSpace;
use catanatron::mcts::load_net;
use catanatron::nn::Scratch;
use catanatron::players::from_key;
use catanatron::*;

fn main() {
    let net = load_net(&std::env::args().nth(1).expect("model path")).unwrap();
    let mut g = Game::new(&COLORS[..2], &GameConfig { seed: Some(1), ..Default::default() });
    let mut bots: Vec<_> = (0..2).map(|i| from_key("F", COLORS[i], i as u64).unwrap()).collect();
    let space = ObservationSpace::for_network(&g.state.board.map, net.obs_size);
    let mut obs_set = Vec::new();
    while !g.is_over() {
        obs_set.push(space.encode(&g.state, g.state.current_color()));
        g.play_tick(&mut bots).unwrap();
    }
    let nonzero: f64 = obs_set.iter().map(|o| o.iter().filter(|&&x| x != 0.0).count() as f64).sum::<f64>() / obs_set.len() as f64;
    let (mut scratch, mut logits) = (Scratch::default(), Vec::new());
    let n = 20_000;
    let t = std::time::Instant::now();
    let mut acc = 0.0;
    for i in 0..n {
        acc += net.forward(std::hint::black_box(&obs_set[i % obs_set.len()]), &mut scratch, &mut logits);
    }
    println!("{:.1} us per evaluation ({:.0} of {} inputs nonzero on average) [{acc:.1}]",
        t.elapsed().as_secs_f64() * 1e6 / n as f64, nonzero, space.len());
    let mode = std::env::args().nth(2).unwrap_or_default();
    if mode == "single" {
        return;
    }
    if mode == "cross" {
        cross_game(&net);
        return;
    }

    // Batches of consecutive positions (similar, like leaves of one search).
    let sparse: Vec<Vec<(u32, f32)>> = obs_set
        .iter()
        .map(|o| o.iter().enumerate().filter(|(_, &x)| x != 0.0).map(|(i, &x)| (i as u32, x)).collect())
        .collect();
    let mut batch = catanatron::nn::BatchScratch::default();
    let mut blogits = vec![Vec::new(); catanatron::nn::MAX_BATCH];
    let mut values = [0.0; catanatron::nn::MAX_BATCH];
    for size in [1, 2, 4, 8] {
        let t = std::time::Instant::now();
        let mut acc = 0.0;
        let mut i = 0;
        while i < n {
            let start = i % (sparse.len() - size);
            let xs: Vec<&[(u32, f32)]> = sparse[start..start + size].iter().map(|p| &p[..]).collect();
            net.forward_batch(std::hint::black_box(&xs), &mut batch, &mut blogits, &mut values);
            acc += values[0];
            i += size;
        }
        println!("batch {size}: {:.1} us per evaluation [{acc:.1}]", t.elapsed().as_secs_f64() * 1e6 / n as f64);
    }
}

/// Batches of positions from *different* games, as cross-game lockstep
/// batching would form them: per-position cost of the single sparse pass
/// (legal-moves policy head) against `forward_batch` at batch sizes 4 and
/// 8, on the same positions, for games on different boards and for games on
/// one board (different dice and seats). Positions are taken at the same
/// decision index of each game, so they are at similar stages. The modes
/// are interleaved over rounds and the median per-position time reported,
/// with the size of the first layer's input union relative to the summed
/// per-position inputs (1 = no sharing at all).
fn cross_game(net: &catanatron::nn::PolicyValueNet) {
    use catanatron::encoding::ActionSpace;
    const GAMES: usize = 8;
    const ROUNDS: usize = 9;
    let board = Game::new(&COLORS[..2], &GameConfig { seed: Some(100), ..Default::default() }).state.board.map.clone();
    for (label, same_board) in [("different boards", false), ("same board", true)] {
        let mut per_game: Vec<Vec<(Vec<(u32, f32)>, Vec<u32>)>> = Vec::new();
        for seed in 0..GAMES as u64 {
            let catan_map = same_board.then(|| board.clone());
            let mut g = Game::new(&COLORS[..2], &GameConfig { seed: Some(100 + seed), catan_map, ..Default::default() });
            let mut bots: Vec<_> = (0..2).map(|i| from_key("F", COLORS[i], seed * 2 + i as u64).unwrap()).collect();
            let space = ObservationSpace::for_network(&g.state.board.map, net.obs_size);
            let act = ActionSpace::new(&g.state.board.map);
            let mut positions = Vec::new();
            while !g.is_over() {
                if g.playable_actions.len() > 1 {
                    let mut nz = Vec::new();
                    let n = space.encode_sparse_into(&g.state, g.state.current_color(), &mut nz);
                    nz.truncate(n);
                    let slots: Vec<u32> =
                        g.playable_actions.iter().map(|a| act.encode(&g.state, a).map_or(u32::MAX, |i| i as u32)).collect();
                    positions.push((nz, slots));
                }
                g.play_tick(&mut bots).unwrap();
            }
            per_game.push(positions);
        }
        let shortest = per_game.iter().map(|p| p.len()).min().unwrap();
        // Input-union ratio for groups of 4 and 8.
        for size in [4usize, 8] {
            let (mut union, mut total) = (0usize, 0usize);
            for k in 0..shortest {
                for group in per_game.chunks(size) {
                    let mut set = std::collections::HashSet::new();
                    for g in group {
                        total += g[k].0.len();
                        set.extend(g[k].0.iter().map(|&(i, _)| i));
                    }
                    union += set.len();
                }
            }
            println!("{label}: batch {size} first-layer union / summed inputs = {:.2}", union as f64 / total as f64);
        }
        let (mut scratch, mut logits) = (Scratch::default(), Vec::new());
        let mut batch = catanatron::nn::BatchScratch::default();
        let mut blogits = vec![Vec::new(); catanatron::nn::MAX_BATCH];
        let mut values = [0.0; catanatron::nn::MAX_BATCH];
        let mut times: [Vec<f64>; 3] = Default::default();
        let mut acc = [0.0f32; 3];
        for _ in 0..ROUNDS {
            let t = std::time::Instant::now();
            for k in 0..shortest {
                for game in &per_game {
                    let (nz, slots) = &game[k];
                    acc[0] += net.forward_sparse_for(std::hint::black_box(nz), slots, &mut scratch, &mut logits);
                }
            }
            times[0].push(t.elapsed().as_secs_f64() * 1e6 / (shortest * GAMES) as f64);
            for (m, size) in [(1usize, 4usize), (2, 8)] {
                let t = std::time::Instant::now();
                for k in 0..shortest {
                    for group in per_game.chunks(size) {
                        let xs: Vec<&[(u32, f32)]> = group.iter().map(|g| &g[k].0[..]).collect();
                        net.forward_batch(std::hint::black_box(&xs), &mut batch, &mut blogits, &mut values);
                        acc[m] += values[0];
                    }
                }
                times[m].push(t.elapsed().as_secs_f64() * 1e6 / (shortest * GAMES) as f64);
            }
        }
        let median = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.total_cmp(b));
            v[v.len() / 2]
        };
        let single = median(&mut times[0]);
        let (b4, b8) = (median(&mut times[1]), median(&mut times[2]));
        println!(
            "{label}: {} positions x {GAMES} games, median of {ROUNDS} rounds: single {single:.1} us, batch 4 {b4:.1} us ({:.2}x), batch 8 {b8:.1} us ({:.2}x) [{:.1} {:.1} {:.1}]",
            shortest, single / b4, single / b8, acc[0], acc[1], acc[2]
        );
    }
}
