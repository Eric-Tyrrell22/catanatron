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
    if std::env::args().nth(2).is_some_and(|a| a == "single") {
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
