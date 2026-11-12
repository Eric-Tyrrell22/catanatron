//! Checks a network file's three inference paths agree (full sparse forward,
//! legal-slots-only forward, batched forward) on real positions: the first
//! placement of a board, and the second seat's reply after a settlement. Prints
//! the largest logit difference between the paths; for a version 3 file this
//! exercises the spot head on every path.
//!
//!   cargo run --release --example head_check -- <net.ctnn> [board seed] [plies=3]  (plies > 8 adds random mid-game positions)
use catanatron::encoding::{ActionSpace, ObservationSpace};
use catanatron::map::{build_map, MapType, NumberPlacement};
use catanatron::nn::{BatchScratch, PolicyValueNet, Scratch};
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let net = PolicyValueNet::load(&args[1]).expect("network");
    let board: u64 = args.get(2).map_or(1030001, |s| s.parse().expect("board seed"));
    let plies: usize = args.get(3).map_or(3, |s| s.parse().expect("plies"));
    let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(board));
    let rules = GameConfig {
        map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
        dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(1),
        catan_map: Some(map.clone()), record_history: false, fixed_seats: true, ..Default::default()
    };
    let obs_space = ObservationSpace::for_network(&map, net.obs_size);
    let act_space = ActionSpace::new(&map);
    let mut game = Game::new(&COLORS[..2], &rules);
    let mut positions = Vec::new();
    let mut rng = Rng::new(board);
    for step in 0..plies {
        if game.is_over() { break; }
        let s = &game.state;
        let mut nz = Vec::new();
        let n = obs_space.encode_sparse_into(s, s.current_color(), &mut nz);
        nz.truncate(n);
        let slots: Vec<u32> = game.playable_actions.iter().map(|a| act_space.encode(s, a).map_or(u32::MAX, |x| x as u32)).collect();
        positions.push((nz, slots));
        let k = (rng.next_u64() % game.playable_actions.len() as u64) as usize;
        let a = game.playable_actions[if step < 8 { step % game.playable_actions.len() } else { k }];
        game.execute(a).expect("legal");
    }
    // Only positions with several legal moves are interesting; keep at most 12, spread over the game.
    if positions.len() > 12 {
        let step = positions.len() / 12;
        positions = positions.into_iter().enumerate().filter(|(i, _)| i % step == 0).map(|(_, p)| p).take(12).collect();
    }
    let mut scratch = Scratch::default();
    // Speed: full sparse forwards on the last position.
    {
        let (nz, _) = positions.last().unwrap();
        let mut lg = Vec::new();
        let iters: usize = std::env::var("HEAD_CHECK_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(500);
        let t0 = std::time::Instant::now();
        for _ in 0..iters { net.forward_sparse(nz, &mut scratch, &mut lg); }
        println!("forward_sparse: {:.1} us per call ({iters} calls)", t0.elapsed().as_secs_f64() * 1e6 / iters as f64);
        let (_, slots) = positions.last().unwrap();
        for (name, sl) in [("roll slot only (heads skipped)", vec![0u32]), ("the position's legal slots", slots.clone())] {
            let t0 = std::time::Instant::now();
            for _ in 0..iters { net.forward_sparse_for(nz, &sl, &mut scratch, &mut lg); }
            println!("forward_sparse_for, {name}: {:.1} us per call", t0.elapsed().as_secs_f64() * 1e6 / iters as f64);
        }
    }
    let mut batch = BatchScratch::default();
    let mut worst = 0.0f32;
    let mut worst_batch = 0.0f32;
    let mut full: Vec<Vec<f32>> = vec![Vec::new(); positions.len()];
    let mut values = vec![0.0f32; positions.len()];
    for (k, (nz, slots)) in positions.iter().enumerate() {
        let v1 = net.forward_sparse(nz, &mut scratch, &mut full[k]);
        let mut sub = Vec::new();
        let v2 = net.forward_sparse_for(nz, slots, &mut scratch, &mut sub);
        for (&s, &l) in slots.iter().zip(&sub) {
            if s != u32::MAX {
                worst = worst.max((full[k][s as usize] - l).abs());
            }
        }
        worst = worst.max((v1 - v2).abs());
        println!("position {k}: {} legal, value {v1:+.4}, top legal logit {:+.4}", slots.len(),
                 slots.iter().filter(|&&s| s != u32::MAX).map(|&s| full[k][s as usize]).fold(f32::NEG_INFINITY, f32::max));
    }
    positions.truncate(8); // the batched path takes at most MAX_BATCH positions
    full.truncate(8);
    let refs: Vec<&[(u32, f32)]> = positions.iter().map(|p| p.0.as_slice()).collect();
    let mut blog: Vec<Vec<f32>> = vec![Vec::new(); positions.len()];
    values.truncate(positions.len());
    net.forward_batch(&refs, &mut batch, &mut blog, &mut values);
    for k in 0..positions.len() {
        let (mut wk, mut at) = (0.0f32, 0usize);
        for (i, (a, b)) in full[k].iter().zip(&blog[k]).enumerate() {
            if (a - b).abs() > wk { wk = (a - b).abs(); at = i; }
        }
        println!("position {k}: batched vs sparse worst {wk:.2e} at slot {at} (sparse {:+.3}, batched {:+.3}); value {:+.4} vs {:+.4}", full[k][at], blog[k][at], values[k], net.forward_sparse(&positions[k].0, &mut scratch, &mut Vec::new()));
        worst_batch = worst_batch.max(wk);
    }
    println!("subset vs sparse: {worst:.2e}; batched vs sparse: {worst_batch:.2e}");
    // The batched path per position alone, and in pairs.
    for k in 0..positions.len() {
        let mut one = vec![Vec::new()];
        let mut v1 = [0.0f32];
        net.forward_batch(&[positions[k].0.as_slice()], &mut batch, &mut one, &mut v1);
        let d = full[k].iter().zip(&one[0]).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("position {k} batched alone: worst {d:.2e}; nonzero inputs {}", positions[k].0.len());
    }
}
