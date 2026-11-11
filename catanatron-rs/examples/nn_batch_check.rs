//! Checks that a network's batched forward pass (`forward_batch`, used by
//! self-play) gives the same results as one `forward_sparse` per position,
//! on positions from random OpenHex 1v1 games.
//!
//!   cargo run --release --example nn_batch_check -- <model.ctnn>
use catanatron::encoding::ObservationSpace;
use catanatron::map::{MapType, NumberPlacement};
use catanatron::nn::{BatchScratch, PolicyValueNet, Scratch, MAX_BATCH};
use catanatron::players::{Player, RandomPlayer};
use catanatron::state::DiceMode;
use catanatron::*;

fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let net = PolicyValueNet::load(&path).expect("model");
    let mut obs: Vec<Vec<(u32, f32)>> = Vec::new();
    for seed in 0..20u64 {
        let rules = GameConfig {
            map_type: MapType::Base,
            vps_to_win: 15,
            discard_limit: 9,
            friendly_robber: FriendlyRobber::PublicPoints,
            dice: DiceMode::OpenHex,
            number_placement: NumberPlacement::SpiralRandomCorner,
            seed: Some(seed),
            record_history: false,
            ..Default::default()
        };
        let mut g = Game::new(&COLORS[..2], &rules);
        let space = ObservationSpace::for_network(&g.state.board.map, net.obs_size);
        let mut bots: Vec<RandomPlayer> = COLORS[..2].iter().map(|&c| RandomPlayer::new(c, seed)).collect();
        let mut step = 0;
        while !g.is_over() && step < 400 {
            if step % 10 == 0 {
                let mut nz = Vec::new();
                let n = space.encode_sparse_into(&g.state, g.state.current_color(), &mut nz);
                nz.truncate(n);
                obs.push(nz);
            }
            let c = g.state.current_color();
            let bot = bots.iter_mut().find(|b| b.color() == c).unwrap();
            let actions = g.playable_actions.clone();
            let a = bot.decide(&g, &actions);
            g.execute(a).expect("legal");
            step += 1;
        }
    }
    let mut scratch = Scratch::default();
    let mut batch = BatchScratch::default();
    let (mut max_dl, mut max_dv) = (0f32, 0f32);
    for chunk in obs.chunks(MAX_BATCH) {
        let xs: Vec<&[(u32, f32)]> = chunk.iter().map(|o| o.as_slice()).collect();
        let mut logits = vec![Vec::new(); chunk.len()];
        let mut values = vec![0f32; chunk.len()];
        net.forward_batch(&xs, &mut batch, &mut logits, &mut values);
        for (b, o) in chunk.iter().enumerate() {
            let mut l = Vec::new();
            let v = net.forward_sparse(o, &mut scratch, &mut l);
            max_dv = max_dv.max((v - values[b]).abs());
            for (x, y) in l.iter().zip(&logits[b]) {
                max_dl = max_dl.max((x - y).abs());
            }
        }
    }
    println!("{} positions: max |dlogit| {max_dl}, max |dvalue| {max_dv}", obs.len());
}
