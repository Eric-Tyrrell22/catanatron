//! Cost of observation encoding and action masking per decision.
use std::time::Instant;
use catanatron::encoding::{ActionSpace, ObservationSpace};
use catanatron::players::from_key;
use catanatron::*;

fn main() {
    let mut g = Game::new(&COLORS, &GameConfig { seed: Some(1), ..Default::default() });
    let mut p: Vec<_> = COLORS.iter().enumerate().map(|(i, &c)| from_key("R", c, i as u64).unwrap()).collect();
    for _ in 0..300 { g.play_tick(&mut p).unwrap(); }
    let obs = ObservationSpace::new(&g.state.board.map);
    let space = ActionSpace::new(&g.state.board.map);
    let mut buf = vec![0.0f32; obs.len()];
    let mut mask = vec![false; space.len()];
    let n = 200_000;
    let t = Instant::now();
    for _ in 0..n { obs.encode_into(std::hint::black_box(&g.state), g.state.current_color(), &mut buf); }
    println!("observation ({} floats): {:.0} ns", obs.len(), t.elapsed().as_nanos() as f64 / n as f64);
    let t = Instant::now();
    for _ in 0..n { space.fill_mask(std::hint::black_box(&g.state), &g.playable_actions, &mut mask); }
    println!("mask ({} slots, {} legal): {:.0} ns", space.len(), g.playable_actions.len(), t.elapsed().as_nanos() as f64 / n as f64);
}
