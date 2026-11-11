//! Rough per-phase timing of the game loop (perf is unavailable here).
use std::collections::HashMap;
use std::time::{Duration, Instant};

use catanatron::actions::generate_playable_actions;
use catanatron::apply::apply_action;
use catanatron::players::from_key;
use catanatron::*;

fn main() {
    let n: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let mut t: HashMap<&str, Duration> = HashMap::new();
    let mut gen_by_prompt: HashMap<String, (Duration, u64)> = HashMap::new();
    let mut apply_by_kind: HashMap<String, (Duration, u64)> = HashMap::new();
    let total = Instant::now();
    for seed in 0..n {
        let i0 = Instant::now();
        let mut g = Game::new(&COLORS, &GameConfig { seed: Some(seed), ..Default::default() });
        *t.entry("new").or_default() += i0.elapsed();
        let mut players: Vec<_> = COLORS.iter().enumerate().map(|(i, &c)| from_key("R", c, seed * 31 + i as u64).unwrap()).collect();
        while !g.is_over() {
            let c = g.state.current_color();
            let p = players.iter_mut().find(|p| p.color() == c).unwrap();
            let i0 = Instant::now();
            let a = p.decide(&g, &g.playable_actions);
            *t.entry("decide").or_default() += i0.elapsed();
            let i0 = Instant::now();
            assert!(g.is_valid_action(&a));
            *t.entry("validate").or_default() += i0.elapsed();
            let i0 = Instant::now();
            apply_action(&mut g.state, a, None).unwrap();
            let d = i0.elapsed();
            let k = format!("{:?}", a.kind).split(|c: char| !c.is_alphanumeric()).next().unwrap().to_string();
            let e = apply_by_kind.entry(k).or_default(); e.0 += d; e.1 += 1;
            *t.entry("apply").or_default() += d;
            let i0 = Instant::now();
            g.playable_actions = generate_playable_actions(&g.state);
            let d = i0.elapsed();
            let e = gen_by_prompt.entry(format!("{:?}", g.state.current_prompt)).or_default(); e.0 += d; e.1 += 1;
            *t.entry("generate").or_default() += d;
            let i0 = Instant::now();
            let _ = g.is_over();
            *t.entry("is_over").or_default() += i0.elapsed();
        }
    }
    let all = total.elapsed();
    println!("total {:.2}s  {:.0} games/s", all.as_secs_f64(), n as f64 / all.as_secs_f64());
    let mut v: Vec<_> = t.into_iter().collect(); v.sort_by_key(|x| std::cmp::Reverse(x.1));
    for (k, d) in v { println!("  {:<10} {:>6.1}%", k, 100.0 * d.as_secs_f64() / all.as_secs_f64()); }
    for (name, m) in [("generate by prompt", gen_by_prompt), ("apply by action", apply_by_kind)] {
        println!("{name}:");
        let mut v: Vec<_> = m.into_iter().collect(); v.sort_by_key(|x| std::cmp::Reverse(x.1 .0));
        for (k, (d, c)) in v.iter().take(8) { println!("  {:<24} {:>6.1}%  {:>9} calls  {:>6.0} ns/call", k, 100.0 * d.as_secs_f64() / all.as_secs_f64(), c, d.as_nanos() as f64 / *c as f64); }
    }
}
