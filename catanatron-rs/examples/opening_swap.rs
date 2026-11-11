//! How much of net B's edge over net A on a board is its openings? Plays
//! A vs B network-only (openings sampled from the controlling net's prior,
//! the rest of the game by top move) under four regimes: free (each side
//! places its own), both sides' placements chosen by A, both by B, and
//! reports A's win rate in each. Seats alternate per game.
//!
//! A fourth regime gives A's own placements to a search of `search sims`
//! (B still places by prior): how much of the gap play-time opening search recovers.
//!
//!   cargo run --release --example opening_swap -- <A.ctnn> <B.ctnn> <board seed> [games per regime] [threads] [search sims] [opener.ctnn]
//! With an opener net, one more regime: A's placements are chosen by the opener
//! (B's by B itself), both sides then play with their own nets. With A = B this
//! measures the opener's placements alone, under identical play.
use catanatron::encoding::{ActionSpace, ObservationSpace};
use catanatron::map::{build_map, MapType, NumberPlacement};
use catanatron::mcts::{load_net, MctsConfig, MctsPlayer, Utility};
use catanatron::nn::{PolicyValueNet, Scratch};
use catanatron::parallel::{default_threads, par_map};
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::*;
use std::sync::Arc;

fn pick(net: &PolicyValueNet, obs_space: &ObservationSpace, act_space: &ActionSpace, game: &Game, sample: bool, rng: &mut Rng, obs: &mut [f32], scratch: &mut Scratch, logits: &mut Vec<f32>) -> usize {
    let s = &game.state;
    obs_space.encode_into(s, s.current_color(), obs);
    net.forward(obs, scratch, logits);
    let actions = &game.playable_actions;
    let slots: Vec<usize> = actions.iter().map(|a| act_space.encode(s, a).expect("encodable")).collect();
    if !sample {
        return (0..slots.len()).max_by(|&i, &j| logits[slots[i]].total_cmp(&logits[slots[j]])).unwrap();
    }
    let m = slots.iter().map(|&s| logits[s]).fold(f32::NEG_INFINITY, f32::max);
    let w: Vec<f32> = slots.iter().map(|&s| (logits[s] - m).exp()).collect();
    catanatron::mcts::sample_index(&w, rng)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let a = load_net(&args[1]).expect("A");
    let b = load_net(&args[2]).expect("B");
    let board: u64 = args[3].parse().expect("board seed");
    let games: u64 = args.get(4).map_or(2000, |s| s.parse().expect("games"));
    let threads: usize = args.get(5).map_or(default_threads(), |s| s.parse().expect("threads"));
    let search_sims: u32 = args.get(6).map_or(2400, |s| s.parse().expect("search sims"));
    let opener = args.get(7).map(|p| load_net(p).expect("opener"));
    let search = MctsConfig { simulations: search_sims, c_puct: 1.0, fpu_reduction: 0.25, utility: Utility { margin: 0.1, speed: 0.3, speed_turns: 150.0 }, ..Default::default() };
    let mut regimes = vec!["free", "openings by A", "openings by B", "A places by search", "crossed (A gets B's picks, B gets A's)"];
    if opener.is_some() {
        regimes.push("A placed by the opener net, B by itself");
    }
    for (ri, name) in regimes.iter().enumerate() {
        let wins = par_map(games, threads, |g| {
            let seed = 920_000 + g;
            let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(board));
            let rules = GameConfig {
                map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
                dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(seed),
                catan_map: Some(map.clone()), record_history: false, fixed_seats: true, ..Default::default()
            };
            let mut game = Game::new(&COLORS[..2], &rules);
            let a_color = COLORS[(g % 2) as usize];
            let obs_space = ObservationSpace::for_network(&map, a.obs_size);
            let act_space = ActionSpace::new(&map);
            let mut obs = vec![0.0f32; a.obs_size];
            let mut scratch = Scratch::default();
            let mut logits = Vec::new();
            let mut rng = Rng::new(seed ^ 0xC0FFEE);
            while !game.is_over() && game.state.num_turns < 400 {
                let mover = game.state.current_color();
                let opening = game.state.is_initial_build_phase;
                let own: &Arc<PolicyValueNet> = if mover == a_color { &a } else { &b };
                let other: &Arc<PolicyValueNet> = if mover == a_color { &b } else { &a };
                let net = match (opening, ri) {
                    (true, 1) => &a,
                    (true, 2) => &b,
                    (true, 4) => other,
                    (true, 5) if mover == a_color => opener.as_ref().unwrap(),
                    _ => own,
                };
                let i = if game.playable_actions.len() == 1 {
                    0
                } else if opening && ri == 3 && mover == a_color {
                    let mut m = MctsPlayer::new(mover, a.clone(), search.clone(), seed * 100 + game.state.num_turns as u64);
                    let visits = m.search(&game).expect("search");
                    (0..visits.len()).max_by_key(|&j| visits[j]).unwrap()
                } else {
                    // A network with an opening network attached ("main@opening") places with that one.
                    pick(if opening { net.opening.as_deref().unwrap_or(&**net) } else { &**net }, &obs_space, &act_space, &game, opening, &mut rng, &mut obs, &mut scratch, &mut logits)
                };
                let action = game.playable_actions[i];
                game.execute(action).expect("legal");
            }
            (game.winning_color() == Some(a_color)) as u32
        });
        let w: u32 = wins.iter().sum();
        println!("{name:<14}: A wins {:.1}% of {games} (±{:.1})", 100.0 * w as f32 / games as f32, 100.0 * (0.25 / games as f32).sqrt());
    }
}
