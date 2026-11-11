//! First placement by outcomes instead of by the value head: at a player's
//! first settlement, take the top K spots by the network's prior, play N
//! fast network-only games (top move everywhere, dice vary) from each, and
//! place on the spot with the best empirical win rate.
//!
//!   cargo run --release --example rollout_opening -- board <model.ctnn> <board seed> [K] [N]
//!       the candidate table for the first seat's first placement on that board
//!   cargo run --release --example rollout_opening -- eval <model.ctnn> [boards] [K] [N] [threads] [opponent.ctnn] [opponent samples 0/1] [scope]
//!       scope: how many of the player's opening decisions use rollouts: 1 = first settlement (default),
//!       2 = both settlements, 4 = both settlements and both opening roads; append "+roads" and/or
//!       "+robber" to also pick every mid-game road / robber move by rollouts (top K by prior)
//!       rollout-opening player vs a plain network (itself by default; optionally another net,
//!       optionally sampling its moves from its prior) across random boards (seats alternate)
use catanatron::encoding::{ActionSpace, ObservationSpace};
use catanatron::map::{build_map, MapType, NumberPlacement};
use catanatron::mcts::load_net;
use catanatron::nn::{PolicyValueNet, Scratch};
use catanatron::parallel::{default_threads, par_map};
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::*;
use std::sync::Arc;

struct Ctx {
    net: Arc<PolicyValueNet>,
    obs_space: ObservationSpace,
    act_space: ActionSpace,
    obs: Vec<f32>,
    scratch: Scratch,
    logits: Vec<f32>,
}

impl Ctx {
    fn new(net: &Arc<PolicyValueNet>, map: &map::CatanMap) -> Ctx {
        Ctx { net: net.clone(), obs_space: ObservationSpace::for_network(map, net.obs_size), act_space: ActionSpace::new(map), obs: vec![0.0; net.obs_size], scratch: Scratch::default(), logits: Vec::new() }
    }
    /// Prior over the playable actions.
    fn prior(&mut self, game: &Game) -> Vec<f32> {
        let s = &game.state;
        self.obs_space.encode_into(s, s.current_color(), &mut self.obs);
        self.net.forward(&self.obs, &mut self.scratch, &mut self.logits);
        let l: Vec<f32> = game.playable_actions.iter().map(|a| self.logits[self.act_space.encode(s, a).expect("encodable")]).collect();
        let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = l.iter().map(|x| (x - m).exp()).collect();
        let z: f32 = e.iter().sum();
        e.iter().map(|x| x / z).collect()
    }
    fn top(&mut self, game: &Game) -> usize {
        let p = self.prior(game);
        (0..p.len()).max_by(|&i, &j| p[i].total_cmp(&p[j])).unwrap()
    }
    fn sample(&mut self, game: &Game, rng: &mut Rng) -> usize {
        let p = self.prior(game);
        catanatron::mcts::sample_index(&p, rng)
    }
    /// Plays `game` out with the network's top move for everyone; the winner.
    fn play_out(&mut self, mut game: Game) -> Option<Color> {
        while !game.is_over() && game.state.num_turns < 300 {
            let i = if game.playable_actions.len() == 1 { 0 } else { self.top(&game) };
            let a = game.playable_actions[i];
            game.execute(a).expect("legal");
        }
        game.winning_color()
    }
    /// Win rate of `mover` after placing at action `i` now, over `n` rollouts with fresh dice.
    fn rollout_value(&mut self, game: &Game, i: usize, n: u64, seed: u64) -> f32 {
        let mover = game.state.current_color();
        let mut wins = 0;
        for k in 0..n {
            let mut g = rerandomized(game, seed * 7919 + k);
            let a = g.playable_actions[i];
            g.execute(a).expect("legal");
            wins += (self.play_out(g) == Some(mover)) as u32;
        }
        wins as f32 / n as f32
    }
    /// The first settlement by outcomes: the best of the top `k` prior spots over `n` rollouts each.
    fn rollout_pick(&mut self, game: &Game, k: usize, n: u64, seed: u64) -> (usize, Vec<(usize, f32, f32)>) {
        let p = self.prior(game);
        let mut order: Vec<usize> = (0..p.len()).collect();
        order.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
        let mut table = Vec::new();
        for &i in order.iter().take(k) {
            let v = self.rollout_value(game, i, n, seed + i as u64);
            table.push((i, p[i], v));
        }
        let best = table.iter().max_by(|a, b| a.2.total_cmp(&b.2)).unwrap().0;
        (best, table)
    }
}

/// The same position with fresh randomness for everything still to come.
fn rerandomized(game: &Game, seed: u64) -> Game {
    let mut g = game.copy_without_history();
    let mut rng = Rng::new(seed);
    g.state.rng = Rng::new(rng.next_u64());
    rng.shuffle(&mut g.state.dev_deck);
    g
}

fn rules(board_seed: u64, game_seed: u64, map: &Arc<map::CatanMap>) -> GameConfig {
    GameConfig {
        map_type: MapType::Base, vps_to_win: 15, discard_limit: 9, friendly_robber: FriendlyRobber::PublicPoints,
        dice: DiceMode::OpenHex, number_placement: NumberPlacement::SpiralRandomCorner, seed: Some(game_seed ^ (board_seed << 20)),
        catan_map: Some(map.clone()), record_history: false, fixed_seats: true, ..Default::default()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args[1].as_str();
    let net = load_net(&args[2]).expect("model");
    match mode {
        "board" => {
            let board: u64 = args[3].parse().expect("board seed");
            let k: usize = args.get(4).map_or(6, |s| s.parse().expect("K"));
            let n: u64 = args.get(5).map_or(200, |s| s.parse().expect("N"));
            let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(board));
            let game = Game::new(&COLORS[..2], &rules(board, 1, &map));
            let mut ctx = Ctx::new(&net, &map);
            let (best, table) = ctx.rollout_pick(&game, k, n, 42);
            println!("board {board}, first seat's first settlement: top {k} spots by prior, {n} rollouts each");
            for (i, p, v) in &table {
                let ActionKind::BuildSettlement(nd) = game.playable_actions[*i].kind else { continue };
                println!("  node {nd:3}  prior {p:.3}  rollout win rate {v:.3}{}", if *i == best { "  <- pick" } else { "" });
            }
        }
        "eval" => {
            let boards: u64 = args.get(3).map_or(200, |s| s.parse().expect("boards"));
            let k: usize = args.get(4).map_or(6, |s| s.parse().expect("K"));
            let n: u64 = args.get(5).map_or(100, |s| s.parse().expect("N"));
            let threads: usize = args.get(6).map_or(default_threads(), |s| s.parse().expect("threads"));
            let opp_net = args.get(7).map(|p| load_net(p).expect("opponent")).unwrap_or_else(|| net.clone());
            let opp_samples = args.get(8).is_some_and(|s| s == "1");
            let scope_arg = args.get(9).map_or("1", |s| s.as_str());
            let scope: u32 = scope_arg.split('+').next().unwrap().parse().expect("scope");
            let (mid_roads, mid_robber) = (scope_arg.contains("+roads"), scope_arg.contains("+robber"));
            // Each board twice: the rollout player in each seat.
            let res = par_map(boards * 2, threads, |j| {
                let (board, seat) = (800_000 + j / 2, (j % 2) as usize);
                let map = build_map(MapType::Base, NumberPlacement::SpiralRandomCorner, &mut Rng::new(board));
                let mut game = Game::new(&COLORS[..2], &rules(board, 1000 + j, &map));
                let me = COLORS[seat];
                let mut ctx = Ctx::new(&net, &map);
                let mut opp = Ctx::new(&opp_net, &map);
                let mut rng = Rng::new(j ^ 0xB0A7);
                let (mut settlements, mut roads, mut changed) = (0u32, 0u32, 0u32);
                while !game.is_over() && game.state.num_turns < 400 {
                    let mine = game.state.current_color() == me;
                    let opening = game.state.is_initial_build_phase;
                    let by_rollout = mine && game.playable_actions.len() > 1 && match game.playable_actions[0].kind {
                        ActionKind::BuildSettlement(_) if opening => { settlements += 1; settlements <= scope.min(2) }
                        ActionKind::BuildRoad(_) if opening => { roads += 1; scope >= 4 }
                        // Mid-game: a turn whose top prior move is a road, or a robber move.
                        ActionKind::MoveRobber { .. } => mid_robber,
                        _ if mid_roads && !opening => matches!(game.playable_actions[ctx.top(&game)].kind, ActionKind::BuildRoad(_)),
                        _ => false,
                    };
                    let i = if game.playable_actions.len() == 1 {
                        0
                    } else if by_rollout {
                        let top = ctx.top(&game);
                        let (best, _) = ctx.rollout_pick(&game, k, n, j * 31 + 7 + settlements as u64 * 3 + roads as u64);
                        changed += (best != top) as u32;
                        best
                    } else if mine {
                        ctx.top(&game)
                    } else if opp_samples {
                        opp.sample(&game, &mut rng)
                    } else {
                        opp.top(&game)
                    };
                    let a = game.playable_actions[i];
                    game.execute(a).expect("legal");
                }
                ((game.winning_color() == Some(me)) as u32, changed)
            });
            let wins: u32 = res.iter().map(|r| r.0).sum();
            let changed: u32 = res.iter().map(|r| r.1).sum();
            let g = (boards * 2) as f32;
            println!("rollout-opening (scope {scope_arg}) {} vs {}{}: {:.1}% of {} games (±{:.1}); changed {:.2} opening decisions per game",
                     args[2], args.get(7).map_or(args[2].as_str(), |s| s.as_str()), if opp_samples { " (sampling)" } else { "" },
                     100.0 * wins as f32 / g, boards * 2, 100.0 * (0.25 / g).sqrt(), 100.0 * changed as f32 / g);
        }
        other => panic!("mode {other}: board or eval"),
    }
}
