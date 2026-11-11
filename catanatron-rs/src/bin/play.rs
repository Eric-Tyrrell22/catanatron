//! Simulate many games between bots and print win statistics.
//!
//!   catanatron-play --num 1000 --players R,W,VP,R --seed 1 --map BASE

use std::process::exit;
use std::time::Instant;

use catanatron::map::{MapType, NumberPlacement};
use catanatron::players::from_key;
use catanatron::{Color, Game, GameConfig, COLORS};

const USAGE: &str = "\
Usage: catanatron-play [options]

  -n, --num N             games to play (default 5)
  -p, --players LIST      comma-separated bot keys, 2-4 (default R,R,R,R)
                            R  = random
                            S  = simple (first action)
                            W  = weighted random
                            VP = greedy victory points
                            F  = value function (Python F)
                            AB = alphabeta, depth 2 (AB:3 for depth 3)
                            NN:<model.ctnn>          = trained network, no search
                            MCTS:<sims>:<model.ctnn> = network-guided search
  -s, --seed N            base seed; game i uses seed+i (default: random)
      --map TYPE          BASE | MINI | TOURNAMENT (default BASE)
      --random-numbers    shuffle number tokens instead of official spiral
      --vps N             victory points to win (default 10)
      --discard-limit N   default 7
      --friendly-robber   robber can't target players under 3 VP
  -t, --threads N         worker threads (default: all cores)
  -h, --help";

fn fail(msg: &str) -> ! {
    eprintln!("error: {msg}\n\n{USAGE}");
    exit(2)
}

fn main() {
    let mut num = 5u64;
    let mut keys = vec!["R".to_string(); 4];
    let mut seed: Option<u64> = None;
    let mut threads = catanatron::parallel::default_threads();
    let mut config = GameConfig { record_history: false, ..GameConfig::default() };

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) => (f.to_string(), Some(v.to_string())),
            None => (arg.clone(), None),
        };
        let mut value = || {
            inline.clone().or_else(|| args.next()).unwrap_or_else(|| fail(&format!("{flag} needs a value")))
        };
        let parse = |v: String| v.parse::<u64>().unwrap_or_else(|_| fail(&format!("bad number: {v}")));
        match flag.as_str() {
            "-n" | "--num" => num = parse(value()),
            "-p" | "--players" => keys = value().split(',').map(|s| s.trim().to_string()).collect(),
            "-s" | "--seed" => seed = Some(parse(value())),
            "--map" => {
                config.map_type = match value().to_uppercase().as_str() {
                    "BASE" => MapType::Base,
                    "MINI" => MapType::Mini,
                    "TOURNAMENT" => MapType::Tournament,
                    other => fail(&format!("unknown map {other}")),
                }
            }
            "--random-numbers" => config.number_placement = NumberPlacement::Random,
            "--vps" => config.vps_to_win = parse(value()) as u32,
            "--discard-limit" => config.discard_limit = parse(value()) as u32,
            "--friendly-robber" => config.friendly_robber = catanatron::FriendlyRobber::ActualPoints,
            "--friendly-robber-public" => config.friendly_robber = catanatron::FriendlyRobber::PublicPoints,
            "--balanced-dice" => config.dice = catanatron::state::DiceMode::Deck,
            "--openhex-dice" => config.dice = catanatron::state::DiceMode::OpenHex,
            "-t" | "--threads" => threads = parse(value()).max(1) as usize,
            "-h" | "--help" => {
                println!("{USAGE}");
                return;
            }
            other => fail(&format!("unknown option {other}")),
        }
    }
    if !(2..=4).contains(&keys.len()) {
        fail("need 2 to 4 players");
    }
    let colors: Vec<Color> = COLORS[..keys.len()].to_vec();
    for k in &keys {
        if from_key(k, Color::Red, 0).is_none() {
            fail(&format!("unknown player key {k}"));
        }
    }
    let base_seed = seed.unwrap_or_else(|| catanatron::rng::Rng::from_entropy().next_u64());

    let start = Instant::now();

    // One game per index; everything about game i derives from its seed,
    // so results don't depend on the thread count.
    let results = catanatron::parallel::par_map(num, threads, |i| {
        let game_seed = base_seed.wrapping_add(i);
        let mut players: Vec<_> = keys
            .iter()
            .zip(&colors)
            .enumerate()
            .map(|(j, (k, &c))| from_key(k, c, game_seed.wrapping_mul(31).wrapping_add(j as u64 + 1)).unwrap())
            .collect();
        let config = GameConfig { seed: Some(game_seed), ..config.clone() };
        let mut game = Game::new(&colors, &config);
        let winner = game.play(&mut players).map_err(|e| format!("game with seed {game_seed} failed: {e}"))?;
        Ok::<_, String>((winner, game.state.num_turns as u64, game.state.num_actions as u64))
    });

    let mut wins = vec![0u64; colors.len()];
    let mut no_winner = 0u64;
    let mut total_turns = 0u64;
    let mut total_ticks = 0u64;
    for r in results {
        let (winner, turns, actions) = r.unwrap_or_else(|e| {
            eprintln!("{e}");
            exit(1)
        });
        match winner {
            Some(w) => wins[colors.iter().position(|&c| c == w).unwrap()] += 1,
            None => no_winner += 1,
        }
        total_turns += turns;
        total_ticks += actions;
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!("{:<8} {:<6} {:>8} {:>8}", "COLOR", "BOT", "WINS", "WIN%");
    for (j, c) in colors.iter().enumerate() {
        println!(
            "{:<8} {:<6} {:>8} {:>7.1}%",
            c.name(),
            keys[j],
            wins[j],
            100.0 * wins[j] as f64 / num.max(1) as f64
        );
    }
    println!("{:<15} {:>8}", "no winner", no_winner);
    println!();
    println!("games: {num}  seed: {base_seed}  threads: {threads}");
    println!(
        "avg turns: {:.1}  avg actions: {:.1}",
        total_turns as f64 / num.max(1) as f64,
        total_ticks as f64 / num.max(1) as f64
    );
    println!("time: {:.2}s  ({:.0} games/s)", elapsed, num as f64 / elapsed.max(1e-9));
}
