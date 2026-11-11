//! Tree reuse check: a searcher that keeps its tree between moves plays the
//! same searcher without it (same simulations per move), on OpenHex 1v1
//! rules. Prints the reuser's win rate and how much tree it kept: the share
//! of its searches that started from a kept node, and the visits that node
//! already had.
//!
//!   cargo run --release --example tree_reuse -- <model.ctnn> [games] [sims] [threads]
use std::sync::{Arc, Mutex};

use catanatron::map::{MapType, NumberPlacement};
use catanatron::mcts::{load_net, MctsConfig, MctsPlayer, Utility};
use catanatron::parallel::{default_threads, par_map};
use catanatron::players::Player;
use catanatron::state::DiceMode;
use catanatron::*;

/// The reuser, reporting its counts after every decision.
struct Probe {
    inner: MctsPlayer,
    /// (decisions with a choice, searches from a kept node, visits kept)
    out: Arc<Mutex<(u64, u64, u64)>>,
    counted: (u64, u64),
}

impl Player for Probe {
    fn color(&self) -> Color {
        self.inner.color
    }
    fn decide(&mut self, game: &Game, actions: &[Action]) -> Action {
        let a = self.inner.decide(game, actions);
        let (searches, visits) = self.inner.reuse_stats();
        let mut out = self.out.lock().unwrap();
        out.0 += (actions.len() > 1) as u64;
        out.1 += searches - self.counted.0;
        out.2 += visits - self.counted.1;
        self.counted = (searches, visits);
        a
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = args.get(1).expect("model path");
    let games: u64 = args.get(2).map_or(200, |s| s.parse().expect("games"));
    let sims: u32 = args.get(3).map_or(1000, |s| s.parse().expect("sims"));
    let threads: usize = args.get(4).map_or(default_threads(), |s| s.parse().expect("threads"));
    let net = load_net(model).expect("model");
    // Ham's live search settings.
    let config = MctsConfig {
        simulations: sims,
        c_puct: 1.0,
        fpu_reduction: 0.25,
        utility: Utility { margin: 0.1, speed: 0.3, speed_turns: 150.0 },
        ..Default::default()
    };
    let out = Arc::new(Mutex::new((0u64, 0u64, 0u64)));
    let results = par_map(games, threads, |i| {
        let seed = 70_000 + i;
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
        let reuse = MctsConfig { tree_reuse: true, ..config.clone() };
        // Alternate seats: the reuser is the first color in even games.
        let (a, b) = if i % 2 == 0 { (COLORS[0], COLORS[1]) } else { (COLORS[1], COLORS[0]) };
        let probe = Probe { inner: MctsPlayer::new(a, net.clone(), reuse, seed * 2), out: out.clone(), counted: (0, 0) };
        let plain = MctsPlayer::new(b, net.clone(), config.clone(), seed * 2 + 1);
        let mut players: Vec<Box<dyn Player>> = if i % 2 == 0 { vec![Box::new(probe), Box::new(plain)] } else { vec![Box::new(plain), Box::new(probe)] };
        let mut g = Game::new(&COLORS[..2], &rules);
        let winner = g.play(&mut players).expect("legal play");
        winner == Some(a)
    });
    let wins = results.iter().filter(|&&w| w).count();
    let (decisions, reused, visits) = *out.lock().unwrap();
    println!(
        "reuser won {wins}/{games} ({:.1}%); {:.0}% of its {decisions} decisions started from a kept node, \
         with {:.0} visits on average ({:.0}% of a search's {sims})",
        100.0 * wins as f64 / games as f64,
        100.0 * reused as f64 / decisions.max(1) as f64,
        visits as f64 / reused.max(1) as f64,
        100.0 * visits as f64 / reused.max(1) as f64 / sims as f64
    );
}
