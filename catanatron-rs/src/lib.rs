//! Rust port of the catanatron core engine: map generation, board rules,
//! move generation, action application, and a few simple bots.

pub mod actions;
pub mod apply;
pub mod board;
pub mod encoding;
pub mod fixed;
pub mod game;
pub mod map;
pub mod mcts;
pub mod nn;
pub mod openhex;
pub mod parallel;
pub mod players;
pub mod rng;
pub mod search;
pub mod selfplay;
pub mod state;
pub mod types;
pub mod vecenv;

pub use game::{Game, GameConfig, TURNS_LIMIT};
pub use state::FriendlyRobber;
pub use players::Player;
pub use types::*;
