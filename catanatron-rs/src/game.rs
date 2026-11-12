//! Thin wrapper over `State`: seeding, validation, move generation, and the
//! play loop.

use std::sync::Arc;

use crate::actions::{generate_playable_actions, generate_playable_actions_into, is_valid_trade};
use crate::apply::apply_action;
use crate::map::{build_map, CatanMap, MapType, NumberPlacement};
use crate::players::Player;
use crate::rng::Rng;
use crate::state::DiceMode;
use crate::state::{FriendlyRobber, State};
use crate::types::*;

/// Games are cut off after this many turns and count as a loss for everyone
/// (`mcts::terminal_values`). Searching bots finish 1v1 games well before it
/// (99.9% within 200 turns, the longest seen 247): a game this long is a
/// stall, and cutting it early makes stalling cheap to punish in training.
pub const TURNS_LIMIT: u32 = 300;

#[derive(Clone, Debug)]
pub struct GameConfig {
    pub seed: Option<u64>,
    pub discard_limit: u32,
    pub friendly_robber: FriendlyRobber,
    /// How dice are rolled (see `state::DiceMode`).
    pub dice: DiceMode,
    pub vps_to_win: u32,
    pub map_type: MapType,
    pub number_placement: NumberPlacement,
    /// Overrides `map_type` when set.
    pub catan_map: Option<Arc<CatanMap>>,
    /// Keep a log of every action in `Game::history`.
    pub record_history: bool,
    /// Seat players in the order given instead of shuffling (to mirror a
    /// game played elsewhere).
    pub fixed_seats: bool,
    /// Turns after which the game ends with no winner (`None`: never, for a
    /// mirror of a game whose server decides when it ends).
    pub turns_limit: Option<u32>,
    /// Shares of games on selected real boards (see `map::BoardMix`).
    pub board_mix: crate::map::BoardMix,
    /// Only boards whose scarcest resource is this one (see
    /// `map::scarcest_resource`): a game redraws its random board until so.
    pub board_group: Option<Resource>,
    /// Each game on one of these boards, drawn uniformly (a copy per game,
    /// as with `catan_map`). Overridden by `catan_map`.
    pub board_pool: Option<Arc<[CatanMap]>>,
}

impl Default for GameConfig {
    fn default() -> Self {
        GameConfig {
            seed: None,
            discard_limit: 7,
            friendly_robber: FriendlyRobber::Off,
            dice: DiceMode::Fair,
            vps_to_win: 10,
            map_type: MapType::Base,
            number_placement: NumberPlacement::OfficialSpiral,
            catan_map: None,
            record_history: true,
            fixed_seats: false,
            turns_limit: Some(TURNS_LIMIT),
            board_mix: Default::default(),
            board_group: None,
            board_pool: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Game {
    pub seed: u64,
    pub vps_to_win: u32,
    pub turns_limit: Option<u32>,
    pub state: State,
    pub playable_actions: Vec<Action>,
    /// Every action applied, with its random outcome, if recording is on.
    pub history: Vec<ActionRecord>,
    pub record_history: bool,
}

impl Game {
    pub fn new(colors: &[Color], config: &GameConfig) -> Game {
        assert!(
            (2..=4).contains(&colors.len()),
            "a game needs 2 to 4 players"
        );
        let seed = config.seed.unwrap_or_else(|| Rng::from_entropy().next_u64());
        let mut rng = Rng::new(seed);
        let map = match &config.catan_map {
            // A copy per game, not the shared Arc: search clones the state
            // (and so the map's Arc) constantly, and with one Arc for every
            // game, the threads all hit one reference count. On 64-thread
            // actors that made fixed-board self-play ~1.7x slower per example.
            Some(m) => Arc::new(CatanMap::clone(m)),
            None if config.board_pool.is_some() => {
                let pool = config.board_pool.as_ref().unwrap();
                assert!(!pool.is_empty(), "an empty board pool");
                assert!(config.board_mix.is_empty() && config.board_group.is_none(), "a board pool with a board mix or group");
                Arc::new(pool[(rng.next_u64() % pool.len() as u64) as usize].clone())
            }
            None => {
                // Drawn only with a mix set, so standard games are unchanged.
                let style = match config.board_mix.is_empty() {
                    true => None,
                    false => config.board_mix.pick(&mut rng),
                };
                let mut map = build_map(config.map_type, config.number_placement, &mut rng);
                if let Some(style) = style {
                    assert!(config.map_type != MapType::Tournament, "a board mix needs randomized boards");
                    while !style.matches(&map) {
                        map = build_map(config.map_type, config.number_placement, &mut rng);
                    }
                }
                if let Some(group) = config.board_group {
                    assert!(config.board_mix.is_empty(), "a board group and a board mix together");
                    assert!(config.map_type != MapType::Tournament, "a board group needs randomized boards");
                    while crate::map::scarcest_resource(&map) != group {
                        map = build_map(config.map_type, config.number_placement, &mut rng);
                    }
                }
                map
            }
        };
        let state = State::new(colors, map, config.discard_limit, config.friendly_robber, config.dice, !config.fixed_seats, rng);
        let playable_actions = generate_playable_actions(&state);
        Game {
            seed,
            vps_to_win: config.vps_to_win,
            turns_limit: config.turns_limit,
            state,
            playable_actions,
            history: Vec::new(),
            record_history: config.record_history,
        }
    }

    /// Cheap copy for lookahead: same state and playable actions, no
    /// history, and recording turned off.
    pub fn copy_without_history(&self) -> Game {
        Game {
            seed: self.seed,
            vps_to_win: self.vps_to_win,
            turns_limit: self.turns_limit,
            state: self.state.clone(),
            playable_actions: self.playable_actions.clone(),
            history: Vec::new(),
            record_history: false,
        }
    }

    /// `copy_without_history` for a copy that is advanced right away: the
    /// playable actions are left empty (with room for a typical list)
    /// instead of copied, since `execute_unchecked` regenerates them.
    pub(crate) fn copy_for_step(&self) -> Game {
        Game {
            seed: self.seed,
            vps_to_win: self.vps_to_win,
            turns_limit: self.turns_limit,
            state: self.state.clone(),
            playable_actions: Vec::with_capacity(32),
            history: Vec::new(),
            record_history: false,
        }
    }

    /// True if `action` may be taken now: it is among the playable actions,
    /// or it is a well-formed trade offer by the player whose turn it is
    /// after rolling.
    pub fn is_valid_action(&self, action: &Action) -> bool {
        if let ActionKind::OfferTrade { offering, asking } = action.kind {
            let s = &self.state;
            return s.current_color() == action.color
                && s.current_prompt == ActionPrompt::PlayTurn
                && !s.is_road_building
                && s.player(action.color).has_rolled
                && fd_contains(&s.player(action.color).resources, &offering)
                && is_valid_trade(&offering, &asking);
        }
        self.playable_actions.contains(action)
    }

    /// Validates and applies `action`, then regenerates playable actions.
    pub fn execute(&mut self, action: Action) -> Result<ActionRecord, String> {
        if !self.is_valid_action(&action) {
            return Err(format!("{action} not playable right now"));
        }
        self.execute_unchecked(action, None)
    }

    /// Applies without checking playability; `replay` forces the random
    /// outcome (dice, stolen card, drawn dev card).
    pub fn execute_unchecked(&mut self, action: Action, replay: Option<ActionResult>) -> Result<ActionRecord, String> {
        let record = apply_action(&mut self.state, action, replay)?;
        if self.record_history {
            self.history.push(record);
        }
        generate_playable_actions_into(&self.state, &mut self.playable_actions);
        Ok(record)
    }

    /// Winning color, if anyone has reached `vps_to_win`.
    pub fn winning_color(&self) -> Option<Color> {
        self.state
            .players
            .iter()
            .filter(|p| p.actual_victory_points >= self.vps_to_win)
            .map(|p| p.color)
            .next_back()
    }

    pub fn is_over(&self) -> bool {
        self.winning_color().is_some() || self.turns_limit.is_some_and(|t| self.state.num_turns >= t)
    }

    /// Asks the current player for an action and applies it.
    pub fn play_tick(&mut self, players: &mut [Box<dyn Player>]) -> Result<ActionRecord, String> {
        let color = self.state.current_color();
        let player = players
            .iter_mut()
            .find(|p| p.color() == color)
            .ok_or_else(|| format!("no player for {color}"))?;
        let action = player.decide(self, &self.playable_actions);
        self.execute(action)
    }

    /// Plays until someone wins or `TURNS_LIMIT` is hit.
    pub fn play(&mut self, players: &mut [Box<dyn Player>]) -> Result<Option<Color>, String> {
        while !self.is_over() {
            self.play_tick(players)?;
        }
        Ok(self.winning_color())
    }
}
