//! Complete game state. Players are referred to by seat index (position in
//! `colors`), which is shuffled at game start.

use std::sync::Arc;

use crate::board::Board;
use crate::fixed::FixedVec;
use crate::map::CatanMap;
use crate::rng::Rng;
use crate::types::*;

pub const MAX_PLAYERS: usize = 4;
pub const NUM_DEV_CARDS: usize = 25;

/// Seat-indexed list (at most `MAX_PLAYERS` entries).
pub type PerSeat<T> = FixedVec<T, MAX_PLAYERS>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlayerState {
    pub color: Color,
    /// Points visible to opponents (excludes hidden VP cards).
    pub victory_points: u32,
    pub actual_victory_points: u32,
    pub roads_available: u32,
    pub settlements_available: u32,
    pub cities_available: u32,
    pub has_road: bool,
    pub has_army: bool,
    pub has_rolled: bool,
    pub has_played_development_card_in_turn: bool,
    pub longest_road_length: u32,
    pub resources: Freqdeck,
    /// Indexed by `DevCard::idx`.
    pub dev_cards: [u32; 5],
    pub played_dev_cards: [u32; 5],
    /// Whether a card of that type was held at the start of the turn
    /// (cards bought this turn can't be played).
    pub dev_owned_at_start: [bool; 5],
    /// Development cards bought this turn (purchases are public).
    pub dev_bought_this_turn: u32,
    /// Own turns ended holding a card bought before that turn without
    /// playing any, since the last card played. Public; a long streak
    /// suggests victory point cards.
    pub dev_idle_turns: u32,
    /// `State::num_turns` at the last purchase.
    pub last_dev_buy_turn: Option<u32>,
    pub settlements: FixedVec<NodeId, 5>,
    pub cities: FixedVec<NodeId, 4>,
    pub roads: FixedVec<EdgeId, 15>,
}

impl PlayerState {
    pub fn new(color: Color) -> Self {
        PlayerState {
            color,
            victory_points: 0,
            actual_victory_points: 0,
            roads_available: 15,
            settlements_available: 5,
            cities_available: 4,
            has_road: false,
            has_army: false,
            has_rolled: false,
            has_played_development_card_in_turn: false,
            longest_road_length: 0,
            resources: [0; 5],
            dev_cards: [0; 5],
            played_dev_cards: [0; 5],
            dev_owned_at_start: [false; 5],
            dev_bought_this_turn: 0,
            dev_idle_turns: 0,
            last_dev_buy_turn: None,
            settlements: FixedVec::new(),
            cities: FixedVec::new(),
            roads: FixedVec::new(),
        }
    }

    pub fn num_resources(&self) -> u32 {
        self.resources.iter().sum()
    }

    pub fn num_dev_cards(&self) -> u32 {
        self.dev_cards.iter().sum()
    }

    pub fn can_play_dev(&self, card: DevCard) -> bool {
        !self.has_played_development_card_in_turn
            && self.dev_cards[card.idx()] >= 1
            && self.dev_owned_at_start[card.idx()]
    }
}

/// Complete game state. Holds no heap data apart from the shared map
/// (an `Arc`), so cloning is a memory copy plus a reference-count bump.
/// The action history lives on `Game`, not here.
/// Players the robber may not be moved onto or steal from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FriendlyRobber {
    #[default]
    Off,
    /// Players with under 3 points, hidden VP cards included (catanatron's rule).
    ActualPoints,
    /// Players with under 3 visible points, VP cards not counted (OpenHex's
    /// rule): the robber can't block an opponent's buildings.
    PublicPoints,
    /// Colonist's rule (as the colonizing-pig engine implements it): under 3
    /// visible points, the mover included, nobody's buildings can be blocked,
    /// and a player under 3 visible points is never stolen from (when every
    /// tile is shielded and the robber must go somewhere anyway).
    Colonist,
}

/// How dice are rolled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DiceMode {
    /// Two independent fair dice.
    #[default]
    Fair,
    /// Each of the 36 (die 1, die 2) combinations once per 36 rolls, in
    /// random order (the deck refills when empty).
    Deck,
    /// OpenHex's balanced dice (its `dice.py`, after colonist.io's design):
    /// a 36-card deck reshuffled once 13 or fewer cards remain, each total
    /// weighted by its share of the remaining cards, times max(0, 1 - 0.34 *
    /// its appearances in the last 5 rolls), with the 7 weighted per roller
    /// by a streak and imbalance multiplier in [0, 2]; a card of the chosen
    /// total is then taken at random.
    OpenHex,
}

const OPENHEX_RESHUFFLE_AT: u8 = 13;
const OPENHEX_RECENCY_WINDOW: usize = 5;
const OPENHEX_RECENCY_PENALTY: f32 = 0.34;
const OPENHEX_SEVEN_STREAK_PENALTY: f32 = 0.4;

/// Balanced dice state (`DiceMode::Deck` or `DiceMode::OpenHex`). Drawing a
/// random remaining card (rather than keeping a shuffled order) means
/// reseeding `State::rng` re-randomizes future rolls, as search needs.
#[derive(Clone, Copy, Debug)]
pub struct DiceDeck {
    left: [u8; 36],
    len: u8,
    openhex: bool,
    /// OpenHex: the last (up to 5) roll totals, oldest first.
    recent: [u8; OPENHEX_RECENCY_WINDOW],
    recent_len: u8,
    /// OpenHex: 7s rolled per seat.
    sevens: [u8; MAX_PLAYERS],
    /// OpenHex: seat on the current run of consecutive 7s, and its length.
    streak_seat: u8,
    streak: u8,
}

impl DiceDeck {
    /// A full deck (`Deck` mode).
    pub fn full() -> DiceDeck {
        let mut left = [0; 36];
        for (i, c) in left.iter_mut().enumerate() {
            *c = i as u8;
        }
        DiceDeck { left, len: 36, openhex: false, recent: [0; OPENHEX_RECENCY_WINDOW], recent_len: 0, sevens: [0; MAX_PLAYERS], streak_seat: 0, streak: 0 }
    }

    /// A full deck for `mode` (`None` for fair dice).
    pub fn new(mode: DiceMode) -> Option<DiceDeck> {
        match mode {
            DiceMode::Fair => None,
            DiceMode::Deck => Some(DiceDeck::full()),
            DiceMode::OpenHex => Some(DiceDeck { openhex: true, ..DiceDeck::full() }),
        }
    }

    pub fn is_openhex(&self) -> bool {
        self.openhex
    }

    /// Same future rolls: the same combinations left (in any order) and the
    /// same OpenHex history.
    pub fn same_odds(&self, other: &DiceDeck) -> bool {
        let sorted = |d: &DiceDeck| {
            let mut left = d.left;
            left[..d.len as usize].sort_unstable();
            left
        };
        self.len == other.len
            && sorted(self)[..self.len as usize] == sorted(other)[..other.len as usize]
            && self.openhex == other.openhex
            && self.recent[..self.recent_len as usize] == other.recent[..other.recent_len as usize]
            && self.sevens == other.sevens
            && self.streak_seat == other.streak_seat
            && self.streak == other.streak
    }

    /// Combinations left before the next refill.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Combinations not yet rolled since the last refill, as (die 1, die 2).
    pub fn remaining(&self) -> impl Iterator<Item = (u8, u8)> + '_ {
        self.left[..self.len as usize].iter().map(|&c| (c / 6 + 1, c % 6 + 1))
    }

    /// Whether the deck is refilled before the next draw.
    fn refills_next(&self) -> bool {
        if self.openhex { self.len <= OPENHEX_RESHUFFLE_AT } else { self.len == 0 }
    }

    /// The deck the next roll is drawn from (after any refill), keeping the
    /// recent-roll and 7 history.
    pub fn next_deck(&self) -> DiceDeck {
        if self.refills_next() {
            DiceDeck { left: DiceDeck::full().left, len: 36, ..*self }
        } else {
            *self
        }
    }

    /// OpenHex: recent appearances of `total` among the last 5 rolls.
    pub fn recent_count(&self, total: u8) -> u32 {
        self.recent[..self.recent_len as usize].iter().filter(|&&t| t == total).count() as u32
    }

    /// OpenHex: 7s rolled by `seat`.
    pub fn sevens(&self, seat: usize) -> u32 {
        self.sevens[seat] as u32
    }

    /// OpenHex: the seat on the current 7 streak and its length (length 0 =
    /// no streak).
    pub fn seven_streak(&self) -> (usize, u32) {
        (self.streak_seat as usize, self.streak as u32)
    }

    /// Probability of each total (index = total, 2..=12) for the next roll
    /// by `seat` among `num_players`.
    pub fn probabilities(&self, seat: usize, num_players: usize) -> [f32; 13] {
        let (d, base) = self.base_weights();
        d.seat_probabilities(&base, seat, num_players)
    }

    /// `probabilities` for two seats at once: the deck's weights are
    /// counted once and each seat's 7 weighting and normalisation applied
    /// to a copy, the same operations as two separate calls.
    pub fn probabilities_pair(&self, seats: [usize; 2], num_players: usize) -> [[f32; 13]; 2] {
        let (d, base) = self.base_weights();
        seats.map(|seat| d.seat_probabilities(&base, seat, num_players))
    }

    /// The deck the next roll is drawn from and each total's weight before
    /// the roller's 7 adjustment and normalisation.
    fn base_weights(&self) -> (DiceDeck, [f32; 13]) {
        let d = self.next_deck();
        let mut count = [0u32; 13];
        for (a, b) in d.remaining() {
            count[(a + b) as usize] += 1;
        }
        let mut w = [0f32; 13];
        for t in 2..=12 {
            w[t] = count[t] as f32 / d.len as f32;
            if d.openhex && count[t] > 0 {
                w[t] *= (1.0 - OPENHEX_RECENCY_PENALTY * d.recent_count(t as u8) as f32).max(0.0);
            }
        }
        (d, w)
    }

    /// `base` (from `base_weights` of this deck) adjusted for `seat` rolling
    /// and normalised.
    fn seat_probabilities(&self, base: &[f32; 13], seat: usize, num_players: usize) -> [f32; 13] {
        let mut w = *base;
        if self.openhex && w[7] > 0.0 {
            w[7] *= self.seven_multiplier(seat, num_players);
        }
        let total: f32 = w.iter().sum();
        assert!(total > 0.0, "some total can always be rolled");
        for x in w.iter_mut() {
            *x /= total;
        }
        w
    }

    fn seven_multiplier(&self, seat: usize, num_players: usize) -> f32 {
        let streak_adj = if self.streak > 0 {
            let direction = if self.streak_seat as usize == seat { -1.0 } else { 1.0 };
            OPENHEX_SEVEN_STREAK_PENALTY * self.streak as f32 * direction
        } else {
            0.0
        };
        let total: u32 = self.sevens.iter().map(|&x| x as u32).sum();
        let imbalance_adj = if total >= num_players as u32 {
            let pct = self.sevens[seat] as f32 / total as f32;
            let ideal = 1.0 / num_players as f32;
            1.0 + (ideal - pct) / ideal
        } else {
            1.0
        };
        (imbalance_adj + streak_adj).clamp(0.0, 2.0)
    }

    /// Rolls for `seat` among `num_players`.
    pub fn draw(&mut self, rng: &mut Rng, seat: usize, num_players: usize) -> (u8, u8) {
        let p = self.probabilities(seat, num_players);
        *self = self.next_deck();
        // Pick a total by weight, then one of its remaining cards uniformly.
        let x = (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        let mut acc = 0.0;
        let mut total = 12;
        for (t, &pt) in p.iter().enumerate().skip(2) {
            acc += pt;
            if x < acc && pt > 0.0 {
                total = t as u8;
                break;
            }
        }
        while p[total as usize] == 0.0 {
            total -= 1; // rounding at the very top of the range
        }
        // One of the remaining cards of that total, uniformly.
        let matches = |c: &u8| c / 6 + c % 6 + 2 == total;
        let count = self.left[..self.len as usize].iter().filter(|c| matches(c)).count();
        let k = rng.below(count);
        let i = self.left[..self.len as usize].iter().position({
            let mut seen = 0;
            move |c| {
                let hit = matches(c) && seen == k;
                seen += matches(c) as usize;
                hit
            }
        }).expect("a card of the chosen total remains");
        let roll = self.take(i);
        self.record(total, seat);
        roll
    }

    /// Takes out a given roll by `seat` (replaying a recorded game, or a
    /// forced outcome).
    pub fn remove(&mut self, a: u8, b: u8, seat: usize) {
        *self = self.next_deck();
        let code = (a - 1) * 6 + (b - 1);
        if let Some(i) = self.left[..self.len as usize].iter().position(|&c| c == code) {
            self.take(i);
        }
        self.record(a + b, seat);
    }

    fn record(&mut self, total: u8, seat: usize) {
        if !self.openhex {
            return;
        }
        if (self.recent_len as usize) < OPENHEX_RECENCY_WINDOW {
            self.recent[self.recent_len as usize] = total;
            self.recent_len += 1;
        } else {
            self.recent.copy_within(1.., 0);
            self.recent[OPENHEX_RECENCY_WINDOW - 1] = total;
        }
        if total == 7 {
            self.sevens[seat] += 1;
            if self.streak > 0 && self.streak_seat as usize == seat {
                self.streak += 1;
            } else {
                self.streak_seat = seat as u8;
                self.streak = 1;
            }
        } else {
            self.streak = 0;
        }
    }

    fn take(&mut self, i: usize) -> (u8, u8) {
        let c = self.left[i];
        self.len -= 1;
        self.left[i] = self.left[self.len as usize];
        (c / 6 + 1, c % 6 + 1)
    }
}

#[derive(Clone, Debug)]
pub struct State {
    /// Seating order.
    pub colors: PerSeat<Color>,
    /// Seat-indexed.
    pub players: PerSeat<PlayerState>,
    pub board: Board,
    pub discard_limit: u32,
    pub friendly_robber: FriendlyRobber,
    /// Set with balanced dice (`DiceMode::Deck` / `OpenHex`).
    pub dice_deck: Option<DiceDeck>,
    /// Only a knight may be played before the roll (OpenHex's live server);
    /// otherwise any development card may.
    pub knight_only_before_roll: bool,
    pub bank: Freqdeck,
    /// Already shuffled; drawn from the end.
    pub dev_deck: FixedVec<DevCard, NUM_DEV_CARDS>,
    pub num_turns: u32,
    /// Actions applied so far.
    pub num_actions: u32,
    /// Seat that must decide now (may differ from the turn owner, e.g. discards).
    pub current_player_index: usize,
    /// Seat whose turn it is.
    pub current_turn_index: usize,
    pub current_prompt: ActionPrompt,
    pub is_initial_build_phase: bool,
    pub is_discarding: bool,
    pub discard_counts: PerSeat<u32>,
    pub is_moving_knight: bool,
    pub is_road_building: bool,
    pub free_roads_available: u32,
    pub is_resolving_trade: bool,
    /// (offering, asking, offering seat) of the trade being resolved.
    pub current_trade: Option<(Freqdeck, Freqdeck, usize)>,
    pub acceptees: PerSeat<bool>,
    pub rng: Rng,
    /// `Color::idx` => seat, for O(1) lookups.
    seat_of: [u8; 4],
}

impl State {
    /// Seats `colors` in random order and sets up a fresh game on `map`.
    pub fn new(
        colors: &[Color],
        map: Arc<CatanMap>,
        discard_limit: u32,
        friendly_robber: FriendlyRobber,
        dice: DiceMode,
        shuffle_seats: bool,
        mut rng: Rng,
    ) -> State {
        assert!(colors.len() <= MAX_PLAYERS, "at most {MAX_PLAYERS} players");
        let mut colors: PerSeat<Color> = colors.iter().copied().collect();
        if shuffle_seats {
            rng.shuffle(&mut colors);
        }
        let mut dev_deck: FixedVec<DevCard, NUM_DEV_CARDS> = starting_dev_deck().into_iter().collect();
        rng.shuffle(&mut dev_deck);
        let n = colors.len();
        let mut seat_of = [u8::MAX; 4];
        for (i, c) in colors.iter().enumerate() {
            seat_of[c.idx()] = i as u8;
        }
        State {
            players: colors.iter().map(|&c| PlayerState::new(c)).collect(),
            colors,
            board: Board::new(map),
            discard_limit,
            friendly_robber,
            dice_deck: DiceDeck::new(dice),
            knight_only_before_roll: false,
            bank: STARTING_BANK,
            dev_deck,
            num_turns: 0,
            num_actions: 0,
            current_player_index: 0,
            current_turn_index: 0,
            current_prompt: ActionPrompt::BuildInitialSettlement,
            is_initial_build_phase: true,
            is_discarding: false,
            discard_counts: FixedVec::filled(0, n),
            is_moving_knight: false,
            is_road_building: false,
            free_roads_available: 0,
            is_resolving_trade: false,
            current_trade: None,
            acceptees: FixedVec::filled(false, n),
            rng,
            seat_of,
        }
    }

    #[inline]
    pub fn current_color(&self) -> Color {
        self.colors[self.current_player_index]
    }

    #[inline]
    pub fn seat(&self, color: Color) -> usize {
        let s = self.seat_of[color.idx()];
        assert!(s != u8::MAX, "color is not seated in this game");
        s as usize
    }

    #[inline]
    pub fn player(&self, color: Color) -> &PlayerState {
        &self.players[self.seat(color)]
    }

    #[inline]
    pub fn player_mut(&mut self, color: Color) -> &mut PlayerState {
        let s = self.seat(color);
        &mut self.players[s]
    }

    pub fn largest_army(&self) -> Option<(Color, u32)> {
        self.players
            .iter()
            .find(|p| p.has_army)
            .map(|p| (p.color, p.played_dev_cards[DevCard::Knight.idx()]))
    }

    pub fn longest_road_color(&self) -> Option<Color> {
        self.players.iter().find(|p| p.has_road).map(|p| p.color)
    }
}
