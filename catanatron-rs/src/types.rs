use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Resource {
    Wood,
    Brick,
    Sheep,
    Wheat,
    Ore,
}

pub const RESOURCES: [Resource; 5] = [
    Resource::Wood,
    Resource::Brick,
    Resource::Sheep,
    Resource::Wheat,
    Resource::Ore,
];

impl Resource {
    #[inline]
    pub fn idx(self) -> usize {
        self as usize
    }
    pub fn name(self) -> &'static str {
        ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"][self.idx()]
    }
}

/// Histogram of resource cards: [WOOD, BRICK, SHEEP, WHEAT, ORE].
pub type Freqdeck = [u32; 5];

pub const ROAD_COST: Freqdeck = [1, 1, 0, 0, 0];
pub const SETTLEMENT_COST: Freqdeck = [1, 1, 1, 1, 0];
pub const CITY_COST: Freqdeck = [0, 0, 0, 2, 3];
pub const DEV_CARD_COST: Freqdeck = [0, 0, 1, 1, 1];
pub const STARTING_BANK: Freqdeck = [19, 19, 19, 19, 19];

/// True if `a >= b` element-wise.
#[inline]
pub fn fd_contains(a: &Freqdeck, b: &Freqdeck) -> bool {
    a.iter().zip(b).all(|(x, y)| x >= y)
}
#[inline]
pub fn fd_add(a: &mut Freqdeck, b: &Freqdeck) {
    for i in 0..5 {
        a[i] += b[i];
    }
}
#[inline]
pub fn fd_sub(a: &mut Freqdeck, b: &Freqdeck) {
    for i in 0..5 {
        a[i] -= b[i];
    }
}
pub fn fd_single(r: Resource, n: u32) -> Freqdeck {
    let mut d = [0; 5];
    d[r.idx()] = n;
    d
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DevCard {
    Knight,
    YearOfPlenty,
    Monopoly,
    RoadBuilding,
    VictoryPoint,
}

pub const DEV_CARDS: [DevCard; 5] = [
    DevCard::Knight,
    DevCard::YearOfPlenty,
    DevCard::Monopoly,
    DevCard::RoadBuilding,
    DevCard::VictoryPoint,
];

impl DevCard {
    #[inline]
    pub fn idx(self) -> usize {
        self as usize
    }
    pub fn name(self) -> &'static str {
        [
            "KNIGHT",
            "YEAR_OF_PLENTY",
            "MONOPOLY",
            "ROAD_BUILDING",
            "VICTORY_POINT",
        ][self.idx()]
    }
}

pub fn starting_dev_deck() -> Vec<DevCard> {
    let mut deck = Vec::with_capacity(25);
    deck.extend(std::iter::repeat_n(DevCard::Knight, 14));
    deck.extend(std::iter::repeat_n(DevCard::YearOfPlenty, 2));
    deck.extend(std::iter::repeat_n(DevCard::RoadBuilding, 2));
    deck.extend(std::iter::repeat_n(DevCard::Monopoly, 2));
    deck.extend(std::iter::repeat_n(DevCard::VictoryPoint, 5));
    deck
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Color {
    Red,
    Blue,
    Orange,
    White,
}

pub const COLORS: [Color; 4] = [Color::Red, Color::Blue, Color::Orange, Color::White];

impl Color {
    #[inline]
    pub fn idx(self) -> usize {
        self as usize
    }
    pub fn name(self) -> &'static str {
        ["RED", "BLUE", "ORANGE", "WHITE"][self.idx()]
    }
}

impl Default for Color {
    fn default() -> Self {
        Color::Red
    }
}

impl Default for DevCard {
    fn default() -> Self {
        DevCard::Knight
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Cube coordinate of a tile.
pub type Coordinate = (i8, i8, i8);
pub type NodeId = u8;
/// Edge between two nodes, always stored with the smaller id first.
pub type EdgeId = (NodeId, NodeId);

#[inline]
pub fn edge(a: NodeId, b: NodeId) -> EdgeId {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActionPrompt {
    BuildInitialSettlement,
    BuildInitialRoad,
    PlayTurn,
    Discard,
    MoveRobber,
    DecideTrade,
    DecideAcceptees,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActionKind {
    Roll,
    MoveRobber {
        coordinate: Coordinate,
        victim: Option<Color>,
    },
    DiscardResource(Resource),
    BuildRoad(EdgeId),
    BuildSettlement(NodeId),
    BuildCity(NodeId),
    BuyDevelopmentCard,
    PlayKnightCard,
    /// Second resource is None when the bank only allows taking one card.
    PlayYearOfPlenty(Resource, Option<Resource>),
    PlayMonopoly(Resource),
    PlayRoadBuilding,
    /// Give `rate` of `give` to the bank for one `get`.
    MaritimeTrade {
        give: Resource,
        rate: u8,
        get: Resource,
    },
    OfferTrade {
        offering: Freqdeck,
        asking: Freqdeck,
    },
    AcceptTrade,
    RejectTrade,
    ConfirmTrade {
        partner: Color,
    },
    CancelTrade,
    EndTurn,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Action {
    pub color: Color,
    pub kind: ActionKind,
}

impl Action {
    pub fn new(color: Color, kind: ActionKind) -> Self {
        Action { color, kind }
    }
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {:?}", self.color, self.kind)
    }
}

/// Outcome of the non-deterministic part of an action, if any.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionResult {
    None,
    Dice(u8, u8),
    DevCard(DevCard),
    Stolen(Option<Resource>),
    Discarded(Resource),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActionRecord {
    pub action: Action,
    pub result: ActionResult,
}

/// Fixed-size bitset over node or edge ids (ids are u8, so 256 bits).
pub type Bits = [u64; 4];

#[inline]
pub fn bit_get(b: &Bits, i: usize) -> bool {
    b[i >> 6] >> (i & 63) & 1 == 1
}
#[inline]
pub fn bit_set(b: &mut Bits, i: usize) {
    b[i >> 6] |= 1 << (i & 63);
}
#[inline]
pub fn bit_clear(b: &mut Bits, i: usize) {
    b[i >> 6] &= !(1 << (i & 63));
}
#[inline]
pub fn bits_any(b: &Bits) -> bool {
    b.iter().any(|&w| w != 0)
}
#[inline]
pub fn bits_and(a: &Bits, b: &Bits) -> Bits {
    [a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]]
}
/// Set bit indices, ascending.
#[inline]
pub fn bits_iter(b: Bits) -> impl Iterator<Item = usize> {
    (0..4).flat_map(move |w| {
        let mut word = b[w];
        std::iter::from_fn(move || {
            if word == 0 {
                return None;
            }
            let t = word.trailing_zeros() as usize;
            word &= word - 1;
            Some(w * 64 + t)
        })
    })
}
