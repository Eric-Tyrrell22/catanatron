//! Fixed-size encodings for learning: a discrete action space with legal
//! masks, and an observation vector.
//!
//! Everything is from the point of view of the player deciding: other
//! players are referred to by *relative seat* (1 = next to act after me,
//! 2 = the one after, ...), so a policy learned in one seat transfers to
//! every seat and color.
//!
//! Observations contain only what that player may know: opponents'
//! resource and development cards appear as totals, never by type.

use crate::board::{BuildingType, MAX_NODES};
use crate::map::CatanMap;
use crate::state::{DiceDeck, State, MAX_PLAYERS};
use crate::types::*;

/// Shape of one slot in the action space. Victims/partners are relative
/// seats (1..=3); the acting color is filled in when decoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionSlot {
    Roll,
    EndTurn,
    Discard(Resource),
    BuildRoad(EdgeId),
    BuildSettlement(NodeId),
    BuildCity(NodeId),
    BuyDevelopmentCard,
    PlayKnight,
    PlayYearOfPlenty(Resource, Option<Resource>),
    PlayRoadBuilding,
    PlayMonopoly(Resource),
    /// Land tile index (position in `CatanMap::land_tiles`), relative victim (0 = none).
    MoveRobber { tile: u8, victim: u8 },
    MaritimeTrade { give: Resource, rate: u8, get: Resource },
    AcceptTrade,
    RejectTrade,
    CancelTrade,
    /// Relative seat of the partner.
    ConfirmTrade(u8),
}

/// The discrete action space for one map layout.
///
/// Layout (in order): roll, end turn, 5 discards, a road per land edge, a
/// settlement and a city per land node, buy dev card, knight, 20 year of
/// plenty (15 pairs + 5 singles), road building, 5 monopoly, robber moves
/// (land tile x {no victim, 3 relative seats}), 60 maritime trades
/// (give x get x rate 4/3/2), accept, reject, cancel, 3 confirm.
///
/// `OfferTrade` is not representable: learned agents don't propose trades.
#[derive(Clone, Debug)]
pub struct ActionSpace {
    slots: Vec<ActionSlot>,
    num_land_nodes: usize,
    /// `CatanMap::edges` index => road slot index (usize::MAX if not land).
    road_slot: Vec<usize>,
    o_discard: usize,
    o_settlement: usize,
    o_city: usize,
    o_buy: usize,
    o_yop: usize,
    o_monopoly: usize,
    o_robber: usize,
    o_maritime: usize,
    o_trade: usize,
}

const ROBBER_VICTIMS: usize = MAX_PLAYERS; // none + 3 relative seats

fn yop_options() -> Vec<(Resource, Option<Resource>)> {
    let mut v = Vec::new();
    for i in 0..5 {
        for j in i..5 {
            v.push((RESOURCES[i], Some(RESOURCES[j])));
        }
    }
    v.extend(RESOURCES.iter().map(|&r| (r, None)));
    v
}

fn maritime_offset(give: Resource, rate: u8, get: Resource) -> usize {
    let rate_i = match rate {
        4 => 0,
        3 => 1,
        2 => 2,
        _ => panic!("invalid maritime rate {rate}"),
    };
    // 20 (give, get) pairs with give != get.
    let get_i = get.idx() - (get.idx() > give.idx()) as usize;
    rate_i * 20 + give.idx() * 4 + get_i
}

impl ActionSpace {
    pub fn new(map: &CatanMap) -> ActionSpace {
        let num_land_nodes = map.land_nodes.iter().filter(|&&x| x).count();
        assert!(
            map.land_nodes[..num_land_nodes].iter().all(|&x| x),
            "action space assumes land node ids are 0..num_land_nodes"
        );
        let mut slots = vec![ActionSlot::Roll, ActionSlot::EndTurn];
        let o_discard = slots.len();
        slots.extend(RESOURCES.iter().map(|&r| ActionSlot::Discard(r)));

        let mut road_slot = vec![usize::MAX; map.edges.len()];
        for (i, &e) in map.edges.iter().enumerate() {
            if map.land_edge[i] {
                road_slot[i] = slots.len();
                slots.push(ActionSlot::BuildRoad(e));
            }
        }
        let o_settlement = slots.len();
        slots.extend((0..num_land_nodes).map(|n| ActionSlot::BuildSettlement(n as NodeId)));
        let o_city = slots.len();
        slots.extend((0..num_land_nodes).map(|n| ActionSlot::BuildCity(n as NodeId)));

        let o_buy = slots.len();
        slots.push(ActionSlot::BuyDevelopmentCard);
        slots.push(ActionSlot::PlayKnight);
        let o_yop = slots.len();
        slots.extend(yop_options().into_iter().map(|(a, b)| ActionSlot::PlayYearOfPlenty(a, b)));
        slots.push(ActionSlot::PlayRoadBuilding);
        let o_monopoly = slots.len();
        slots.extend(RESOURCES.iter().map(|&r| ActionSlot::PlayMonopoly(r)));

        let o_robber = slots.len();
        for t in 0..map.land_tiles.len() {
            for v in 0..ROBBER_VICTIMS {
                slots.push(ActionSlot::MoveRobber { tile: t as u8, victim: v as u8 });
            }
        }

        let o_maritime = slots.len();
        for rate in [4u8, 3, 2] {
            for give in RESOURCES {
                for get in RESOURCES {
                    if give != get {
                        debug_assert_eq!(slots.len() - o_maritime, maritime_offset(give, rate, get));
                        slots.push(ActionSlot::MaritimeTrade { give, rate, get });
                    }
                }
            }
        }

        let o_trade = slots.len();
        slots.extend([ActionSlot::AcceptTrade, ActionSlot::RejectTrade, ActionSlot::CancelTrade]);
        slots.extend((1..MAX_PLAYERS as u8).map(ActionSlot::ConfirmTrade));

        ActionSpace {
            slots,
            num_land_nodes,
            road_slot,
            o_discard,
            o_settlement,
            o_city,
            o_buy,
            o_yop,
            o_monopoly,
            o_robber,
            o_maritime,
            o_trade,
        }
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn slot(&self, i: usize) -> ActionSlot {
        self.slots[i]
    }

    pub fn slots(&self) -> &[ActionSlot] {
        &self.slots
    }

    /// Index of `action` in this space, from the acting player's view.
    /// `None` only for `OfferTrade`.
    pub fn encode(&self, s: &State, action: &Action) -> Option<usize> {
        let rel = |c: Color| relative_seat(s, action.color, c) as usize;
        Some(match action.kind {
            ActionKind::Roll => 0,
            ActionKind::EndTurn => 1,
            ActionKind::DiscardResource(r) => self.o_discard + r.idx(),
            ActionKind::BuildRoad((a, b)) => {
                let ei = s.board.map.edge_idx(a, b)?;
                let slot = self.road_slot[ei];
                (slot != usize::MAX).then_some(slot)?
            }
            ActionKind::BuildSettlement(n) => self.o_settlement + self.land_node(n)?,
            ActionKind::BuildCity(n) => self.o_city + self.land_node(n)?,
            ActionKind::BuyDevelopmentCard => self.o_buy,
            ActionKind::PlayKnightCard => self.o_buy + 1,
            ActionKind::PlayYearOfPlenty(a, b) => {
                let (a, b) = match b {
                    Some(b) if b < a => (b, Some(a)),
                    _ => (a, b),
                };
                self.o_yop + yop_options().iter().position(|&o| o == (a, b))?
            }
            ActionKind::PlayRoadBuilding => self.o_monopoly - 1,
            ActionKind::PlayMonopoly(r) => self.o_monopoly + r.idx(),
            ActionKind::MoveRobber { coordinate, victim } => {
                let map = &s.board.map;
                let t = map.land_tiles.iter().position(|&i| map.tiles[i].coordinate == coordinate)?;
                let v = victim.map_or(0, rel);
                self.o_robber + t * ROBBER_VICTIMS + v
            }
            ActionKind::MaritimeTrade { give, rate, get } => {
                if give == get || !(2..=4).contains(&rate) {
                    return None;
                }
                self.o_maritime + maritime_offset(give, rate, get)
            }
            ActionKind::AcceptTrade => self.o_trade,
            ActionKind::RejectTrade => self.o_trade + 1,
            ActionKind::CancelTrade => self.o_trade + 2,
            ActionKind::ConfirmTrade { partner } => self.o_trade + 2 + rel(partner),
            ActionKind::OfferTrade { .. } => return None,
        })
    }

    /// The action for slot `i`, taken by the player currently deciding.
    /// Not necessarily legal; check the mask.
    pub fn decode(&self, s: &State, i: usize) -> Action {
        let color = s.current_color();
        let abs = |r: u8| absolute_color(s, color, r);
        let kind = match self.slots[i] {
            ActionSlot::Roll => ActionKind::Roll,
            ActionSlot::EndTurn => ActionKind::EndTurn,
            ActionSlot::Discard(r) => ActionKind::DiscardResource(r),
            ActionSlot::BuildRoad(e) => ActionKind::BuildRoad(e),
            ActionSlot::BuildSettlement(n) => ActionKind::BuildSettlement(n),
            ActionSlot::BuildCity(n) => ActionKind::BuildCity(n),
            ActionSlot::BuyDevelopmentCard => ActionKind::BuyDevelopmentCard,
            ActionSlot::PlayKnight => ActionKind::PlayKnightCard,
            ActionSlot::PlayYearOfPlenty(a, b) => ActionKind::PlayYearOfPlenty(a, b),
            ActionSlot::PlayRoadBuilding => ActionKind::PlayRoadBuilding,
            ActionSlot::PlayMonopoly(r) => ActionKind::PlayMonopoly(r),
            ActionSlot::MoveRobber { tile, victim } => {
                let map = &s.board.map;
                ActionKind::MoveRobber {
                    coordinate: map.tiles[map.land_tiles[tile as usize]].coordinate,
                    victim: if victim == 0 { None } else { abs(victim) },
                }
            }
            ActionSlot::MaritimeTrade { give, rate, get } => ActionKind::MaritimeTrade { give, rate, get },
            ActionSlot::AcceptTrade => ActionKind::AcceptTrade,
            ActionSlot::RejectTrade => ActionKind::RejectTrade,
            ActionSlot::CancelTrade => ActionKind::CancelTrade,
            ActionSlot::ConfirmTrade(r) => ActionKind::ConfirmTrade {
                // Out-of-range seats decode to an illegal (masked-out) action.
                partner: abs(r).unwrap_or(color),
            },
        };
        Action::new(color, kind)
    }

    /// Sets `mask[i] = true` for each playable action's slot, false elsewhere.
    pub fn fill_mask(&self, s: &State, playable: &[Action], mask: &mut [bool]) {
        assert_eq!(mask.len(), self.len());
        mask.fill(false);
        for a in playable {
            if let Some(i) = self.encode(s, a) {
                mask[i] = true;
            }
        }
    }

    pub fn mask(&self, s: &State, playable: &[Action]) -> Vec<bool> {
        let mut m = vec![false; self.len()];
        self.fill_mask(s, playable, &mut m);
        m
    }

    fn land_node(&self, n: NodeId) -> Option<usize> {
        ((n as usize) < self.num_land_nodes).then_some(n as usize)
    }
}

/// Seats after `me` until `other` (0 = `other` is `me`).
pub fn relative_seat(s: &State, me: Color, other: Color) -> u8 {
    let n = s.colors.len();
    ((s.seat(other) + n - s.seat(me)) % n) as u8
}

/// The color `rel` seats after `me`, if that many players are seated.
pub fn absolute_color(s: &State, me: Color, rel: u8) -> Option<Color> {
    let n = s.colors.len();
    ((rel as usize) < n).then(|| s.colors[(s.seat(me) + rel as usize) % n])
}

// ===== Observation =====

const PROMPTS: [ActionPrompt; 7] = [
    ActionPrompt::BuildInitialSettlement,
    ActionPrompt::BuildInitialRoad,
    ActionPrompt::PlayTurn,
    ActionPrompt::Discard,
    ActionPrompt::MoveRobber,
    ActionPrompt::DecideTrade,
    ActionPrompt::DecideAcceptees,
];

/// Observation encoder for one map layout. Values are scaled to roughly
/// [0, 1]. Use `names()` to see what each position means.
#[derive(Clone, Debug)]
pub struct ObservationSpace {
    /// Balanced-dice features (see `DICE_FEATURES`) after the base layout.
    dice: bool,
    /// OpenHex dice odds and history (see `DICE_HISTORY_FEATURES`), next.
    dice_history: bool,
    /// Opponents' hands by resource (see `HAND_FEATURES`).
    hands: bool,
    /// Development-card tracking (see `DEV_FEATURES`).
    devs: bool,
    /// Tile numbers, first player, opponents' this-turn dev activity (see
    /// `with_features`), after the devs block.
    extras: bool,
    /// Settlement spots and trade rates (see `ObsFeatures::spots`).
    spots: bool,
    /// Reach, contention and affordability (see `ObsFeatures::reach`).
    reach: bool,
    /// Spot structure (see `ObsFeatures::structure`), last.
    structure: bool,
    num_land_nodes: usize,
    /// Land node => its port, if any (0..5 resources, 5 = 3:1). Fixed per map.
    node_port: Vec<Option<usize>>,
    land_edges: Vec<usize>,
    size: usize,
}

/// With balanced dice: share of each sum's (2..=12) combinations still in the
/// deck, and share of the deck left.
pub const DICE_FEATURES: usize = 11 + 1;

/// For OpenHex dice, whose odds depend on more than the deck: per total
/// (2..=12) its probability on the next roll if the observer rolls and if
/// the next opponent rolls (times 6), its appearances in the last 5 rolls
/// (/ 3), then per relative seat its 7s rolled (/ 10) and whether it is on
/// the current 7 streak (streak length / 3), and whether the deck is
/// reshuffled before the next roll. Only together with the deck block.
pub const DICE_HISTORY_FEATURES: usize = 11 * 3 + MAX_PLAYERS * 2 + 1;

/// Each opponent's (relative seats 1..=3) resource cards by type. Exact,
/// public knowledge in 1v1, where every resource movement is seen by both
/// players; with more players, steals between others are not public, so
/// this tells the observer slightly more than it could know.
pub const HAND_FEATURES: usize = (MAX_PLAYERS - 1) * 5;

/// Development-card tracking, all public information: the unseen pool by
/// type (the full deck minus everyone's played cards minus the observer's
/// hand), then per opponent seat its expected hidden cards by type (its
/// hidden count spread over the unseen pool), its idle-holding streak and
/// turns since its last purchase.
pub const DEV_FEATURES: usize = 5 + (MAX_PLAYERS - 1) * (5 + 2);

/// Optional blocks of the observation, appended in this order after the
/// base layout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ObsFeatures {
    /// Balanced-dice deck (`DICE_FEATURES`).
    pub dice: bool,
    /// OpenHex dice odds and history (`DICE_HISTORY_FEATURES`).
    pub dice_history: bool,
    /// Opponents' hands by resource (`HAND_FEATURES`).
    pub hands: bool,
    /// Development-card tracking (`DEV_FEATURES`).
    pub devs: bool,
    /// Per land tile its number (one-hot over the 10 numbers) and the share
    /// of that number's rolls left in a balanced-dice deck (`EXTRA_PER_TILE`);
    /// the first player's relative seat (one-hot); per opponent seat whether
    /// it played a development card this turn and cards it bought this turn.
    pub extras: bool,
    /// Settlement spots, all public: per land node (`SPOT_PER_NODE`) its
    /// production pips by resource (robber tile excluded), whether it is a
    /// legal settlement spot now for the observer and for any opponent,
    /// whether it is open under the distance rule, and how close it is by
    /// road to the observer's and to the nearest opponent's network (1 in
    /// the network, 2/3 one road away, 1/3 two roads, else 0; opponents'
    /// roads and buildings block). Then per relative seat
    /// (`SPOT_PER_PLAYER`) its port discount per resource (4:1 0, 3:1 0.5,
    /// 2:1 1) and its open spots within two roads.
    pub spots: bool,
    /// Reach and contention, the relational quantities an expansion plan is
    /// made of (all public): per land node (`REACH_PER_NODE`) the roads the
    /// observer and the nearest opponent need to reach it (/ `REACH_CAP`),
    /// whether the observer reaches it strictly first ("secure") and
    /// whether both reach it equally soon ("contested"), open spots only.
    /// Then per relative seat (`REACH_PER_PLAYER`): secure spots (/ 8), their
    /// pips (/ 40) and best (/ 15), contested spots (/ 4) and best (/ 15),
    /// spots 3 and 4 roads away (/ 6) and their best pips (/ 15), the best
    /// reachable spot's value (pips - 2 roads, / 15), a greedy settle plan
    /// (/ 30), and for road, settlement, city and dev card: can build now,
    /// and exactly one card short.
    pub reach: bool,
    /// Spot structure, board-constant (what the board specialists' first
    /// picks track): the board's pips per resource (/ 20); then per land node
    /// (`STRUCT_PER_NODE`) its share of the board's pips in each resource,
    /// their sum (/ 2), the pips of its weakest resource among wood+brick
    /// (roads), ore+wheat (cities) and wood+brick+sheep+wheat (settlements)
    /// (/ 5), its 6s and 8s (/ 2), duplicated numbers (/ 2), tiles (/ 3) and
    /// whether it is coastal; then one constant marker.
    pub structure: bool,
}

/// Per land node entries of the structure block.
pub const STRUCT_PER_NODE: usize = 5 + 1 + 3 + 4;
/// Board-level entries of the structure block (before the nodes): pips per
/// resource (/ 20), the scarcest and richest resource's pips and their gap
/// (/ 20), 2:1 ports (/ 5), 3:1 ports (/ 4), tiles numbered 6 or 8 (/ 4).
pub const STRUCT_BOARD: usize = 5 + 6;
/// Constant markers closing the block (three, so its size coincides with no
/// other combination of blocks on any map).
const STRUCT_MARKERS: usize = 3;

fn struct_len(num_land_nodes: usize) -> usize {
    STRUCT_BOARD + num_land_nodes * STRUCT_PER_NODE + STRUCT_MARKERS
}

/// Start of the structure block, if present.
pub fn struct_offset_of(obs: &ObservationSpace) -> Option<usize> {
    obs.structure.then(|| obs.size - struct_len(obs.num_land_nodes))
}

/// Per land node entries of the reach block.
pub const REACH_PER_NODE: usize = 4;
/// Per relative seat entries of the reach block.
pub const REACH_PER_PLAYER: usize = 11 + 8;
/// Road distances are capped here (farther counts as this).
pub const REACH_CAP: u8 = 6;

/// The per-node part first, then per seat, then one constant marker (so the
/// block's size never coincides with another combination of blocks).
fn reach_len(num_land_nodes: usize) -> usize {
    num_land_nodes * REACH_PER_NODE + MAX_PLAYERS * REACH_PER_PLAYER + 1
}

/// Start of the reach block, if present.
pub fn reach_offset_of(obs: &ObservationSpace) -> Option<usize> {
    obs.reach.then(|| obs.size - reach_len(obs.num_land_nodes) - if obs.structure { struct_len(obs.num_land_nodes) } else { 0 })
}

impl ObsFeatures {
    pub const ALL: ObsFeatures =
        ObsFeatures { dice: true, dice_history: true, hands: true, devs: true, extras: true, spots: true, reach: true, structure: true };
}

/// Per land node entries of the spots block.
pub const SPOT_PER_NODE: usize = 5 + 2 + 1 + 2;
/// Per relative seat entries of the spots block, after the nodes.
pub const SPOT_PER_PLAYER: usize = 5 + 1;

fn spots_len(num_land_nodes: usize) -> usize {
    num_land_nodes * SPOT_PER_NODE + MAX_PLAYERS * SPOT_PER_PLAYER
}

/// Roads `color` would need to build to reach each node from its network
/// (0 = in the network), up to `cap` (nodes farther away, or unreachable,
/// get `cap`). Opponents' roads can't be crossed, and nodes with an
/// opponent's building are neither reached nor passed through.
pub fn road_distances(b: &crate::board::Board, color: Color, cap: u8) -> [u8; MAX_NODES] {
    let map = &b.map;
    let mut dist = [cap; MAX_NODES];
    let mut queue = [0usize; MAX_NODES];
    let (mut head, mut tail) = (0, 0);
    for n in bits_iter(b.network_bits(color)) {
        dist[n] = 0;
        queue[tail] = n;
        tail += 1;
    }
    while head < tail {
        let u = queue[head];
        head += 1;
        for &(v, ei) in &map.node_edges[u] {
            let v = v as usize;
            let taken = b.buildings[v].is_some_and(|(c, _)| c != color);
            if taken || !map.land_edge[ei] || b.roads[ei].is_some_and(|c| c != color) {
                continue;
            }
            let d = if b.roads[ei] == Some(color) { dist[u] } else { dist[u] + 1 };
            if d < dist[v] && d < cap {
                dist[v] = d;
                queue[tail] = v;
                tail += 1;
            }
        }
    }
    dist
}

/// Per-tile entries of the extras block.
pub const EXTRA_PER_TILE: usize = 10 + 1;

fn extras_len(map: &CatanMap) -> usize {
    map.land_tiles.len() * EXTRA_PER_TILE + MAX_PLAYERS + 2 * (MAX_PLAYERS - 1)
}

fn starting_dev_count(d: DevCard) -> u32 {
    match d {
        DevCard::Knight => 14,
        DevCard::VictoryPoint => 5,
        _ => 2,
    }
}

const PER_NODE: usize = 2 * MAX_PLAYERS + 6; // settlement/city per rel player, port type
const PER_EDGE: usize = MAX_PLAYERS;
const PER_TILE: usize = 6 + 2; // resource one-hot (incl. desert), probability, robber
const PER_PLAYER: usize = 16 + 5; // public stats + production per resource
const PRIVATE: usize = 5 + 5 + 4 + 2; // hand, dev cards, playable flags, rolled/played
const GLOBAL: usize = 7 + 5 + 1 + 1 + 3 + 10 + 1;

impl ObservationSpace {
    pub fn new(map: &CatanMap) -> ObservationSpace {
        ObservationSpace::with_dice(map, false)
    }

    /// The layout a network with `obs_size` inputs was trained on: the base
    /// layout plus whichever optional blocks account for the extra inputs.
    pub fn for_network(map: &CatanMap, obs_size: usize) -> ObservationSpace {
        for k in 0..32 {
            for k2 in 0..8 {
                let f = ObsFeatures {
                    dice: k & 1 != 0,
                    hands: k & 2 != 0,
                    devs: k & 4 != 0,
                    extras: k & 8 != 0,
                    dice_history: k & 16 != 0,
                    spots: k2 & 1 != 0,
                    reach: k2 & 2 != 0,
                    structure: k2 & 4 != 0,
                };
                if f.dice_history && !f.dice {
                    continue;
                }
                let space = ObservationSpace::with_features(map, f);
                if space.len() == obs_size {
                    return space;
                }
            }
        }
        ObservationSpace::new(map)
    }

    pub fn with_dice(map: &CatanMap, dice: bool) -> ObservationSpace {
        ObservationSpace::with_features(map, ObsFeatures { dice, ..Default::default() })
    }

    /// Start of the extras block, if present.
    pub fn extras_offset(&self, map: &CatanMap) -> Option<usize> {
        self.extras.then(|| self.size - extras_len(map) - self.spots_size() - self.reach_size() - self.struct_size())
    }

    /// Start of the spots block, if present (its per-node part first).
    pub fn spots_offset(&self) -> Option<usize> {
        self.spots.then(|| self.size - self.spots_size() - self.reach_size() - self.struct_size())
    }

    fn reach_size(&self) -> usize {
        if self.reach { reach_len(self.num_land_nodes) } else { 0 }
    }

    fn struct_size(&self) -> usize {
        if self.structure { struct_len(self.num_land_nodes) } else { 0 }
    }

    fn spots_size(&self) -> usize {
        if self.spots { spots_len(self.num_land_nodes) } else { 0 }
    }

    pub fn with_features(map: &CatanMap, f: ObsFeatures) -> ObservationSpace {
        let ObsFeatures { dice, dice_history, hands, devs, extras, spots, reach, structure } = f;
        assert!(dice || !dice_history, "the dice-history block comes with the dice block");
        let num_land_nodes = map.land_nodes.iter().filter(|&&x| x).count();
        let land_edges: Vec<usize> = (0..map.edges.len()).filter(|&i| map.land_edge[i]).collect();
        let mut node_port = vec![None; num_land_nodes];
        for (r, nodes) in &map.port_nodes {
            for &nd in nodes {
                if (nd as usize) < num_land_nodes {
                    let port = &mut node_port[nd as usize];
                    assert!(port.is_none(), "node {nd} touches two ports");
                    *port = Some(r.map_or(5, |r| r.idx()));
                }
            }
        }
        let size = num_land_nodes * PER_NODE
            + land_edges.len() * PER_EDGE
            + map.land_tiles.len() * PER_TILE
            + MAX_PLAYERS * PER_PLAYER
            + PRIVATE
            + GLOBAL
            + if dice { DICE_FEATURES } else { 0 }
            + if dice_history { DICE_HISTORY_FEATURES } else { 0 }
            + if hands { HAND_FEATURES } else { 0 }
            + if devs { DEV_FEATURES } else { 0 }
            + if extras { extras_len(map) } else { 0 }
            + if spots { spots_len(num_land_nodes) } else { 0 }
            + if reach { reach_len(num_land_nodes) } else { 0 }
            + if structure { struct_len(num_land_nodes) } else { 0 };
        ObservationSpace { dice, dice_history, hands, devs, extras, spots, reach, structure, num_land_nodes, node_port, land_edges, size }
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn encode(&self, s: &State, me: Color) -> Vec<f32> {
        let mut out = vec![0.0; self.size];
        self.encode_into(s, me, &mut out);
        out
    }

    /// Writes the observation for `me` into `out` (length `len()`).
    pub fn encode_into(&self, s: &State, me: Color, out: &mut [f32]) {
        self.write(s, me, &mut Dense { out, i: 0 });
    }

    /// Writes the nonzero entries of the observation for `me` into `nz` as
    /// (index, value), in increasing index order; returns how many. `nz` is
    /// grown to `len()` entries if shorter.
    pub fn encode_sparse_into(&self, s: &State, me: Color, nz: &mut Vec<(u32, f32)>) -> usize {
        if nz.len() < self.size {
            nz.resize(self.size, (0, 0.0));
        }
        let mut sink = Sparse { out: &mut nz[..], i: 0, n: 0 };
        self.write(s, me, &mut sink);
        sink.n
    }

    /// Human-readable name of each position, in order.
    pub fn names(&self, s: &State) -> Vec<String> {
        let mut names = Vec::with_capacity(self.size);
        self.write(s, s.colors[0], &mut Names(&mut names));
        names
    }

    /// Single definition of the layout, used both for values and names.
    fn write<W: Sink>(&self, s: &State, me: Color, w: &mut W) {
        let b = &s.board;
        let map = &b.map;
        let n = s.colors.len();
        let rel = |c: Color| relative_seat(s, me, c) as usize;
        let player_at = |r: usize| (r < n).then(|| s.player(absolute_color(s, me, r as u8).unwrap()));

        // Nodes: buildings per relative player, then port access.
        for nd in 0..self.num_land_nodes {
            // Slot r * 2 + kind: settlement/city of each relative player.
            let kinds = [BuildingType::Settlement, BuildingType::City];
            let building = b.buildings[nd].map(|(c, k)| rel(c) * 2 + (k == BuildingType::City) as usize);
            w.one_hot(2 * MAX_PLAYERS, building, |k| format!("NODE{nd}_P{}_{:?}", k / 2, kinds[k % 2]));
            w.one_hot(6, self.node_port[nd], |k| format!("NODE{nd}_PORT_{}", port_name(k)));
        }

        // Edges: road owner per relative player.
        for &ei in &self.land_edges {
            let (a, c) = map.edges[ei];
            w.one_hot(MAX_PLAYERS, b.roads[ei].map(rel), |r| format!("EDGE{a}_{c}_P{r}_ROAD"));
        }

        // Tiles.
        for (t, &ti) in map.land_tiles.iter().enumerate() {
            let tile = &map.tiles[ti];
            let kind = tile.resource().map_or(5, |r| r.idx());
            w.one_hot(6, Some(kind), |k| format!("TILE{t}_IS_{}", resource_or_desert(k)));
            let proba = tile.number().map_or(0.0, crate::map::number_probability) as f32;
            w.put(|| format!("TILE{t}_PROBA"), proba * 6.0); // 6/36 => 1.0
            w.put(|| format!("TILE{t}_HAS_ROBBER"), (tile.coordinate == b.robber_coordinate) as u8 as f32);
        }

        // Public per-player features, relative seat order.
        let production = |c: Color| {
            let mut prod = [0.0f32; 5];
            for &nd in &s.player(c).settlements {
                add_production(s, nd, 1.0, &mut prod);
            }
            for &nd in &s.player(c).cities {
                add_production(s, nd, 2.0, &mut prod);
            }
            prod
        };
        for r in 0..MAX_PLAYERS {
            let p = player_at(r);
            let get = |f: &dyn Fn(&crate::state::PlayerState) -> f32| p.map_or(0.0, f);
            w.put(|| format!("P{r}_SEATED"), p.is_some() as u8 as f32);
            w.put(|| format!("P{r}_PUBLIC_VPS"), get(&|p| p.victory_points as f32 / 10.0));
            w.put(|| format!("P{r}_HAS_ROAD"), get(&|p| p.has_road as u8 as f32));
            w.put(|| format!("P{r}_HAS_ARMY"), get(&|p| p.has_army as u8 as f32));
            w.put(|| format!("P{r}_LONGEST_ROAD_LENGTH"), get(&|p| p.longest_road_length as f32 / 15.0));
            w.put(|| format!("P{r}_ROADS_LEFT"), get(&|p| p.roads_available as f32 / 15.0));
            w.put(|| format!("P{r}_SETTLEMENTS_LEFT"), get(&|p| p.settlements_available as f32 / 5.0));
            w.put(|| format!("P{r}_CITIES_LEFT"), get(&|p| p.cities_available as f32 / 4.0));
            w.put(|| format!("P{r}_NUM_RESOURCES"), get(&|p| p.num_resources() as f32 / 20.0));
            w.put(|| format!("P{r}_NUM_DEVS"), get(&|p| p.num_dev_cards() as f32 / 10.0));
            w.put(|| format!("P{r}_HAS_ROLLED"), get(&|p| p.has_rolled as u8 as f32));
            for card in [DevCard::Knight, DevCard::YearOfPlenty, DevCard::Monopoly, DevCard::RoadBuilding] {
                w.put(
                    || format!("P{r}_PLAYED_{}", card.name()),
                    get(&|p| p.played_dev_cards[card.idx()] as f32 / if card == DevCard::Knight { 14.0 } else { 2.0 }),
                );
            }
            w.put(|| format!("P{r}_MUST_DISCARD"), if r < n {
                s.discard_counts[s.seat(absolute_color(s, me, r as u8).unwrap())] as f32 / 10.0
            } else {
                0.0
            });
            let prod = p.map_or([0.0; 5], |p| production(p.color));
            for (k, &v) in prod.iter().enumerate() {
                w.put(|| format!("P{r}_{}_PRODUCTION", RESOURCES[k].name()), v);
            }
        }

        // Private: only the observer's own cards.
        let p = s.player(me);
        for r in RESOURCES {
            w.put(|| format!("SELF_{}_IN_HAND", r.name()), p.resources[r.idx()] as f32 / 10.0);
        }
        for d in DEV_CARDS {
            w.put(|| format!("SELF_{}_IN_HAND", d.name()), p.dev_cards[d.idx()] as f32 / 5.0);
        }
        for d in [DevCard::Knight, DevCard::YearOfPlenty, DevCard::Monopoly, DevCard::RoadBuilding] {
            w.put(|| format!("SELF_CAN_PLAY_{}", d.name()), p.can_play_dev(d) as u8 as f32);
        }
        w.put(|| "SELF_ACTUAL_VPS".into(), p.actual_victory_points as f32 / 10.0);
        w.put(|| "SELF_PLAYED_DEV_THIS_TURN".into(), p.has_played_development_card_in_turn as u8 as f32);

        // Global.
        for pr in PROMPTS {
            w.put(|| format!("PROMPT_{pr:?}"), (s.current_prompt == pr) as u8 as f32);
        }
        for r in RESOURCES {
            w.put(|| format!("BANK_{}", r.name()), s.bank[r.idx()] as f32 / 19.0);
        }
        w.put(|| "DEV_DECK_SIZE".into(), s.dev_deck.len() as f32 / 25.0);
        w.put(|| "FREE_ROADS".into(), s.free_roads_available as f32 / 2.0);
        // Whose turn it is, relative to the observer (one-hot over 1..=3;
        // all zero means it is the observer's own turn).
        let turn_rel = relative_seat(s, me, s.colors[s.current_turn_index]) as usize;
        for r in 1..MAX_PLAYERS {
            w.put(|| format!("TURN_IS_P{r}"), (turn_rel == r) as u8 as f32);
        }
        let trade = s.current_trade.map(|(o, a, _)| (o, a)).unwrap_or(([0; 5], [0; 5]));
        for r in RESOURCES {
            w.put(|| format!("TRADE_OFFERING_{}", r.name()), trade.0[r.idx()] as f32 / 5.0);
        }
        for r in RESOURCES {
            w.put(|| format!("TRADE_ASKING_{}", r.name()), trade.1[r.idx()] as f32 / 5.0);
        }
        w.put(|| "NUM_TURNS".into(), (s.num_turns as f32 / 400.0).min(1.0));

        if self.dice {
            // An empty deck refills before the next roll, so it counts as
            // full (as does a game without balanced dice).
            let deck = s.dice_deck.filter(|d| !d.is_empty()).unwrap_or_else(DiceDeck::full);
            let mut left = [0u32; 13];
            for (a, b) in deck.remaining() {
                left[(a + b) as usize] += 1;
            }
            for (sum, &n) in left.iter().enumerate().skip(2) {
                let combos = 6 - (7 - sum as i32).unsigned_abs();
                w.put(|| format!("DICE_LEFT_{sum}"), n as f32 / combos as f32);
            }
            w.put(|| "DICE_DECK_LEFT".into(), deck.len() as f32 / 36.0);
        }

        if self.dice_history {
            let n = s.colors.len();
            let my_seat = s.seat(me);
            let deck = s.dice_deck.unwrap_or_else(DiceDeck::full);
            let fair = |t: usize| (6 - (7 - t as i32).unsigned_abs()) as f32 / 36.0;
            let odds = |seat: usize| match s.dice_deck {
                Some(d) => d.probabilities(seat, n),
                None => std::array::from_fn(|t| if (2..=12).contains(&t) { fair(t) } else { 0.0 }),
            };
            for (who, seat) in [("ME", my_seat), ("NEXT", (my_seat + 1) % n)] {
                let p = odds(seat);
                for t in 2..=12 {
                    w.put(|| format!("DICE_P_{t}_IF_{who}_ROLLS"), p[t] * 6.0);
                }
            }
            for t in 2..=12u8 {
                w.put(|| format!("DICE_RECENT_{t}"), deck.recent_count(t) as f32 / 3.0);
            }
            let (streak_seat, streak) = deck.seven_streak();
            for r in 0..MAX_PLAYERS {
                let seat = (r < n).then(|| s.seat(absolute_color(s, me, r as u8).unwrap()));
                w.put(|| format!("P{r}_SEVENS_ROLLED"), seat.map_or(0.0, |k| deck.sevens(k) as f32 / 10.0));
                let on_streak = seat.is_some_and(|k| streak > 0 && k == streak_seat);
                w.put(|| format!("P{r}_SEVEN_STREAK"), if on_streak { streak as f32 / 3.0 } else { 0.0 });
            }
            let reshuffles = s.dice_deck.is_some_and(|d| d.next_deck().len() > d.len());
            w.put(|| "DICE_RESHUFFLE_BEFORE_NEXT_ROLL".into(), reshuffles as u8 as f32);
        }

        if self.hands {
            for r in 1..MAX_PLAYERS {
                let p = player_at(r);
                for res in RESOURCES {
                    w.put(
                        || format!("P{r}_{}_IN_HAND", res.name()),
                        p.map_or(0.0, |p| p.resources[res.idx()] as f32 / 10.0),
                    );
                }
            }
        }

        if self.devs {
            let mine = &s.player(me).dev_cards;
            let mut unseen = [0f32; 5];
            for d in DEV_CARDS {
                let played: u32 = s.players.iter().map(|p| p.played_dev_cards[d.idx()]).sum();
                unseen[d.idx()] = starting_dev_count(d).saturating_sub(played + mine[d.idx()]) as f32;
            }
            let total: f32 = unseen.iter().sum();
            for d in DEV_CARDS {
                w.put(|| format!("DEV_UNSEEN_{}", d.name()), unseen[d.idx()] / starting_dev_count(d) as f32);
            }
            for r in 1..MAX_PLAYERS {
                let p = player_at(r);
                let hidden = p.map_or(0, |p| p.num_dev_cards()) as f32;
                for d in DEV_CARDS {
                    let expected = if total > 0.0 { hidden * unseen[d.idx()] / total } else { 0.0 };
                    w.put(|| format!("P{r}_EXPECTED_{}", d.name()), expected / 2.0);
                }
                w.put(|| format!("P{r}_DEV_IDLE_TURNS"), p.map_or(0.0, |p| (p.dev_idle_turns as f32 / 10.0).min(1.0)));
                w.put(
                    || format!("P{r}_TURNS_SINCE_DEV_BUY"),
                    p.and_then(|p| p.last_dev_buy_turn).map_or(0.0, |t| ((s.num_turns - t) as f32 / 20.0).min(1.0)),
                );
            }
        }

        if self.extras {
            let deck = s.dice_deck.filter(|d| !d.is_empty()).unwrap_or_else(DiceDeck::full);
            let mut left = [0u32; 13];
            for (a, b) in deck.remaining() {
                left[(a + b) as usize] += 1;
            }
            for (t, &ti) in map.land_tiles.iter().enumerate() {
                let number = map.tiles[ti].number();
                // Numbers 2..=6 and 8..=12 as 0..=9.
                let hot = number.map(|n| n as usize - if n < 7 { 2 } else { 3 });
                w.one_hot(10, hot, |k| format!("TILE{t}_NUMBER_{}", k + if k < 5 { 2 } else { 3 }));
                let share = number.map_or(0.0, |n| left[n as usize] as f32 / (6 - (7 - n as i32).unsigned_abs()) as f32);
                w.put(|| format!("TILE{t}_ROLLS_LEFT"), share);
            }
            w.one_hot(MAX_PLAYERS, Some(rel(s.colors[0])), |r| format!("FIRST_PLAYER_IS_P{r}"));
            for r in 1..MAX_PLAYERS {
                let p = player_at(r);
                w.put(
                    || format!("P{r}_PLAYED_DEV_THIS_TURN"),
                    p.map_or(0.0, |p| p.has_played_development_card_in_turn as u8 as f32),
                );
                w.put(|| format!("P{r}_DEVS_BOUGHT_THIS_TURN"), p.map_or(0.0, |p| p.dev_bought_this_turn as f32 / 5.0));
            }
        }

        if self.spots {
            const CAP: u8 = 3;
            let seats: Vec<Color> = (0..n).map(|r| absolute_color(s, me, r as u8).unwrap()).collect();
            let dists: Vec<[u8; MAX_NODES]> = seats.iter().map(|&c| road_distances(b, c, CAP)).collect();
            let closeness = |d: u8| (CAP - d) as f32 / CAP as f32;
            let legal = |c: Color, nd: usize| {
                b.buildable[nd] && (s.is_initial_build_phase || b.in_network(c, nd as NodeId))
            };
            for nd in 0..self.num_land_nodes {
                let mut pips = [0.0f32; 5];
                add_production(s, nd as NodeId, 6.0, &mut pips);
                for (k, &v) in pips.iter().enumerate() {
                    w.put(|| format!("SPOT{nd}_{}_PIPS", RESOURCES[k].name()), v);
                }
                w.put(|| format!("SPOT{nd}_LEGAL_FOR_ME"), legal(me, nd) as u8 as f32);
                w.put(|| format!("SPOT{nd}_LEGAL_FOR_OPPONENT"), seats[1..].iter().any(|&c| legal(c, nd)) as u8 as f32);
                w.put(|| format!("SPOT{nd}_OPEN"), b.buildable[nd] as u8 as f32);
                w.put(|| format!("SPOT{nd}_MY_ROAD_CLOSENESS"), closeness(dists[0][nd]));
                let nearest = dists[1..].iter().map(|d| d[nd]).min().unwrap_or(CAP);
                w.put(|| format!("SPOT{nd}_OPPONENT_ROAD_CLOSENESS"), closeness(nearest));
            }
            for r in 0..MAX_PLAYERS {
                let mask = if r < n { b.port_mask(seats[r]) } else { 0 };
                for res in RESOURCES {
                    let rate = if r >= n { 4 } else if mask >> res.idx() & 1 == 1 { 2 } else if mask >> 5 & 1 == 1 { 3 } else { 4 };
                    w.put(|| format!("P{r}_{}_PORT_DISCOUNT", res.name()), (4 - rate) as f32 / 2.0);
                }
                let reachable = if r < n {
                    (0..self.num_land_nodes).filter(|&nd| b.buildable[nd] && dists[r][nd] <= 2).count()
                } else {
                    0
                };
                w.put(|| format!("P{r}_OPEN_SPOTS_WITHIN_2_ROADS"), reachable as f32 / 10.0);
            }
        }

        if self.reach {
            let seats: Vec<Color> = (0..n).map(|r| absolute_color(s, me, r as u8).unwrap()).collect();
            let dists: Vec<[u8; MAX_NODES]> = seats.iter().map(|&c| road_distances(b, c, REACH_CAP)).collect();
            let pips = |nd: usize| {
                let mut p = [0.0f32; 5];
                add_production(s, nd as NodeId, 36.0, &mut p);
                p.iter().sum::<f32>()
            };
            // Nearest opponent's distance to each node, from each seat's point of view.
            let nearest_other = |r: usize, nd: usize| dists.iter().enumerate().filter(|&(j, _)| j != r).map(|(_, d)| d[nd]).min().unwrap_or(REACH_CAP);
            for nd in 0..self.num_land_nodes {
                let (mine, theirs) = (dists[0][nd], nearest_other(0, nd));
                let open = b.buildable[nd];
                w.put(|| format!("REACH{nd}_MINE"), mine as f32 / REACH_CAP as f32);
                w.put(|| format!("REACH{nd}_OPPONENT"), theirs as f32 / REACH_CAP as f32);
                w.put(|| format!("REACH{nd}_SECURE"), (open && mine < theirs && mine < REACH_CAP) as u8 as f32);
                w.put(|| format!("REACH{nd}_CONTESTED"), (open && mine == theirs && mine < REACH_CAP) as u8 as f32);
            }
            for r in 0..MAX_PLAYERS {
                let p = (r < n).then(|| s.player(seats[r]));
                let (mut secure, mut secure_pips, mut secure_top, mut contested, mut contested_top) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
                let (mut far3, mut far3_top, mut far4, mut far4_top, mut best_value) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
                let mut plan: Vec<f32> = Vec::new();
                if r < n {
                    for nd in 0..self.num_land_nodes {
                        if !b.buildable[nd] {
                            continue;
                        }
                        let (d, t, pp) = (dists[r][nd], nearest_other(r, nd), pips(nd));
                        if d >= 5 {
                            continue;
                        }
                        if d == 3 { far3 += 1.0; far3_top = far3_top.max(pp); }
                        if d == 4 { far4 += 1.0; far4_top = far4_top.max(pp); }
                        if d < t { secure += 1.0; secure_pips += pp; secure_top = secure_top.max(pp); }
                        else if d == t { contested += 1.0; contested_top = contested_top.max(pp); }
                        if d <= t && d <= 4 {
                            let v = pp - 2.0 * d as f32;
                            best_value = best_value.max(v);
                            plan.push(v);
                        }
                    }
                }
                plan.sort_by(|x, y| y.total_cmp(x));
                let plan_v: f32 = plan.iter().take(5).zip([1.0, 0.8, 0.64, 0.51, 0.41]).map(|(v, f)| v.max(0.0) * f).sum();
                w.put(|| format!("P{r}_SECURE_SPOTS"), secure.min(8.0) / 8.0);
                w.put(|| format!("P{r}_SECURE_PIPS"), secure_pips.min(80.0) / 40.0);
                w.put(|| format!("P{r}_SECURE_TOP"), secure_top / 15.0);
                w.put(|| format!("P{r}_CONTESTED_SPOTS"), contested.min(4.0) / 4.0);
                w.put(|| format!("P{r}_CONTESTED_TOP"), contested_top / 15.0);
                w.put(|| format!("P{r}_FAR3_SPOTS"), far3.min(6.0) / 6.0);
                w.put(|| format!("P{r}_FAR3_TOP"), far3_top / 15.0);
                w.put(|| format!("P{r}_FAR4_SPOTS"), far4.min(6.0) / 6.0);
                w.put(|| format!("P{r}_FAR4_TOP"), far4_top / 15.0);
                w.put(|| format!("P{r}_BEST_REACH_VALUE"), best_value / 15.0);
                w.put(|| format!("P{r}_SETTLE_PLAN"), plan_v / 30.0);
                for (name, cost) in [("ROAD", &ROAD_COST), ("SETTLEMENT", &SETTLEMENT_COST), ("CITY", &CITY_COST), ("DEV", &DEV_CARD_COST)] {
                    let short: u32 = p.map_or(99, |p| cost.iter().zip(&p.resources).map(|(&c, &h)| c.saturating_sub(h)).sum());
                    w.put(|| format!("P{r}_CAN_{name}"), (short == 0) as u8 as f32);
                    w.put(|| format!("P{r}_SHORT_{name}"), (short == 1) as u8 as f32);
                }
            }
            w.put(|| "REACH_BLOCK".to_string(), 1.0);
        }

        if self.structure {
            // Board-constant spot structure: what the board specialists' first picks track.
            let map = &b.map;
            let pips_of = |num: u8| (6 - (7 - num as i32).abs()).max(0) as f32;
            let mut board_pips = [0.0f32; 5];
            for &ti in &map.land_tiles {
                let t = &map.tiles[ti];
                if let (Some(r), Some(num)) = (t.resource(), t.number()) {
                    board_pips[r.idx()] += pips_of(num);
                }
            }
            for r in 0..5 {
                w.put(|| format!("BOARD_{}_PIPS", RESOURCES[r].name()), board_pips[r] / 20.0);
            }
            let (lo, hi) = (board_pips.iter().copied().fold(f32::INFINITY, f32::min), board_pips.iter().copied().fold(0.0, f32::max));
            w.put(|| "BOARD_SCARCEST_PIPS".to_string(), lo / 20.0);
            w.put(|| "BOARD_RICHEST_PIPS".to_string(), hi / 20.0);
            w.put(|| "BOARD_PIPS_GAP".to_string(), (hi - lo) / 20.0);
            w.put(|| "BOARD_PORTS_2TO1".to_string(), map.port_nodes.iter().filter(|(r, _)| r.is_some()).count() as f32 / 5.0);
            w.put(|| "BOARD_PORTS_3TO1".to_string(), map.port_nodes.iter().filter(|(r, _)| r.is_none()).count() as f32 / 4.0);
            w.put(|| "BOARD_RED_TILES".to_string(), map.land_tiles.iter().filter(|&&ti| matches!(map.tiles[ti].number(), Some(6) | Some(8))).count() as f32 / 4.0);
            for nd in 0..self.num_land_nodes {
                let mut p = [0.0f32; 5];
                let mut nums: Vec<u8> = Vec::new();
                let tiles = map.adjacent_tiles[nd].len();
                for &ti in &map.adjacent_tiles[nd] {
                    let t = &map.tiles[ti];
                    if let (Some(r), Some(num)) = (t.resource(), t.number()) {
                        p[r.idx()] += pips_of(num);
                        nums.push(num);
                    }
                }
                let mut scarce = 0.0;
                for r in 0..5 {
                    let share = p[r] / board_pips[r].max(1.0);
                    scarce += share;
                    w.put(|| format!("STRUCT{nd}_{}_SHARE", RESOURCES[r].name()), share);
                }
                w.put(|| format!("STRUCT{nd}_SCARCE"), scarce / 2.0);
                // Resource indices: wood 0, brick 1, sheep 2, wheat 3, ore 4 (types::RESOURCES).
                w.put(|| format!("STRUCT{nd}_ROAD_PAIR"), p[0].min(p[1]) / 5.0);
                w.put(|| format!("STRUCT{nd}_CITY_PAIR"), p[4].min(p[3]) / 5.0);
                w.put(|| format!("STRUCT{nd}_SETTLE_SET"), p[0].min(p[1]).min(p[2]).min(p[3]) / 5.0);
                let red = nums.iter().filter(|&&k| k == 6 || k == 8).count();
                let mut distinct = nums.clone();
                distinct.sort_unstable();
                distinct.dedup();
                w.put(|| format!("STRUCT{nd}_RED"), red as f32 / 2.0);
                w.put(|| format!("STRUCT{nd}_DUP_NUMBERS"), (nums.len() - distinct.len()) as f32 / 2.0);
                w.put(|| format!("STRUCT{nd}_TILES"), tiles as f32 / 3.0);
                w.put(|| format!("STRUCT{nd}_COAST"), (tiles < 3) as u8 as f32);
            }
            for k in 0..STRUCT_MARKERS {
                w.put(|| format!("STRUCT_BLOCK_{k}"), 1.0);
            }
        }

        w.finish(self.size);
    }
}

pub fn add_production(s: &State, node: NodeId, weight: f32, prod: &mut [f32; 5]) {
    let map = &s.board.map;
    for &ti in &map.adjacent_tiles[node as usize] {
        let tile = &map.tiles[ti];
        if tile.coordinate == s.board.robber_coordinate {
            continue;
        }
        if let (Some(r), Some(num)) = (tile.resource(), tile.number()) {
            // Expected cards of this resource per dice roll.
            prod[r.idx()] += weight * crate::map::number_probability(num) as f32;
        }
    }
}

fn port_name(k: usize) -> &'static str {
    if k == 5 { "THREE_TO_ONE" } else { RESOURCES[k].name() }
}

fn resource_or_desert(k: usize) -> &'static str {
    if k == 5 { "DESERT" } else { RESOURCES[k].name() }
}

/// Receives the observation layout in order: values (dense or nonzero-only)
/// or names, so the layout is defined once. Generic, so each sink gets its
/// own copy of `write` with no per-value dispatch.
trait Sink {
    fn put(&mut self, name: impl FnOnce() -> String, value: f32);

    /// `n` positions, all 0 except 1 at `hot` (if any); `name(k)` names
    /// position k.
    #[inline(always)]
    fn one_hot(&mut self, n: usize, hot: Option<usize>, name: impl Fn(usize) -> String) {
        for k in 0..n {
            self.put(|| name(k), (hot == Some(k)) as u8 as f32);
        }
    }
    /// Positions received so far.
    fn written(&self) -> usize;

    fn finish(&self, size: usize) {
        assert_eq!(self.written(), size, "observation layout and size disagree");
    }
}

struct Dense<'a> {
    out: &'a mut [f32],
    i: usize,
}

impl Sink for Dense<'_> {
    #[inline(always)]
    fn put(&mut self, _name: impl FnOnce() -> String, value: f32) {
        self.out[self.i] = value;
        self.i += 1;
    }
    fn written(&self) -> usize {
        self.i
    }
}

/// Nonzero entries as (index, value); `n` of them kept.
struct Sparse<'a> {
    out: &'a mut [(u32, f32)],
    i: usize,
    n: usize,
}

impl Sink for Sparse<'_> {
    #[inline(always)]
    fn put(&mut self, _name: impl FnOnce() -> String, value: f32) {
        // Branch-free: always write, keep only nonzeros.
        self.out[self.n] = (self.i as u32, value);
        self.n += (value != 0.0) as usize;
        self.i += 1;
    }

    /// Skips the zeros: only the hot position is recorded.
    #[inline(always)]
    fn one_hot(&mut self, n: usize, hot: Option<usize>, _name: impl Fn(usize) -> String) {
        if let Some(k) = hot {
            assert!(k < n);
            self.out[self.n] = ((self.i + k) as u32, 1.0);
            self.n += 1;
        }
        self.i += n;
    }
    fn written(&self) -> usize {
        self.i
    }
}

struct Names<'a>(&'a mut Vec<String>);

impl Sink for Names<'_> {
    fn put(&mut self, name: impl FnOnce() -> String, _value: f32) {
        self.0.push(name());
    }
    fn written(&self) -> usize {
        self.0.len()
    }
}

/// A board symmetry as permutations: entry `i` of a transformed observation
/// (or action vector: mask, policy, logits) is entry `obs[i]` (`actions[i]`)
/// of the original.
#[derive(Clone, Debug)]
pub struct Symmetry {
    pub obs: Vec<usize>,
    pub actions: Vec<usize>,
}

/// The map's symmetries (up to 12 for the hexagonal base board: 6 rotations,
/// each optionally mirrored; the identity first) for data augmentation. They
/// relocate everything attached to nodes, edges and tiles, ports included;
/// other features are unchanged. Port positions are not symmetric on the
/// real board, so transformed positions are legal but unusual layouts.
pub fn symmetries(map: &CatanMap, obs: &ObservationSpace, act: &ActionSpace) -> Vec<Symmetry> {
    let land: Vec<&crate::map::Tile> = map.land_tiles.iter().map(|&ti| &map.tiles[ti]).collect();
    let n_nodes = obs.num_land_nodes;
    let edge_pos: std::collections::HashMap<EdgeId, usize> =
        obs.land_edges.iter().enumerate().map(|(j, &ei)| (map.edges[ei], j)).collect();
    let rot = |(x, y, z): Coordinate| (-z, -x, -y);
    let mut out = Vec::new();
    for mirror in [false, true] {
        for r in 0..6 {
            let transform = |c: Coordinate| {
                let mut c = if mirror { (c.0, c.2, c.1) } else { c };
                for _ in 0..r {
                    c = rot(c);
                }
                c
            };
            let Some(tile_perm): Option<Vec<usize>> =
                land.iter().map(|t| land.iter().position(|u| u.coordinate == transform(t.coordinate))).collect()
            else {
                continue;
            };
            // Find the corner relabeling that maps nodes consistently.
            let node_perm = (0..6).find_map(|k| {
                let corner = |c: usize| if mirror { (k + 6 - c) % 6 } else { (c + k) % 6 };
                let mut node = vec![usize::MAX; n_nodes];
                for (t, tile) in land.iter().enumerate() {
                    for c in 0..6 {
                        let (a, b) = (tile.nodes[c] as usize, land[tile_perm[t]].nodes[corner(c)] as usize);
                        if node[a] != usize::MAX && node[a] != b {
                            return None;
                        }
                        node[a] = b;
                    }
                }
                let mut seen = vec![false; n_nodes];
                node.iter().all(|&b| b < n_nodes && !std::mem::replace(&mut seen[b], true)).then_some(node)
            });
            let Some(node) = node_perm else { continue };
            let Some(edge_perm): Option<Vec<usize>> = obs
                .land_edges
                .iter()
                .map(|&ei| {
                    let (a, b) = map.edges[ei];
                    edge_pos.get(&edge(node[a as usize] as NodeId, node[b as usize] as NodeId)).copied()
                })
                .collect()
            else {
                continue;
            };

            let mut o: Vec<usize> = (0..obs.len()).collect();
            let o_edge = n_nodes * PER_NODE;
            let o_tile = o_edge + obs.land_edges.len() * PER_EDGE;
            for (n, &m) in node.iter().enumerate() {
                for j in 0..PER_NODE {
                    o[m * PER_NODE + j] = n * PER_NODE + j;
                }
            }
            for (e, &f) in edge_perm.iter().enumerate() {
                for j in 0..PER_EDGE {
                    o[o_edge + f * PER_EDGE + j] = o_edge + e * PER_EDGE + j;
                }
            }
            for (t, &u) in tile_perm.iter().enumerate() {
                for j in 0..PER_TILE {
                    o[o_tile + u * PER_TILE + j] = o_tile + t * PER_TILE + j;
                }
            }
            if let Some(o_x) = obs.extras_offset(map) {
                for (t, &u) in tile_perm.iter().enumerate() {
                    for j in 0..EXTRA_PER_TILE {
                        o[o_x + u * EXTRA_PER_TILE + j] = o_x + t * EXTRA_PER_TILE + j;
                    }
                }
            }
            if let Some(o_s) = obs.spots_offset() {
                for (n, &m) in node.iter().enumerate() {
                    for j in 0..SPOT_PER_NODE {
                        o[o_s + m * SPOT_PER_NODE + j] = o_s + n * SPOT_PER_NODE + j;
                    }
                }
            }
            if let Some(o_r) = reach_offset_of(obs) {
                // Per-node reach entries move with the nodes; the per-seat
                // summaries and the marker are symmetric.
                for (n, &m) in node.iter().enumerate() {
                    for j in 0..REACH_PER_NODE {
                        o[o_r + m * REACH_PER_NODE + j] = o_r + n * REACH_PER_NODE + j;
                    }
                }
            }
            if let Some(o_t) = struct_offset_of(obs) {
                // Board totals and the marker are symmetric; the per-node entries move with the nodes.
                let base = o_t + STRUCT_BOARD;
                for (n, &m) in node.iter().enumerate() {
                    for j in 0..STRUCT_PER_NODE {
                        o[base + m * STRUCT_PER_NODE + j] = base + n * STRUCT_PER_NODE + j;
                    }
                }
            }

            let mut a: Vec<usize> = (0..act.len()).collect();
            for (e, &f) in edge_perm.iter().enumerate() {
                a[act.road_slot[obs.land_edges[f]]] = act.road_slot[obs.land_edges[e]];
            }
            for (n, &m) in node.iter().enumerate() {
                a[act.o_settlement + m] = act.o_settlement + n;
                a[act.o_city + m] = act.o_city + n;
            }
            for (t, &u) in tile_perm.iter().enumerate() {
                for v in 0..ROBBER_VICTIMS {
                    a[act.o_robber + u * ROBBER_VICTIMS + v] = act.o_robber + t * ROBBER_VICTIMS + v;
                }
            }
            out.push(Symmetry { obs: o, actions: a });
        }
    }
    out
}
