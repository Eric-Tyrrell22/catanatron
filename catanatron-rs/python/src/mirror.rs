//! `Mirror`: a local copy of a game played on OpenHex (1v1), kept in step
//! by applying the server's events, used to search for our moves.
//!
//! Everything crossing this boundary is in OpenHex's frame and names:
//! hexes and vertices as (x, y, z), colors "red"/"blue"/..., resources
//! "wood"/..., dev cards "knight"/"victoryPoint"/... Internally the board is
//! turned 180 degrees into our frame (`catanatron::openhex::turn_hex`), where
//! OpenHex's harbors sit on our harbor slots.
//!
//! Hidden information: the opponent's hand is set from the public snapshot
//! (`sync`: in 1v1 every card is in the bank or a hand), and the types of
//! its unplayed dev cards are arbitrary until it plays one (`ensure_dev`).

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use catanatron::map::{build_map, CatanMap, MapType, NumberPlacement};
use catanatron::openhex::{canonical_edge, turn_hex, turn_vertex, Geometry, Vertex};
use catanatron::rng::Rng;
use catanatron::state::DiceMode;
use catanatron::types::*;
use catanatron::{FriendlyRobber, Game, GameConfig};

const BANK_TOTAL: u32 = 19;

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn color(name: &str) -> PyResult<Color> {
    match name {
        "red" => Ok(Color::Red),
        "blue" => Ok(Color::Blue),
        "orange" => Ok(Color::Orange),
        "white" => Ok(Color::White),
        other => Err(err(format!("unknown color {other}"))),
    }
}

fn color_name(c: Color) -> &'static str {
    match c {
        Color::Red => "red",
        Color::Blue => "blue",
        Color::Orange => "orange",
        Color::White => "white",
    }
}

pub(crate) fn resource(name: &str) -> PyResult<Resource> {
    match name {
        "wood" => Ok(Resource::Wood),
        "brick" => Ok(Resource::Brick),
        "sheep" => Ok(Resource::Sheep),
        "wheat" => Ok(Resource::Wheat),
        "ore" => Ok(Resource::Ore),
        other => Err(err(format!("unknown resource {other}"))),
    }
}

pub(crate) fn resource_name(r: Resource) -> &'static str {
    ["wood", "brick", "sheep", "wheat", "ore"][r.idx()]
}

fn terrain(name: &str) -> PyResult<Option<Resource>> {
    match name {
        "forest" => Ok(Some(Resource::Wood)),
        "hills" => Ok(Some(Resource::Brick)),
        "pasture" => Ok(Some(Resource::Sheep)),
        "fields" => Ok(Some(Resource::Wheat)),
        "mountains" => Ok(Some(Resource::Ore)),
        "desert" => Ok(None),
        other => Err(err(format!("unknown terrain {other}"))),
    }
}

type OpenHexTiles = [((i8, i8, i8), String, Option<u8>)];
type OpenHexPorts = [((i8, i8, i8), (i8, i8, i8), Option<String>)];

/// An OpenHex board (tiles: hex, terrain, number; ports: two vertices and a
/// resource, None for 3:1) as our map, with the vertex geometry.
fn openhex_map(tiles: &OpenHexTiles, ports: &OpenHexPorts) -> PyResult<(Arc<CatanMap>, Geometry)> {
    let base = build_map(MapType::Base, NumberPlacement::OfficialSpiral, &mut Rng::new(0));
    let geo = Geometry::new(&base).map_err(err)?;
    let mut land = Vec::new();
    for (h, t, n) in tiles {
        land.push((turn_hex(*h), terrain(t)?, *n));
    }
    let mut port_list = Vec::new();
    for (a, b, r) in ports {
        let pair = [geo.node(turn_vertex(*a)).map_err(err)?, geo.node(turn_vertex(*b)).map_err(err)?];
        port_list.push((pair, r.as_deref().map(resource).transpose()?));
    }
    let map = Arc::new(base.with_layout(&land, &port_list).map_err(err)?);
    Ok((map, geo))
}

/// An OpenHex board from its compact text form (`catanatron_env.board_spec`):
/// tiles `x,y,z,terrain,number` joined by `;`, then `|`, then ports
/// `x,y,z,x,y,z,resource` joined by `;` (`-` for no number / a 3:1 port).
pub(crate) fn openhex_map_spec(spec: &str) -> PyResult<Arc<CatanMap>> {
    let bad = || PyValueError::new_err(format!("bad board spec: {spec}"));
    let (tiles_s, ports_s) = spec.split_once('|').ok_or_else(bad)?;
    let int = |x: &str| x.parse::<i8>().map_err(|_| bad());
    let mut tiles = Vec::new();
    for t in tiles_s.split(';') {
        let f: Vec<&str> = t.split(',').collect();
        if f.len() != 5 {
            return Err(bad());
        }
        let number = if f[4] == "-" { None } else { Some(f[4].parse::<u8>().map_err(|_| bad())?) };
        tiles.push(((int(f[0])?, int(f[1])?, int(f[2])?), f[3].to_string(), number));
    }
    let mut ports = Vec::new();
    for p in ports_s.split(';') {
        let f: Vec<&str> = p.split(',').collect();
        if f.len() != 7 {
            return Err(bad());
        }
        let r = if f[6] == "-" { None } else { Some(f[6].to_string()) };
        ports.push(((int(f[0])?, int(f[1])?, int(f[2])?), (int(f[3])?, int(f[4])?, int(f[5])?), r));
    }
    Ok(openhex_map(&tiles, &ports)?.0)
}

fn dev_card(name: &str) -> PyResult<DevCard> {
    match name {
        "knight" => Ok(DevCard::Knight),
        "yearOfPlenty" => Ok(DevCard::YearOfPlenty),
        "monopoly" => Ok(DevCard::Monopoly),
        "roadBuilding" => Ok(DevCard::RoadBuilding),
        "victoryPoint" => Ok(DevCard::VictoryPoint),
        other => Err(err(format!("unknown dev card {other}"))),
    }
}

/// A move in OpenHex terms: (kind, args...).
type Move = (String, Vec<String>);

#[pyclass(name = "Mirror", unsendable)]
pub struct PyMirror {
    game: Game,
    geo: Geometry,
    rng: Rng,
}

impl PyMirror {
    /// Before `buyer` draws a seen `card`: if the mirror dealt every copy
    /// left to another seat's unseen cards, give that seat a different card
    /// from the deck instead (equally consistent with what we know).
    fn make_drawable(&mut self, buyer: Color, card: DevCard) -> PyResult<()> {
        let s = &mut self.game.state;
        if s.dev_deck.contains(&card) {
            return Ok(());
        }
        let holder = s
            .colors
            .iter()
            .copied()
            .find(|&c| c != buyer && s.player(c).dev_cards[card.idx()] > 0)
            .ok_or_else(|| err(format!("{} drew a {} but none is unaccounted for", buyer.name(), card.name())))?;
        let pos = s
            .dev_deck
            .iter()
            .position(|&d| d != card && d != DevCard::VictoryPoint)
            .or_else(|| s.dev_deck.iter().position(|&d| d != card))
            .ok_or_else(|| err("no card to swap into the deck"))?;
        let other = s.dev_deck[pos];
        s.dev_deck[pos] = card;
        let p = s.player_mut(holder);
        p.dev_cards[card.idx()] -= 1;
        p.dev_cards[other.idx()] += 1;
        p.actual_victory_points = p.victory_points + p.dev_cards[DevCard::VictoryPoint.idx()];
        Ok(())
    }

    fn node(&self, v: Vertex) -> PyResult<NodeId> {
        self.geo.node(turn_vertex(v)).map_err(err)
    }

    fn edge(&self, a: Vertex, b: Vertex) -> PyResult<EdgeId> {
        self.geo.edge(turn_vertex(a), turn_vertex(b)).map_err(err)
    }

    fn vertex(&self, n: NodeId) -> Vertex {
        turn_vertex(self.geo.node_vertex[&n])
    }

    fn edge_vertices(&self, e: EdgeId) -> (Vertex, Vertex) {
        let (a, b) = self.geo.edge_vertices[&e];
        canonical_edge(turn_vertex(a), turn_vertex(b))
    }

    /// Applies `kind` by `who`, which must be playable now, with an
    /// optional forced random outcome.
    fn play(&mut self, who: Color, kind: ActionKind, result: Option<ActionResult>) -> PyResult<()> {
        let action = Action::new(who, kind);
        if !self.game.playable_actions.contains(&action) {
            return Err(err(format!(
                "mirror out of step: {action} is not playable (current {} prompt {:?})",
                self.game.state.current_color().name(),
                self.game.state.current_prompt
            )));
        }
        self.game.execute_unchecked(action, result).map_err(err)?;
        Ok(())
    }

    fn to_move(&self, a: &Action) -> PyResult<Move> {
        let s = |x: &str| x.to_string();
        let v = |v: Vertex| format!("{},{},{}", v.0, v.1, v.2);
        Ok(match a.kind {
            ActionKind::Roll => (s("roll"), vec![]),
            ActionKind::EndTurn => (s("end"), vec![]),
            ActionKind::BuildSettlement(n) => (s("settlement"), vec![v(self.vertex(n))]),
            ActionKind::BuildCity(n) => (s("city"), vec![v(self.vertex(n))]),
            ActionKind::BuildRoad(e) => {
                let (a, b) = self.edge_vertices(e);
                (s("road"), vec![v(a), v(b)])
            }
            ActionKind::BuyDevelopmentCard => (s("buy"), vec![]),
            ActionKind::PlayKnightCard => (s("knight"), vec![]),
            ActionKind::PlayRoadBuilding => (s("roadBuilding"), vec![]),
            ActionKind::PlayMonopoly(r) => (s("monopoly"), vec![s(resource_name(r))]),
            ActionKind::PlayYearOfPlenty(a, b) => {
                let mut args = vec![s(resource_name(a))];
                if let Some(b) = b {
                    args.push(s(resource_name(b)));
                }
                (s("yearOfPlenty"), args)
            }
            ActionKind::MoveRobber { coordinate, victim } => {
                let h = turn_hex(coordinate);
                let mut args = vec![format!("{},{},{}", h.0, h.1, h.2)];
                if let Some(c) = victim {
                    args.push(s(color_name(c)));
                }
                (s("robber"), args)
            }
            ActionKind::DiscardResource(r) => (s("discard"), vec![s(resource_name(r))]),
            ActionKind::MaritimeTrade { give, rate, get } => {
                (s("bankTrade"), vec![s(resource_name(give)), rate.to_string(), s(resource_name(get))])
            }
            ref other => return Err(err(format!("no OpenHex move for {other:?}"))),
        })
    }
}

#[pymethods]
impl PyMirror {
    /// `tiles`: (hex, terrain, number or None); `ports`: (vertex, vertex,
    /// resource or None for 3:1); `turn_order`: colors, first player first.
    /// `colonist`: Colonist's rules where they differ from OpenHex's (any
    /// one dev card before the roll; the robber shields every player under
    /// 3 visible points and never robs one), as the colonizing-pig engine
    /// plays them.
    #[new]
    #[pyo3(signature = (tiles, ports, turn_order, vps_to_win=15, discard_limit=9, friendly_robber=true, openhex_dice=true, seed=0, colonist=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        tiles: Vec<((i8, i8, i8), String, Option<u8>)>,
        ports: Vec<((i8, i8, i8), (i8, i8, i8), Option<String>)>,
        turn_order: Vec<String>,
        vps_to_win: u32,
        discard_limit: u32,
        friendly_robber: bool,
        openhex_dice: bool,
        seed: u64,
        colonist: bool,
    ) -> PyResult<Self> {
        let (map, geo) = openhex_map(&tiles, &ports)?;
        let colors: Vec<Color> = turn_order.iter().map(|c| color(c)).collect::<PyResult<_>>()?;
        let config = GameConfig {
            seed: Some(seed),
            discard_limit,
            friendly_robber: match (friendly_robber, colonist) {
                (false, _) => FriendlyRobber::Off,
                (true, false) => FriendlyRobber::PublicPoints,
                (true, true) => FriendlyRobber::Colonist,
            },
            dice: if openhex_dice { DiceMode::OpenHex } else { DiceMode::Fair },
            vps_to_win,
            catan_map: Some(map),
            record_history: false,
            fixed_seats: true,
            turns_limit: None,
            ..Default::default()
        };
        let mut game = Game::new(&colors, &config);
        // OpenHex only takes a knight before the roll; Colonist any one card.
        game.state.knight_only_before_roll = !colonist;
        game.playable_actions = catanatron::actions::generate_playable_actions(&game.state);
        Ok(PyMirror { game, geo, rng: Rng::new(seed ^ 0xA11CE) })
    }

    /// (current color, prompt) in our engine's terms, e.g. ("red", "PlayTurn").
    /// The mirror's legal moves for the player to act (OpenHex terms).
    fn legal_moves(&self) -> PyResult<Vec<Move>> {
        self.game.playable_actions.iter().map(|a| self.to_move(a)).collect()
    }

    fn prompt(&self) -> (String, String) {
        (color_name(self.game.state.current_color()).into(), format!("{:?}", self.game.state.current_prompt))
    }

    fn is_over(&self) -> bool {
        self.game.is_over()
    }

    /// Debugging: runs a plain MCTS (`sims` simulations, network at
    /// `model`) for the player to move and returns per root action (move,
    /// prior, visits, mean value or None), plus the root value.
    fn search_stats(&mut self, model: &str, sims: u32, seed: u64) -> PyResult<(Vec<(Move, f32, u32, Option<f32>)>, f32)> {
        use catanatron::mcts::{load_net, MctsConfig, MctsPlayer};
        let net = load_net(model).map_err(err)?;
        let c = self.game.state.current_color();
        let mut m = MctsPlayer::new(c, net, MctsConfig { simulations: sims, ..Default::default() }, seed);
        m.search(&self.game).map_err(err)?;
        let (stats, root) = m.root_stats();
        let actions = self.game.playable_actions.clone();
        let mut out = Vec::new();
        for (a, (p, n, q)) in actions.iter().zip(stats) {
            out.push((self.to_move(a)?, p, n, q));
        }
        Ok((out, root))
    }

    fn settlement(&mut self, who: &str, v: (i8, i8, i8)) -> PyResult<()> {
        let n = self.node(v)?;
        self.play(color(who)?, ActionKind::BuildSettlement(n), None)
    }

    fn city(&mut self, who: &str, v: (i8, i8, i8)) -> PyResult<()> {
        let n = self.node(v)?;
        self.play(color(who)?, ActionKind::BuildCity(n), None)
    }

    fn road(&mut self, who: &str, a: (i8, i8, i8), b: (i8, i8, i8)) -> PyResult<()> {
        let e = self.edge(a, b)?;
        self.play(color(who)?, ActionKind::BuildRoad(e), None)
    }

    fn roll(&mut self, who: &str, die1: u8, die2: u8) -> PyResult<()> {
        self.play(color(who)?, ActionKind::Roll, Some(ActionResult::Dice(die1, die2)))
    }

    fn end_turn(&mut self, who: &str) -> PyResult<()> {
        self.play(color(who)?, ActionKind::EndTurn, None)
    }

    /// A dev card purchase; `card` when we saw it (our own). An unseen
    /// card is dealt as a non-victory-point card when the deck has one, so
    /// the mirror's points stay the visible ones (search re-deals hidden
    /// cards, victory points included, on every determinization).
    #[pyo3(signature = (who, card=None))]
    fn buy(&mut self, who: &str, card: Option<&str>) -> PyResult<()> {
        let buyer = color(who)?;
        let card = match card {
            Some(name) => {
                let d = dev_card(name)?;
                self.make_drawable(buyer, d)?;
                Some(d)
            }
            None => self.game.state.dev_deck.iter().copied().find(|&d| d != DevCard::VictoryPoint),
        };
        self.play(color(who)?, ActionKind::BuyDevelopmentCard, card.map(ActionResult::DevCard))
    }

    /// Makes sure `who` holds a playable `card` before it is played: the
    /// mirror deals the opponent's hidden cards arbitrarily, so it swaps one
    /// of them (or a card from the deck) for the one actually played.
    fn ensure_dev(&mut self, who: &str, card: &str) -> PyResult<()> {
        let (c, d) = (color(who)?, dev_card(card)?);
        let s = &mut self.game.state;
        let p = s.player(c);
        if p.dev_cards[d.idx()] == 0 {
            let other = DEV_CARDS
                .iter()
                .copied()
                .filter(|&o| o != d && s.player(c).dev_cards[o.idx()] > 0)
                .max_by_key(|o| s.player(c).dev_cards[o.idx()])
                .ok_or_else(|| err(format!("{who} plays {card} but holds no hidden dev card")))?;
            let pos = s.dev_deck.iter().position(|&x| x == d).ok_or_else(|| err(format!("no {card} left to deal to {who}")))?;
            s.dev_deck[pos] = other;
            let p = s.player_mut(c);
            p.dev_cards[other.idx()] -= 1;
            p.dev_cards[d.idx()] += 1;
        }
        // The server allowed the play, so the card was held before this turn.
        let p = s.player_mut(c);
        p.dev_owned_at_start[d.idx()] = true;
        p.actual_victory_points = p.victory_points + p.dev_cards[DevCard::VictoryPoint.idx()];
        self.game.playable_actions = catanatron::actions::generate_playable_actions(&self.game.state);
        Ok(())
    }

    fn knight(&mut self, who: &str) -> PyResult<()> {
        self.play(color(who)?, ActionKind::PlayKnightCard, None)
    }

    fn road_building(&mut self, who: &str) -> PyResult<()> {
        self.play(color(who)?, ActionKind::PlayRoadBuilding, None)
    }

    fn monopoly(&mut self, who: &str, r: &str) -> PyResult<()> {
        let r = resource(r)?;
        self.play(color(who)?, ActionKind::PlayMonopoly(r), None)
    }

    /// Year of plenty for one or two resources (either order).
    #[pyo3(signature = (who, a, b=None))]
    fn year_of_plenty(&mut self, who: &str, a: &str, b: Option<&str>) -> PyResult<()> {
        let c = color(who)?;
        let (a, b) = (resource(a)?, b.map(resource).transpose()?);
        let found = self.game.playable_actions.iter().find(|x| {
            x.color == c
                && match x.kind {
                    ActionKind::PlayYearOfPlenty(p, q) => (p == a && q == b) || (Some(p) == b && q == Some(a)),
                    _ => false,
                }
        });
        let kind = found.ok_or_else(|| err("year of plenty not playable"))?.kind;
        self.play(c, kind, None)
    }

    /// Robber to `hex` (OpenHex frame); `victim` and the `stolen` resource
    /// when a card was taken.
    #[pyo3(signature = (who, hex, victim=None, stolen=None))]
    fn robber(&mut self, who: &str, hex: (i8, i8, i8), victim: Option<&str>, stolen: Option<&str>) -> PyResult<()> {
        let victim = victim.map(color).transpose()?;
        let result = match (victim, stolen) {
            (Some(_), Some(r)) => Some(ActionResult::Stolen(Some(resource(r)?))),
            (Some(_), None) => return Err(err("a steal needs the stolen resource (1v1: both seats see it)")),
            (None, _) => None,
        };
        self.play(color(who)?, ActionKind::MoveRobber { coordinate: turn_hex(hex), victim }, result)
    }

    fn discard(&mut self, who: &str, r: &str) -> PyResult<()> {
        let r = resource(r)?;
        self.play(color(who)?, ActionKind::DiscardResource(r), None)
    }

    fn bank_trade(&mut self, who: &str, give: &str, rate: u8, get: &str) -> PyResult<()> {
        let (give, get) = (resource(give)?, resource(get)?);
        self.play(color(who)?, ActionKind::MaritimeTrade { give, rate, get }, None)
    }

    /// Sets hands and bank from the server's snapshot: ours as given, the
    /// opponent's as what's in neither the bank nor our hand.
    fn sync(&mut self, me: &str, hand: HashMap<String, u32>, bank: HashMap<String, u32>) -> PyResult<()> {
        let me = color(me)?;
        let mut mine = [0u32; 5];
        let mut b = [0u32; 5];
        for (k, v) in &hand {
            mine[resource(k)?.idx()] = *v;
        }
        for (k, v) in &bank {
            b[resource(k)?.idx()] = *v;
        }
        let s = &mut self.game.state;
        if s.colors.len() != 2 {
            return Err(err("sync infers the opponent's hand, which needs 1v1"));
        }
        let opp = *s.colors.iter().find(|&&c| c != me).unwrap();
        let mut theirs = [0u32; 5];
        for r in 0..5 {
            theirs[r] = BANK_TOTAL
                .checked_sub(b[r] + mine[r])
                .ok_or_else(|| err(format!("bank {} + hand {} exceed {BANK_TOTAL}", b[r], mine[r])))?;
        }
        s.player_mut(me).resources = mine;
        s.player_mut(opp).resources = theirs;
        s.bank = b;
        self.game.playable_actions = catanatron::actions::generate_playable_actions(&self.game.state);
        Ok(())
    }

    /// Public facts for consistency checks, in OpenHex terms.
    fn snapshot(&self) -> HashMap<String, String> {
        let s = &self.game.state;
        let mut out = HashMap::new();
        let h = turn_hex(s.board.robber_coordinate);
        out.insert("robber".into(), format!("{},{},{}", h.0, h.1, h.2));
        for p in s.players.iter() {
            let c = color_name(p.color);
            out.insert(format!("{c}.publicVP"), p.victory_points.to_string());
            out.insert(format!("{c}.hand"), p.resources.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
            out.insert(format!("{c}.devCards"), p.dev_cards.iter().sum::<u32>().to_string());
            out.insert(format!("{c}.knights"), p.played_dev_cards[DevCard::Knight.idx()].to_string());
            let mut settlements: Vec<String> = Vec::new();
            let mut cities: Vec<String> = Vec::new();
            for (n, b) in s.board.buildings.iter().enumerate() {
                if let Some((bc, kind)) = b {
                    if *bc == p.color {
                        let v = self.vertex(n as NodeId);
                        let txt = format!("{},{},{}", v.0, v.1, v.2);
                        match kind {
                            catanatron::board::BuildingType::City => cities.push(txt),
                            _ => settlements.push(txt),
                        }
                    }
                }
            }
            settlements.sort();
            cities.sort();
            out.insert(format!("{c}.settlements"), settlements.join(";"));
            out.insert(format!("{c}.cities"), cities.join(";"));
            let mut roads: Vec<String> = p
                .roads
                .iter()
                .map(|&e| {
                    let (a, b) = self.edge_vertices(e);
                    format!("{},{},{}|{},{},{}", a.0, a.1, a.2, b.0, b.1, b.2)
                })
                .collect();
            roads.sort();
            out.insert(format!("{c}.roads"), roads.join(";"));
        }
        out.insert("bank".into(), s.bank.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
        out.insert("devDeck".into(), s.dev_deck.len().to_string());
        out.insert("diceDeck".into(), s.dice_deck.map_or(-1, |d| d.len() as i64).to_string());
        out.insert("longestRoad".into(), s.longest_road_color().map_or("".into(), |c| color_name(c).into()));
        out.insert("largestArmy".into(), s.largest_army().map_or("".into(), |(c, _)| color_name(c).into()));
        out
    }

    /// Our move for `who` (which must be the one to act), chosen by the
    /// bot `key` (`catanatron::players::from_key`: F, NN:<net>,
    /// MCTS:<sims>[,t<ms>][,m..][,s..]:<net>, ...). A knight comes back with
    /// its robber move, and a discard as all its cards: the live mirror is
    /// only changed by the server's events.
    #[pyo3(signature = (who, key, seed=0))]
    fn decide(&mut self, who: &str, key: &str, seed: u64) -> PyResult<Vec<Move>> {
        let c = color(who)?;
        let mut player = catanatron::players::from_key(key, c, seed ^ self.rng.next_u64())
            .ok_or_else(|| err(format!("unknown bot key {key}")))?;
        let mut g = self.game.copy_without_history();
        let mut out = Vec::new();
        while !g.is_over() {
            if g.state.current_color() != c {
                // Another seat's discard comes first in our engine: any
                // cards will do on this scratch copy.
                if g.state.current_prompt == ActionPrompt::Discard {
                    let a = g.playable_actions[0];
                    g.execute(a).map_err(err)?;
                    continue;
                }
                break;
            }
            let actions = g.playable_actions.clone();
            let a = player.decide(&g, &actions);
            out.push(self.to_move(&a)?);
            // OpenHex takes a knight with its robber move, and a discard's
            // cards, as one action: keep deciding on the scratch copy.
            match a.kind {
                ActionKind::PlayKnightCard => {
                    g.execute(a).map_err(err)?;
                    if g.is_over() {
                        // The knight won (largest army): OpenHex still wants
                        // a robber spot with it, and any legal one will do.
                        let spot = catanatron::actions::robber_possibilities(&g.state, c)
                            .into_iter()
                            .next()
                            .ok_or_else(|| err("no robber spot after a winning knight"))?;
                        out.push(self.to_move(&spot)?);
                        break;
                    }
                }
                ActionKind::DiscardResource(_) => {
                    g.execute(a).map_err(err)?;
                    if !(g.state.current_prompt == ActionPrompt::Discard && g.state.current_color() == c) {
                        break;
                    }
                    continue;
                }
                _ => break,
            };
        }
        Ok(out)
    }
}
