//! Applies an action to a state. Responsible for keeping the turn/prompt
//! bookkeeping (`current_player_index`, `current_prompt`, ...) consistent.

use crate::board::BuildingType;
use crate::fixed::FixedVec;
use crate::state::State;
use crate::types::*;

/// Applies `action`. If `replay` is given, its result is used instead of
/// rolling dice / drawing randomly, so recorded games can be replayed.
pub fn apply_action(
    s: &mut State,
    action: Action,
    replay: Option<ActionResult>,
) -> Result<ActionRecord, String> {
    let color = action.color;
    let result = match action.kind {
        ActionKind::EndTurn => {
            clean_turn(s, color);
            advance_turn(s, 1);
            s.current_prompt = ActionPrompt::PlayTurn;
            ActionResult::None
        }
        ActionKind::BuildSettlement(n) => {
            build_settlement(s, color, n)?;
            ActionResult::None
        }
        ActionKind::BuildRoad(e) => {
            build_road(s, color, e)?;
            ActionResult::None
        }
        ActionKind::BuildCity(n) => {
            build_city(s, color, n)?;
            ActionResult::None
        }
        ActionKind::BuyDevelopmentCard => buy_development_card(s, color, replay)?,
        ActionKind::Roll => roll(s, color, replay),
        ActionKind::DiscardResource(r) => discard(s, color, r)?,
        ActionKind::MoveRobber { coordinate, victim } => move_robber(s, color, coordinate, victim, replay)?,
        ActionKind::PlayKnightCard => {
            play_dev_card(s, color, DevCard::Knight)?;
            s.current_prompt = ActionPrompt::MoveRobber;
            ActionResult::None
        }
        ActionKind::PlayYearOfPlenty(a, b) => {
            let mut cards = fd_single(a, 1);
            if let Some(b) = b {
                cards[b.idx()] += 1;
            }
            if !s.player(color).can_play_dev(DevCard::YearOfPlenty) {
                return Err("Player cant play year of plenty now".into());
            }
            if !fd_contains(&s.bank, &cards) {
                return Err("Not enough resources of this type in bank".into());
            }
            fd_add(&mut s.player_mut(color).resources, &cards);
            fd_sub(&mut s.bank, &cards);
            play_dev_card(s, color, DevCard::YearOfPlenty)?;
            s.current_prompt = ActionPrompt::PlayTurn;
            ActionResult::None
        }
        ActionKind::PlayMonopoly(r) => {
            if !s.player(color).can_play_dev(DevCard::Monopoly) {
                return Err("Player cant play monopoly now".into());
            }
            let mut stolen = 0;
            for p in s.players.iter_mut().filter(|p| p.color != color) {
                stolen += p.resources[r.idx()];
                p.resources[r.idx()] = 0;
            }
            s.player_mut(color).resources[r.idx()] += stolen;
            play_dev_card(s, color, DevCard::Monopoly)?;
            s.current_prompt = ActionPrompt::PlayTurn;
            ActionResult::None
        }
        ActionKind::PlayRoadBuilding => {
            play_dev_card(s, color, DevCard::RoadBuilding)?;
            s.is_road_building = true;
            s.free_roads_available = 2;
            s.current_prompt = ActionPrompt::PlayTurn;
            ActionResult::None
        }
        ActionKind::MaritimeTrade { give, rate, get } => {
            let offering = fd_single(give, rate as u32);
            let asking = fd_single(get, 1);
            if !fd_contains(&s.player(color).resources, &offering) {
                return Err("Trying to trade without money".into());
            }
            if !fd_contains(&s.bank, &asking) {
                return Err("Bank doesnt have those cards".into());
            }
            let p = s.player_mut(color);
            fd_sub(&mut p.resources, &offering);
            fd_add(&mut p.resources, &asking);
            fd_add(&mut s.bank, &offering);
            fd_sub(&mut s.bank, &asking);
            s.current_prompt = ActionPrompt::PlayTurn;
            ActionResult::None
        }
        ActionKind::OfferTrade { offering, asking } => {
            s.is_resolving_trade = true;
            s.current_trade = Some((offering, asking, s.current_turn_index));
            // Ask in seating order, never yourself.
            s.current_player_index = s.colors.iter().position(|&c| c != color).ok_or("no one to trade with")?;
            s.current_prompt = ActionPrompt::DecideTrade;
            ActionResult::None
        }
        ActionKind::AcceptTrade => {
            let seat = s.seat(color);
            s.acceptees[seat] = true;
            match next_trade_responder(s) {
                Some(i) => s.current_player_index = i,
                None => {
                    s.current_player_index = s.current_turn_index;
                    s.current_prompt = ActionPrompt::DecideAcceptees;
                }
            }
            ActionResult::None
        }
        ActionKind::RejectTrade => {
            match next_trade_responder(s) {
                Some(i) => s.current_player_index = i,
                None => {
                    if s.acceptees.iter().any(|&a| a) {
                        s.current_prompt = ActionPrompt::DecideAcceptees;
                    } else {
                        reset_trading_state(s);
                        s.current_prompt = ActionPrompt::PlayTurn;
                    }
                    s.current_player_index = s.current_turn_index;
                }
            }
            ActionResult::None
        }
        ActionKind::ConfirmTrade { partner } => {
            let (offering, asking, _) = s.current_trade.ok_or("no trade to confirm")?;
            if !fd_contains(&s.player(color).resources, &offering)
                || !fd_contains(&s.player(partner).resources, &asking)
            {
                return Err("Trade parties no longer hold the cards".into());
            }
            let me = s.player_mut(color);
            fd_sub(&mut me.resources, &offering);
            fd_add(&mut me.resources, &asking);
            let them = s.player_mut(partner);
            fd_sub(&mut them.resources, &asking);
            fd_add(&mut them.resources, &offering);
            reset_trading_state(s);
            s.current_player_index = s.current_turn_index;
            s.current_prompt = ActionPrompt::PlayTurn;
            ActionResult::None
        }
        ActionKind::CancelTrade => {
            reset_trading_state(s);
            s.current_player_index = s.current_turn_index;
            s.current_prompt = ActionPrompt::PlayTurn;
            ActionResult::None
        }
    };
    let record = ActionRecord { action, result };
    s.num_actions += 1;
    Ok(record)
}

fn build_settlement(s: &mut State, color: Color, n: NodeId) -> Result<(), String> {
    if s.is_initial_build_phase {
        s.board.build_settlement(color, n, true)?;
        let p = s.player_mut(color);
        p.settlements.push(n);
        p.settlements_available -= 1;
        p.victory_points += 1;
        p.actual_victory_points += 1;

        // Second settlement yields its adjacent resources.
        if p.settlements.len() == 2 {
            let map = s.board.map.clone();
            for &t in &map.adjacent_tiles[n as usize] {
                if let Some(r) = map.tiles[t].resource() {
                    if s.bank[r.idx()] > 0 {
                        s.bank[r.idx()] -= 1;
                        s.player_mut(color).resources[r.idx()] += 1;
                    }
                }
            }
        }
        s.current_prompt = ActionPrompt::BuildInitialRoad;
    } else {
        if !fd_contains(&s.player(color).resources, &SETTLEMENT_COST) {
            return Err("Cannot afford settlement".into());
        }
        if s.player(color).settlements_available == 0 {
            return Err("No settlements available".into());
        }
        let previous = s.board.build_settlement(color, n, false)?;
        let p = s.player_mut(color);
        p.settlements.push(n);
        p.settlements_available -= 1;
        p.victory_points += 1;
        p.actual_victory_points += 1;
        fd_sub(&mut p.resources, &SETTLEMENT_COST);
        fd_add(&mut s.bank, &SETTLEMENT_COST);
        maintain_longest_road(s, previous);
    }
    Ok(())
}

fn build_road(s: &mut State, color: Color, e: EdgeId) -> Result<(), String> {
    if s.player(color).roads_available == 0 {
        return Err("No roads available".into());
    }
    if s.is_initial_build_phase {
        let last = *s.player(color).settlements.last().ok_or("place a settlement first")?;
        if e.0 != last && e.1 != last {
            return Err("Initial road must touch the last settlement".into());
        }
        s.board.build_road(color, e)?;
        let p = s.player_mut(color);
        p.roads.push(edge(e.0, e.1));
        p.roads_available -= 1;

        let num_buildings: usize = s.players.iter().map(|p| p.settlements.len()).sum();
        let num_players = s.players.len();
        if num_buildings < num_players {
            advance_turn(s, 1);
            s.current_prompt = ActionPrompt::BuildInitialSettlement;
        } else if num_buildings == num_players {
            // Last player places again (snake order).
            s.current_prompt = ActionPrompt::BuildInitialSettlement;
        } else if num_buildings == 2 * num_players {
            s.is_initial_build_phase = false;
            s.current_prompt = ActionPrompt::PlayTurn;
        } else {
            advance_turn(s, -1);
            s.current_prompt = ActionPrompt::BuildInitialSettlement;
        }
        return Ok(());
    }

    let free = s.is_road_building && s.free_roads_available > 0;
    if !free && !fd_contains(&s.player(color).resources, &ROAD_COST) {
        return Err("Cannot afford road".into());
    }
    let previous = s.board.build_road(color, e)?;
    let p = s.player_mut(color);
    p.roads.push(edge(e.0, e.1));
    p.roads_available -= 1;
    if !free {
        fd_sub(&mut p.resources, &ROAD_COST);
        fd_add(&mut s.bank, &ROAD_COST);
    }
    maintain_longest_road(s, previous);

    if free {
        s.free_roads_available -= 1;
        if s.free_roads_available == 0
            || s.player(color).roads_available == 0
            || !s.board.has_buildable_edge(color)
        {
            s.is_road_building = false;
            s.free_roads_available = 0;
        }
    }
    Ok(())
}

fn build_city(s: &mut State, color: Color, n: NodeId) -> Result<(), String> {
    if !fd_contains(&s.player(color).resources, &CITY_COST) {
        return Err("Cannot afford city".into());
    }
    if s.player(color).cities_available == 0 {
        return Err("No cities available".into());
    }
    s.board.build_city(color, n)?;
    let p = s.player_mut(color);
    p.settlements.retain(|&x| x != n);
    p.cities.push(n);
    p.settlements_available += 1;
    p.cities_available -= 1;
    p.victory_points += 1;
    p.actual_victory_points += 1;
    fd_sub(&mut p.resources, &CITY_COST);
    fd_add(&mut s.bank, &CITY_COST);
    Ok(())
}

fn buy_development_card(s: &mut State, color: Color, replay: Option<ActionResult>) -> Result<ActionResult, String> {
    if s.dev_deck.is_empty() {
        return Err("No more development cards".into());
    }
    if !fd_contains(&s.player(color).resources, &DEV_CARD_COST) {
        return Err("No money to buy development card".into());
    }
    let card = match replay {
        Some(ActionResult::DevCard(card)) if s.dev_deck.last() != Some(&card) => {
            // Replaying onto a deck in a different order: draw that card.
            let i = s.dev_deck.iter().position(|&c| c == card).ok_or("card not in deck")?;
            s.dev_deck.remove(i)
        }
        _ => s.dev_deck.pop().unwrap(),
    };
    let turn = s.num_turns;
    let p = s.player_mut(color);
    p.dev_cards[card.idx()] += 1;
    p.dev_bought_this_turn += 1;
    p.last_dev_buy_turn = Some(turn);
    if card == DevCard::VictoryPoint {
        p.actual_victory_points += 1;
    }
    fd_sub(&mut p.resources, &DEV_CARD_COST);
    fd_add(&mut s.bank, &DEV_CARD_COST);
    Ok(ActionResult::DevCard(card))
}

fn roll(s: &mut State, color: Color, replay: Option<ActionResult>) -> ActionResult {
    s.player_mut(color).has_rolled = true;
    let (seat, num_players) = (s.seat(color), s.colors.len());
    let (a, b) = match (replay, s.dice_deck.as_mut()) {
        (Some(ActionResult::Dice(a, b)), deck) => {
            if let Some(d) = deck {
                d.remove(a, b, seat);
            }
            (a, b)
        }
        (_, Some(d)) => d.draw(&mut s.rng, seat, num_players),
        (_, None) => (s.rng.range_inclusive(1, 6) as u8, s.rng.range_inclusive(1, 6) as u8),
    };
    let number = a + b;

    if number == 7 {
        let limit = s.discard_limit;
        s.discard_counts = s
            .players
            .iter()
            .map(|p| {
                let n = p.num_resources();
                if n > limit { n / 2 } else { 0 }
            })
            .collect();
        match s.discard_counts.iter().position(|&c| c > 0) {
            Some(first) => {
                s.current_player_index = first;
                s.current_prompt = ActionPrompt::Discard;
                s.is_discarding = true;
            }
            None => {
                s.current_prompt = ActionPrompt::MoveRobber;
                s.is_moving_knight = true;
            }
        }
    } else {
        let payout = yield_resources(s, number);
        for (seat, deck) in payout.iter().enumerate() {
            fd_add(&mut s.players[seat].resources, deck);
            fd_sub(&mut s.bank, deck);
        }
        s.current_prompt = ActionPrompt::PlayTurn;
    }
    ActionResult::Dice(a, b)
}

/// Seat-indexed payout for a roll. A resource the bank can't fully pay
/// out is paid to no one.
pub fn yield_resources(s: &State, number: u8) -> Vec<Freqdeck> {
    let mut payout = vec![[0u32; 5]; s.players.len()];
    let mut totals = [0u32; 5];
    let map = &s.board.map;
    for &ti in &map.tiles_by_number[number as usize] {
        let tile = &map.tiles[ti];
        if tile.coordinate == s.board.robber_coordinate {
            continue;
        }
        let r = tile.resource().expect("numbered tile has a resource");
        for &n in &tile.nodes {
            if let Some((c, kind)) = s.board.buildings[n as usize] {
                let amount = if kind == BuildingType::City { 2 } else { 1 };
                payout[s.seat(c)][r.idx()] += amount;
                totals[r.idx()] += amount;
            }
        }
    }
    for r in 0..5 {
        if totals[r] > s.bank[r] {
            for deck in payout.iter_mut() {
                deck[r] = 0;
            }
        }
    }
    payout
}

fn discard(s: &mut State, color: Color, r: Resource) -> Result<ActionResult, String> {
    let seat = s.seat(color);
    if s.discard_counts[seat] == 0 {
        return Err("Trying to discard when not required".into());
    }
    if s.players[seat].resources[r.idx()] == 0 {
        return Err("Trying to discard a card not in hand".into());
    }
    s.players[seat].resources[r.idx()] -= 1;
    s.bank[r.idx()] += 1;
    s.discard_counts[seat] -= 1;

    if s.discard_counts[seat] == 0 {
        let next = (s.current_player_index + 1..s.colors.len()).find(|&i| s.discard_counts[i] > 0);
        match next {
            Some(i) => s.current_player_index = i,
            None => {
                s.current_player_index = s.current_turn_index;
                s.current_prompt = ActionPrompt::MoveRobber;
                s.is_discarding = false;
                s.is_moving_knight = true;
                s.discard_counts = FixedVec::filled(0, s.colors.len());
            }
        }
    }
    Ok(ActionResult::Discarded(r))
}

fn move_robber(
    s: &mut State,
    color: Color,
    coordinate: Coordinate,
    victim: Option<Color>,
    replay: Option<ActionResult>,
) -> Result<ActionResult, String> {
    match s.board.map.tile_at(coordinate) {
        Some(t) if t.is_land() => {}
        _ => return Err("Robber must move to a land tile".into()),
    }
    let mut stolen = None;
    if let Some(v) = victim {
        let r = match replay {
            Some(ActionResult::Stolen(Some(r))) => r,
            _ => {
                let hand = s.player(v).resources;
                let total: u32 = hand.iter().sum();
                if total == 0 {
                    return Err("Victim has no cards".into());
                }
                let mut k = s.rng.below(total as usize) as u32;
                let mut chosen = Resource::Wood;
                for res in RESOURCES {
                    if k < hand[res.idx()] {
                        chosen = res;
                        break;
                    }
                    k -= hand[res.idx()];
                }
                chosen
            }
        };
        if s.player(v).resources[r.idx()] == 0 {
            return Err("Victim doesnt have that card".into());
        }
        s.player_mut(v).resources[r.idx()] -= 1;
        s.player_mut(color).resources[r.idx()] += 1;
        stolen = Some(r);
    }
    s.board.robber_coordinate = coordinate;
    s.current_prompt = ActionPrompt::PlayTurn;
    Ok(ActionResult::Stolen(stolen))
}

fn play_dev_card(s: &mut State, color: Color, card: DevCard) -> Result<(), String> {
    if !s.player(color).can_play_dev(card) {
        return Err(format!("Player cant play {} now", card.name()));
    }
    let previous_army = s.largest_army();
    let p = s.player_mut(color);
    p.dev_cards[card.idx()] -= 1;
    p.has_played_development_card_in_turn = true;
    p.dev_idle_turns = 0;
    p.played_dev_cards[card.idx()] += 1;
    if card == DevCard::Knight {
        maintain_largest_army(s, color, previous_army);
    }
    Ok(())
}

fn maintain_largest_army(s: &mut State, color: Color, previous: Option<(Color, u32)>) {
    let size = s.player(color).played_dev_cards[DevCard::Knight.idx()];
    if size < 3 {
        return;
    }
    match previous {
        None => award(s, color, true),
        Some((holder, holder_size)) if holder != color && holder_size < size => {
            award(s, color, true);
            award(s, holder, false);
        }
        _ => {}
    }
}

fn award(s: &mut State, color: Color, army_gained: bool) {
    let p = s.player_mut(color);
    p.has_army = army_gained;
    if army_gained {
        p.victory_points += 2;
        p.actual_victory_points += 2;
    } else {
        p.victory_points -= 2;
        p.actual_victory_points -= 2;
    }
}

/// Syncs per-player road lengths and moves Longest Road points if the
/// holder changed.
fn maintain_longest_road(s: &mut State, previous: Option<Color>) {
    for i in 0..s.players.len() {
        let c = s.players[i].color;
        s.players[i].longest_road_length = s.board.road_lengths[c.idx()] as u32;
    }
    let current = s.board.road_color;
    if current == previous {
        return;
    }
    if let Some(prev) = previous {
        let p = s.player_mut(prev);
        p.has_road = false;
        p.victory_points -= 2;
        p.actual_victory_points -= 2;
    }
    if let Some(cur) = current {
        let p = s.player_mut(cur);
        p.has_road = true;
        p.victory_points += 2;
        p.actual_victory_points += 2;
    }
}

fn clean_turn(s: &mut State, color: Color) {
    let p = s.player_mut(color);
    if p.num_dev_cards() > p.dev_bought_this_turn && !p.has_played_development_card_in_turn {
        p.dev_idle_turns += 1;
    }
    p.dev_bought_this_turn = 0;
    p.has_played_development_card_in_turn = false;
    p.has_rolled = false;
    // Cards held now become playable next turn.
    for card in [DevCard::Knight, DevCard::Monopoly, DevCard::YearOfPlenty, DevCard::RoadBuilding] {
        p.dev_owned_at_start[card.idx()] = p.dev_cards[card.idx()] > 0;
    }
}

fn advance_turn(s: &mut State, direction: i32) {
    let n = s.colors.len() as i32;
    let next = ((s.current_player_index as i32 + direction).rem_euclid(n)) as usize;
    s.current_player_index = next;
    s.current_turn_index = next;
    s.num_turns += 1;
}

/// Next seat after the current one that should answer the trade offer
/// (never the player who made the offer).
fn next_trade_responder(s: &State) -> Option<usize> {
    let offerer = s.current_trade.map(|(_, _, seat)| seat);
    (s.current_player_index + 1..s.colors.len()).find(|&i| Some(i) != offerer)
}

fn reset_trading_state(s: &mut State) {
    s.is_resolving_trade = false;
    s.current_trade = None;
    s.acceptees = FixedVec::filled(false, s.colors.len());
}
