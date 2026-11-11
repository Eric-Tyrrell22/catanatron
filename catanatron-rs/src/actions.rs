//! Move generation: the actions the current player may take.

use crate::board::BuildingType;
use crate::state::{FriendlyRobber, State};
use crate::types::*;

pub fn generate_playable_actions(s: &State) -> Vec<Action> {
    let mut out = Vec::with_capacity(32);
    generate_playable_actions_into(s, &mut out);
    out
}

/// Like `generate_playable_actions`, but reuses `out`'s allocation.
pub fn generate_playable_actions_into(s: &State, out: &mut Vec<Action>) {
    out.clear();
    let color = s.current_color();
    match s.current_prompt {
        ActionPrompt::BuildInitialSettlement => out.extend(settlement_possibilities(s, color, true)),
        ActionPrompt::BuildInitialRoad => out.extend(initial_road_possibilities(s, color)),
        ActionPrompt::MoveRobber => out.extend(robber_possibilities(s, color)),
        ActionPrompt::Discard => {
            if s.discard_counts[s.seat(color)] > 0 {
                let p = s.player(color);
                for r in RESOURCES {
                    if p.resources[r.idx()] > 0 {
                        out.push(Action::new(color, ActionKind::DiscardResource(r)));
                    }
                }
            }
        }
        ActionPrompt::PlayTurn => play_turn_possibilities(s, color, out),
        ActionPrompt::DecideTrade => {
            out.push(Action::new(color, ActionKind::RejectTrade));
            if let Some((_, asking, _)) = s.current_trade {
                if fd_contains(&s.player(color).resources, &asking) {
                    out.push(Action::new(color, ActionKind::AcceptTrade));
                }
            }
        }
        ActionPrompt::DecideAcceptees => {
            out.push(Action::new(color, ActionKind::CancelTrade));
            for (i, &accepted) in s.acceptees.iter().enumerate() {
                if accepted {
                    out.push(Action::new(color, ActionKind::ConfirmTrade { partner: s.colors[i] }));
                }
            }
        }
    }
}

fn push_buildable_roads(s: &State, color: Color, out: &mut Vec<Action>) {
    let b = &s.board;
    for i in bits_iter(b.buildable_edge_bits(color)) {
        out.push(Action::new(color, ActionKind::BuildRoad(b.map.edges[i])));
    }
}

fn play_turn_possibilities(s: &State, color: Color, actions: &mut Vec<Action>) {
    let p = s.player(color);
    if s.is_road_building {
        if p.roads_available > 0 {
            push_buildable_roads(s, color, actions);
        }
        return;
    }
    // Dev cards may be played before or after rolling (before it, only a
    // knight with `knight_only_before_roll`).
    let any_card = p.has_rolled || !s.knight_only_before_roll;
    if any_card && p.can_play_dev(DevCard::YearOfPlenty) {
        actions.extend(year_of_plenty_possibilities(color, &s.bank));
    }
    if any_card && p.can_play_dev(DevCard::Monopoly) {
        actions.extend(RESOURCES.iter().map(|&r| Action::new(color, ActionKind::PlayMonopoly(r))));
    }
    if p.can_play_dev(DevCard::Knight) {
        actions.push(Action::new(color, ActionKind::PlayKnightCard));
    }
    if any_card && p.can_play_dev(DevCard::RoadBuilding) && p.roads_available > 0 && s.board.has_buildable_edge(color) {
        actions.push(Action::new(color, ActionKind::PlayRoadBuilding));
    }
    if !p.has_rolled {
        actions.push(Action::new(color, ActionKind::Roll));
        return;
    }
    actions.push(Action::new(color, ActionKind::EndTurn));
    if p.roads_available > 0 && fd_contains(&p.resources, &ROAD_COST) {
        push_buildable_roads(s, color, actions);
    }
    if p.settlements_available > 0 && fd_contains(&p.resources, &SETTLEMENT_COST) {
        let b = &s.board;
        for n in bits_iter(b.network_bits(color)) {
            if b.buildable[n] {
                actions.push(Action::new(color, ActionKind::BuildSettlement(n as NodeId)));
            }
        }
    }
    if p.cities_available > 0 && fd_contains(&p.resources, &CITY_COST) {
        actions.extend(p.settlements.iter().map(|&n| Action::new(color, ActionKind::BuildCity(n))));
    }
    if fd_contains(&p.resources, &DEV_CARD_COST) && !s.dev_deck.is_empty() {
        actions.push(Action::new(color, ActionKind::BuyDevelopmentCard));
    }
    push_maritime_offers(&p.resources, &s.bank, s.board.port_mask(color), |give, rate, get| {
        actions.push(Action::new(color, ActionKind::MaritimeTrade { give, rate, get }))
    });
}

pub fn year_of_plenty_possibilities(color: Color, bank: &Freqdeck) -> Vec<Action> {
    let mut options: Vec<(Resource, Option<Resource>)> = Vec::new();
    let mut add = |o| {
        if !options.contains(&o) {
            options.push(o);
        }
    };
    for i in 0..5 {
        for j in i..5 {
            let (a, b) = (RESOURCES[i], RESOURCES[j]);
            let mut want = fd_single(a, 1);
            want[b.idx()] += 1;
            if fd_contains(bank, &want) {
                add((a, Some(b)));
            } else {
                if bank[a.idx()] >= 1 {
                    add((a, None));
                }
                if bank[b.idx()] >= 1 {
                    add((b, None));
                }
            }
        }
    }
    options
        .into_iter()
        .map(|(a, b)| Action::new(color, ActionKind::PlayYearOfPlenty(a, b)))
        .collect()
}

pub fn road_building_possibilities(s: &State, color: Color, check_money: bool) -> Vec<Action> {
    let p = s.player(color);
    if p.roads_available == 0 || (check_money && !fd_contains(&p.resources, &ROAD_COST)) {
        return Vec::new();
    }
    s.board
        .buildable_edges(color)
        .into_iter()
        .map(|e| Action::new(color, ActionKind::BuildRoad(e)))
        .collect()
}

pub fn settlement_possibilities(s: &State, color: Color, initial_build_phase: bool) -> Vec<Action> {
    if !initial_build_phase {
        let p = s.player(color);
        if !fd_contains(&p.resources, &SETTLEMENT_COST) || p.settlements_available == 0 {
            return Vec::new();
        }
    }
    s.board
        .buildable_node_ids(color, initial_build_phase)
        .into_iter()
        .map(|n| Action::new(color, ActionKind::BuildSettlement(n)))
        .collect()
}

pub fn city_possibilities(s: &State, color: Color) -> Vec<Action> {
    let p = s.player(color);
    if !fd_contains(&p.resources, &CITY_COST) || p.cities_available == 0 {
        return Vec::new();
    }
    p.settlements
        .iter()
        .map(|&n| Action::new(color, ActionKind::BuildCity(n)))
        .collect()
}

pub fn robber_possibilities(s: &State, color: Color) -> Vec<Action> {
    let actions = robber_possibilities_unfiltered(s, color);
    if s.friendly_robber == FriendlyRobber::Off {
        return actions;
    }
    let filtered: Vec<Action> = actions
        .iter()
        .copied()
        .filter(|a| !robber_blocks_low_vp_enemy(s, color, a))
        .collect();
    if filtered.is_empty() {
        actions
    } else {
        filtered
    }
}

fn robber_possibilities_unfiltered(s: &State, color: Color) -> Vec<Action> {
    let mut actions = Vec::new();
    let map = &s.board.map;
    // Seats that can be stolen from at all.
    let mut stealable = [false; 4];
    for (seat, p) in s.players.iter().enumerate() {
        stealable[seat] = p.color != color
            && p.num_resources() >= 1
            && !(s.friendly_robber == FriendlyRobber::Colonist && p.victory_points < 3);
    }
    for &ti in &map.land_tiles {
        let tile = &map.tiles[ti];
        if tile.coordinate == s.board.robber_coordinate {
            continue; // must move the robber
        }
        let mut victims = [false; 4]; // by seat
        for (seat, p) in s.players.iter().enumerate() {
            victims[seat] = stealable[seat]
                && bits_any(&bits_and(s.board.owned_bits(p.color), &map.tile_node_bits[ti]));
        }
        if !victims.iter().any(|&v| v) {
            actions.push(Action::new(
                color,
                ActionKind::MoveRobber { coordinate: tile.coordinate, victim: None },
            ));
        } else {
            for (seat, &v) in victims.iter().enumerate() {
                if v {
                    actions.push(Action::new(
                        color,
                        ActionKind::MoveRobber { coordinate: tile.coordinate, victim: Some(s.colors[seat]) },
                    ));
                }
            }
        }
    }
    actions
}

fn robber_blocks_low_vp_enemy(s: &State, color: Color, action: &Action) -> bool {
    let ActionKind::MoveRobber { coordinate, .. } = action.kind else {
        return false;
    };
    let tile = s.board.map.tile_at(coordinate).expect("robber tile exists");
    let points = |c: Color| match s.friendly_robber {
        FriendlyRobber::PublicPoints | FriendlyRobber::Colonist => s.player(c).victory_points,
        _ => s.player(c).actual_victory_points,
    };
    let shields_mover = s.friendly_robber == FriendlyRobber::Colonist;
    tile.nodes.iter().any(|&n| match s.board.buildings[n as usize] {
        Some((c, _)) => (c != color || shields_mover) && points(c) < 3,
        None => false,
    })
}

pub fn initial_road_possibilities(s: &State, color: Color) -> Vec<Action> {
    let Some(&last) = s.player(color).settlements.last() else {
        return Vec::new();
    };
    s.board
        .buildable_edges(color)
        .into_iter()
        .filter(|&(a, b)| a == last || b == last)
        .map(|e| Action::new(color, ActionKind::BuildRoad(e)))
        .collect()
}

pub fn discard_possibilities(s: &State, color: Color) -> Vec<Action> {
    if s.discard_counts[s.seat(color)] == 0 {
        return Vec::new();
    }
    let p = s.player(color);
    RESOURCES
        .iter()
        .filter(|r| p.resources[r.idx()] > 0)
        .map(|&r| Action::new(color, ActionKind::DiscardResource(r)))
        .collect()
}

/// Bank/port trades for a hand, given port access. Ordered by (give, get).
pub fn maritime_trade_offers(
    hand: &Freqdeck,
    bank: &Freqdeck,
    port_resources: &[Option<Resource>],
) -> Vec<(Resource, u8, Resource)> {
    let mut mask = 0u8;
    for r in port_resources {
        mask |= 1 << r.map_or(5, |r| r.idx());
    }
    let mut offers = Vec::new();
    push_maritime_offers(hand, bank, mask, |give, rate, get| offers.push((give, rate, get)));
    offers
}

fn push_maritime_offers(hand: &Freqdeck, bank: &Freqdeck, port_mask: u8, mut f: impl FnMut(Resource, u8, Resource)) {
    let base = if port_mask >> 5 & 1 == 1 { 3 } else { 4 };
    for give in RESOURCES {
        let rate = if port_mask >> give.idx() & 1 == 1 { 2 } else { base };
        if hand[give.idx()] >= rate as u32 {
            for get in RESOURCES {
                if get != give && bank[get.idx()] > 0 {
                    f(give, rate, get);
                }
            }
        }
    }
}

pub fn maritime_trade_possibilities(s: &State, color: Color) -> Vec<Action> {
    let mut out = Vec::new();
    push_maritime_offers(&s.player(color).resources, &s.bank, s.board.port_mask(color), |give, rate, get| {
        out.push(Action::new(color, ActionKind::MaritimeTrade { give, rate, get }))
    });
    out
}

/// Whether an OFFER_TRADE value is well-formed: gives and asks something,
/// and doesn't trade a resource for itself.
pub fn is_valid_trade(offering: &Freqdeck, asking: &Freqdeck) -> bool {
    offering.iter().sum::<u32>() > 0
        && asking.iter().sum::<u32>() > 0
        && offering.iter().zip(asking).all(|(o, a)| *o == 0 || *a == 0)
}

/// Settlement and city counts, mostly for tests and display.
pub fn count_buildings(s: &State, color: Color, kind: BuildingType) -> usize {
    s.board.buildings.iter().filter(|b| **b == Some((color, kind))).count()
}
