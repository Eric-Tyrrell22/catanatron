use catanatron::map::{BoardMix, BoardStyle, CatanMap};
use catanatron::players::{from_key, Player};
use catanatron::*;

/// Resource tiles per resource, deserts, and number tokens.
fn counts(m: &CatanMap) -> ([usize; 5], usize, Vec<u8>) {
    let mut res = [0; 5];
    let mut desert = 0;
    let mut numbers = Vec::new();
    for &i in &m.land_tiles {
        match m.tiles[i].resource() {
            Some(r) => res[r.idx()] += 1,
            None => desert += 1,
        }
        numbers.extend(m.tiles[i].number());
    }
    numbers.sort();
    (res, desert, numbers)
}

fn rules(mix: BoardMix, seed: u64) -> GameConfig {
    GameConfig {
        seed: Some(seed),
        number_placement: map::NumberPlacement::SpiralRandomCorner,
        vps_to_win: 15,
        turns_limit: None,
        board_mix: mix,
        ..Default::default()
    }
}

#[test]
fn selected_boards_are_real_boards_with_the_property() {
    let standard = counts(&Game::new(&COLORS[..2], &rules(BoardMix::default(), 0)).state.board.map);
    for (mix, style) in [
        (BoardMix { scarce_ows: 1.0, port_start: 0.0 }, BoardStyle::ScarceOws),
        (BoardMix { scarce_ows: 0.0, port_start: 1.0 }, BoardStyle::PortStart),
    ] {
        for seed in 0..200 {
            let m = &Game::new(&COLORS[..2], &rules(mix, seed)).state.board.map;
            assert_eq!(counts(m), standard, "standard tiles and number tokens");
            assert!(style.matches(m), "seed {seed}: {style:?}");
        }
    }
}

#[test]
fn games_finish_on_selected_boards_and_no_mix_changes_nothing() {
    for seed in 0..10 {
        // An empty mix: the standard board of that seed, exactly.
        let a = Game::new(&COLORS[..2], &rules(BoardMix::default(), seed));
        let b = Game::new(&COLORS[..2], &rules(BoardMix { scarce_ows: 0.0, port_start: 0.0 }, seed));
        assert_eq!(a.state.board.map.layout_spec(), b.state.board.map.layout_spec());
        for mix in [BoardMix { scarce_ows: 1.0, port_start: 0.0 }, BoardMix { scarce_ows: 0.0, port_start: 1.0 }] {
            let mut g = Game::new(&COLORS[..2], &rules(mix, seed));
            let mut bots: Vec<Box<dyn Player>> = (0..2).map(|i| from_key("VP", COLORS[i], seed + i as u64).unwrap()).collect();
            g.play(&mut bots).unwrap();
            assert!(g.winning_color().is_some());
        }
    }
}
