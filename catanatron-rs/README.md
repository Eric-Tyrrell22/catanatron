# catanatron (Rust)

A Rust port of the catanatron core engine: map generation, board rules, move
generation, action application, and simple bots. It has no dependencies.

```sh
cargo run --release -- -n 1000 -p VP,W,R,R --seed 1   # simulate games
cargo test --release
```

Bots: `R` random, `S` first-action, `W` weighted random, `VP` greedy victory
points, `F` hand-tuned value function, `AB` / `AB:<depth>` alphabeta search
(ports of the Python `F` and `AB`).

## Encodings for learning (`encoding.rs`)

- `ActionSpace`: 357 discrete actions on the base map, with `encode`, `decode`
  and `fill_mask`. Robber victims and trade partners are *relative seats*
  (1 = next player), so a policy works in any seat. Offering trades is not
  in the space.
- `ObservationSpace`: 1,324 floats from one player's point of view, scaled
  to about [0, 1]. Opponents' cards appear only as totals. `names()` labels
  every position.

## Parallel runs

Games run on all cores by default (`--threads N` to change). Each game's
seeds derive only from its index, so results are identical for any thread
count. Library code can use `catanatron::parallel::par_map(n, threads, f)`
the same way.

## State layout

`State` holds no heap data except a shared `Arc<CatanMap>`, so cloning it is
a ~1.5 KB memory copy (about 30 ns). Fixed-capacity lists use `FixedVec`.
The action log lives on `Game::history` and is optional
(`GameConfig::record_history`). Use `Game::copy_without_history` for lookahead.

## Differences from the Python engine

- **RNG.** It uses its own seeded RNG (xoshiro256**), so a given seed does not
  replay the same game as in Python. The same seed always gives the same game
  in Rust.
- **Ids match Python.** Node ids, edge ids and tile layout match Python. The
  tournament map is identical.
- **Action order.** Year of Plenty, maritime trade and robber options come in a
  fixed order. Python builds these from sets, so its order is arbitrary.
- **Typed action values.** A maritime trade is `{give, rate, get}`, not a
  5-tuple. Dice, stolen cards and drawn dev cards are recorded in
  `ActionRecord::result`.
- **Longest road follows the official rules:**
  - A road ending at an opponent's building counts.
  - No one can build a road through an opponent's settlement.
  - When a road is cut, the holder keeps the card only while still tied for
    longest at 5 or more. Otherwise it goes to the single leader at 5 or more,
    or to no one.
- **Trade offers.** The offering player is never asked to answer their own
  offer. An offer is valid only if the offering player holds the cards.
