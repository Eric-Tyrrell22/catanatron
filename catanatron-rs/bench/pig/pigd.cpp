// Match driver for the colonizing-pig engine and bot (github.com/TheWeiHu/colonizing-pig,
// checked out separately; compiled against its engine sources, see build.sh). The game runs
// in that engine; a referee (python/pig_match.py) drives it over stdin/stdout, one command
// per line, one JSON line back:
//
//   topo            board topology: tile_nodes [19][6], edge_nodes [72][2]
//   new <seed>      a random board as the web build draws it for <seed>; resets the game
//   legal           {"cur","phase","legal":[action ids]}
//   act <id>        applies a legal action for the current player, then any chance events
//   bot             the bot's action for the current player (it sees only that player's view)
//   state           hands, bank, dev cards, points, robber, winner, ...
//   quit
//
// act and bot answer {"action":id,"events":[[player, action, phase, info], ...]}: the action
// itself (info: the trade rate for a bank trade, else -1), then each chance event (action -1)
// with its outcome (roll total - 2, stolen resource, or drawn card). The referee decides what
// each side may see.
#include "../../../../external/colonizing-pig/engine/model.hpp"
#include "../../../../external/colonizing-pig/engine/champion.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace catan;

namespace {
Contestant champion;
Board board;
State state;
Config config;
Rng chance(1);
uint32_t game_seed = 1;
bool started = false;

template <class T> void array(std::ostream& out, const T* values, int size) {
    out << '[';
    for (int i = 0; i < size; ++i) { if (i) out << ','; out << int(values[i]); }
    out << ']';
}

struct Event { int player, action, phase, info; };

// Applies a legal action and the chance events after it; false if not legal.
bool step(int action, std::vector<Event>& events) {
    uint8_t actions[ACTIONS];
    int count = legal_actions(board, state, config, actions);
    if (action < 0 || action >= ACTIONS || std::find(actions, actions + count, action) == actions + count) return false;
    int info = -1;
    if (action >= TRADE) info = state.p[state.cur].ratio[(action - TRADE) / 4];
    events.push_back({state.cur, action, state.phase, info});
    apply_action(board, state, config, action);
    while (is_chance(state)) {
        int player = state.cur, phase = state.phase;
        int outcome = sample_chance(state, config, chance);
        events.push_back({player, -1, phase, outcome});
        apply_chance(board, state, config, outcome);
    }
    return true;
}

void answer(int action, const std::vector<Event>& events) {
    std::ostringstream out;
    out << "{\"action\":" << action << ",\"events\":[";
    for (size_t i = 0; i < events.size(); ++i) {
        if (i) out << ',';
        out << '[' << events[i].player << ',' << events[i].action << ',' << events[i].phase << ',' << events[i].info << ']';
    }
    out << "]}";
    std::cout << out.str() << std::endl;
}

void error(const std::string& message) { std::cout << "{\"error\":\"" << message << "\"}" << std::endl; }
}  // namespace

int main(int argc, char** argv) {
    // argv[1]: search worker threads for the bot (its search budget doesn't depend on them).
    if (!load(champion, CHAMPION_SPEC)) { std::cerr << "cannot load the bot (run from the colonizing-pig checkout)\n"; return 1; }
    champion.agent.workers = argc > 1 ? std::max(1, std::min(8, atoi(argv[1]))) : 1;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string cmd;
        in >> cmd;
        if (cmd == "quit") break;
        if (cmd == "topo") {
            const Topology& t = topology();
            std::ostringstream out;
            out << "{\"tile_nodes\":[";
            for (int i = 0; i < TILES; ++i) { if (i) out << ','; array(out, t.tile_nodes[i], 6); }
            out << "],\"edge_nodes\":[";
            for (int i = 0; i < EDGES; ++i) { if (i) out << ','; array(out, t.edge_nodes[i], 2); }
            out << "],\"port_nodes\":[";
            for (int a = 0; a < 2; ++a) {
                if (a) out << ',';
                out << '[';
                for (int i = 0; i < 9; ++i) { if (i) out << ','; array(out, t.port_nodes[a][i], 2); }
                out << ']';
            }
            out << "]}";
            std::cout << out.str() << std::endl;
        } else if (cmd == "new" || cmd == "board") {
            // new <seed>: a random board for <seed>.
            // board <seed> r0..r18 n0..n18 p0..p8 <alternate>: a given layout
            // (pig tile/port order; resources -1 = desert, ports -1 = 3:1), <seed> for chance.
            uint64_t seed;
            if (!(in >> seed)) { error(cmd + " needs a seed"); continue; }
            game_seed = uint32_t(seed);
            if (cmd == "new") {
                Rng layout(uint64_t(game_seed) * 1000003 + 1);
                board = Board::random(layout);
            } else {
                int8_t resources[TILES], ports[9]; uint8_t numbers[TILES]; int alt = 0, v = 0; bool ok = true;
                for (int i = 0; i < TILES && ok; ++i) { ok = bool(in >> v); resources[i] = int8_t(v); }
                for (int i = 0; i < TILES && ok; ++i) { ok = bool(in >> v); numbers[i] = uint8_t(v); }
                for (int i = 0; i < 9 && ok; ++i) { ok = bool(in >> v); ports[i] = int8_t(v); }
                ok = ok && bool(in >> alt);
                if (!ok) { error("board needs seed, 19 resources, 19 numbers, 9 ports, alternate"); continue; }
                board = Board::from_layout(resources, numbers, ports, alt != 0);
            }
            chance = Rng(uint64_t(game_seed) * 1000003 + 2);
            state = initial(board);
            started = true;
            const Topology& t = topology();
            std::ostringstream out;
            out << "{\"resources\":"; array(out, board.resources, TILES);
            out << ",\"numbers\":"; array(out, board.numbers, TILES);
            out << ",\"port_resources\":"; array(out, board.port_resources, 9);
            out << ",\"port_nodes\":[";
            for (int i = 0; i < 9; ++i) { if (i) out << ','; array(out, t.port_nodes[board.alternate_ports][i], 2); }
            out << "],\"robber\":" << int(state.robber) << "}";
            std::cout << out.str() << std::endl;
        } else if (!started) {
            error("no game: send new first");
        } else if (cmd == "legal") {
            uint8_t actions[ACTIONS];
            int count = done(state, config) ? 0 : legal_actions(board, state, config, actions);
            std::ostringstream out;
            out << "{\"cur\":" << int(state.cur) << ",\"phase\":" << int(state.phase) << ",\"legal\":";
            array(out, actions, count);
            out << "}";
            std::cout << out.str() << std::endl;
        } else if (cmd == "act") {
            int action;
            std::vector<Event> events;
            if (!(in >> action) || done(state, config) || !step(action, events)) { error("illegal action"); continue; }
            answer(action, events);
        } else if (cmd == "bot") {
            if (done(state, config)) { error("game over"); continue; }
            Rng choice(uint64_t(game_seed) * 1000003 + uint64_t(state.steps) * 1009 + 3);
            int action = choose(champion.agent, board, config, observe(state, state.cur), choice);
            std::vector<Event> events;
            if (!step(action, events)) { error("bot chose an illegal action"); continue; }
            answer(action, events);
        } else if (cmd == "state") {
            std::ostringstream out;
            out << "{\"cur\":" << int(state.cur) << ",\"phase\":" << int(state.phase) << ",\"turn\":" << state.turn
                << ",\"steps\":" << state.steps << ",\"winner\":" << int(state.winner)
                << ",\"done\":" << (done(state, config) ? "true" : "false") << ",\"robber\":" << int(state.robber)
                << ",\"longest_road\":" << int(state.longest_road) << ",\"largest_army\":" << int(state.largest_army)
                << ",\"bank\":"; array(out, state.bank, 5);
            out << ",\"deck\":"; array(out, state.deck, 5);
            out << ",\"players\":[";
            for (int p = 0; p < 2; ++p) {
                if (p) out << ',';
                const Player& pl = state.p[p];
                out << "{\"hand\":"; array(out, pl.hand, 5);
                out << ",\"dev\":"; array(out, pl.dev, 5);
                out << ",\"played\":"; array(out, pl.played, 5);
                out << ",\"ratio\":"; array(out, pl.ratio, 5);
                out << ",\"public\":" << public_points(state, p) << ",\"total\":" << total_points(state, p)
                    << ",\"road_len\":" << int(state.road_len[p]) << "}";
            }
            out << "]}";
            std::cout << out.str() << std::endl;
        } else {
            error("unknown command");
        }
    }
    return 0;
}
