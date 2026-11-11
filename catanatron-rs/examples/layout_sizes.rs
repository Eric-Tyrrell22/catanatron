//! Observation sizes of every block combination without the relation block,
//! per map type, plus the land-node count: for choosing the relation block's
//! marker count so that its layouts collide with none of them.
use catanatron::encoding::{ObsFeatures, ObservationSpace};
use catanatron::map::{build_map, MapType, NumberPlacement};
use catanatron::rng::Rng;

fn main() {
    for mt in [MapType::Base, MapType::Mini] {
        let map = build_map(mt, NumberPlacement::OfficialSpiral, &mut Rng::new(1));
        let n_land = map.land_nodes.iter().filter(|&&x| x).count();
        let mut sizes = Vec::new();
        for k in 0..256 {
            let f = ObsFeatures { dice: k & 1 != 0, hands: k & 2 != 0, devs: k & 4 != 0, extras: k & 8 != 0, dice_history: k & 16 != 0,
                                  spots: k & 32 != 0, reach: k & 64 != 0, structure: k & 128 != 0, relation: false };
            if f.dice_history && !f.dice {
                continue;
            }
            sizes.push(ObservationSpace::with_features(&map, f).len());
        }
        println!("{mt:?} {n_land} {}", sizes.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(","));
    }
}
