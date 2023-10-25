#pragma once

#include <random>
#include "util.h"

extern std::uniform_real_distribution<double> unif_dist;

template<typename Action, typename Config, typename Proposal>
void metropolis_update(
    const Action& action, Config& cfg, const Proposal& propose, my_rand& rng) {
  int acc = 0;
  for (ull x = 0; x < cfg.geom.vol; ++x) {
    const double old_S = action.local_action(cfg, x);
    const typename Config::Spin_t z = cfg.z[x];
    cfg.z[x] = propose(z, rng);
    const double new_S = action.local_action(cfg, x);
    if (unif_dist(rng) < exp(-new_S + old_S)) {
      acc++; // accept
    }
    else {
      cfg.z[x] = z; // reject
    }
  }
}
