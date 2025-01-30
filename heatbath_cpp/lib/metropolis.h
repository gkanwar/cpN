#pragma once

#include <random>
#include "util.h"

extern std::uniform_real_distribution<double> unif_dist;

template<typename Action, typename Config, typename Proposal>
double metropolis_update(
    const Action& action, Config& cfg, const Proposal& propose, my_rand_arr& rng) {
  std::uniform_real_distribution<double> unif_dist;
  // split volume
  // const unsigned block_size = cfg.geom.vol >= N_BLOCK ? cfg.geom.vol/N_BLOCK : 1;
  // split axis 0
  const unsigned block_size = cfg.geom.dims[0] >= N_BLOCK ? cfg.geom.dims[0]/N_BLOCK : 1;
  int acc = 0;

  // even stripes
#pragma omp parallel for schedule(static,)
  for (ull x0 = 0; x0 < cfg.geom.dims[0]; ++x0) {
    if (x0 % 2 != 0) continue;
    auto& thread_rng = rng[x0 / block_size];
    for (ull x1 = 0; x1 < cfg.geom.dims[1]; ++x1) {
      ull x = cfg.geom.get_idx({x0, x1});
      // standard MH update
      const double old_S = action.local_action(cfg, x);
      const typename Config::Spin_t z = cfg.z[x];
      cfg.z[x] = propose(z, thread_rng);
      const double new_S = action.local_action(cfg, x);
      if (unif_dist(thread_rng) < exp(-new_S + old_S)) {
        acc++;
      }
      else {
        cfg.z[x] = z;
      }
    }
  }

  // // odd stripes
  // #pragma omp parallel for
  // for (ull x0 = 0; x0 < cfg.geom.dims[0]; ++x0) {
  //   if (x0 % 2 == 0) continue;
  //   auto& thread_rng = rng[x0 / block_size];
  //   for (ull x1 = 0; x1 < cfg.geom.dims[1]; ++x1) {
  //     ull x = cfg.geom.get_idx({x0, x1});
  //     // standard MH update
  //     const double old_S = action.local_action(cfg, x);
  //     const typename Config::Spin_t z = cfg.z[x];
  //     cfg.z[x] = propose(z, thread_rng);
  //     const double new_S = action.local_action(cfg, x);
  //     if (unif_dist(thread_rng) < exp(-new_S + old_S)) {
  //       acc++;
  //     }
  //     else {
  //       cfg.z[x] = z;
  //     }
  //   }
  // }

  // even sites
  #pragma omp parallel for
  for (ull x = 0; x < cfg.geom.vol; ++x) {
    // check parity even
    assert(cfg.geom.dims.size() == 2);
    auto coord_x = cfg.geom.coord(x);
    const int parity = (coord_x[0] + coord_x[1]) % 2;
    if (parity != 0) continue;
    // standard MH update
    const double old_S = action.local_action(cfg, x);
    const typename Config::Spin_t z = cfg.z[x];
    cfg.z[x] = propose(z, rng[x / block_size]);
    const double new_S = action.local_action(cfg, x);
    if (unif_dist(rng[x / block_size]) < exp(-new_S + old_S)) {
      acc++;
    }
    else {
      cfg.z[x] = z;
    }
  }
  // odd sites
  #pragma omp parallel for
  for (ull x = 0; x < cfg.geom.vol; ++x) {
    // check parity odd
    assert(cfg.geom.dims.size() == 2);
    auto coord_x = cfg.geom.coord(x);
    const int parity = (coord_x[0] + coord_x[1]) % 2;
    if (parity != 1) continue;
    // standard MH update
    const double old_S = action.local_action(cfg, x);
    const typename Config::Spin_t z = cfg.z[x];
    cfg.z[x] = propose(z, rng[x / block_size]);
    const double new_S = action.local_action(cfg, x);
    if (unif_dist(rng[x / block_size]) < exp(-new_S + old_S)) {
      acc++;
    }
    else {
      cfg.z[x] = z;
    }
  }

  return acc / (double)cfg.geom.vol;
}
