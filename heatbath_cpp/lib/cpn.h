#pragma once

#include "config.h"
#include "lattice.h"
#include "util.h"

namespace cpn {

using Spin = std::array<cdouble, NC>;

struct Config {
  using Spin_t = Spin;
  std::vector<Spin_t> z;
  LattGeom geom;

  Config(const LattGeom& geom)
      : geom(geom), z(geom.vol) {}
};

void init_unit(Config& cfg);

class Action {
 public:
  virtual double local_action(const Config& cfg, ull x) const = 0;
  virtual double operator()(const Config& cfg) const = 0;
};

class SpinAction : Action {
 public:
  SpinAction(double beta) : beta(beta) {}
  double link_action(const Spin& z, const Spin& zp) const;
  double local_action(const Config& cfg, ull x) const override;
  double operator()(const Config& cfg) const override;
 private:
  double beta;
};

}
