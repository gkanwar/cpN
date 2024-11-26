#include "cpn.h"
#include "config.h"
#include "util.h"

namespace cpn {

void init_unit(Config& cfg) {
  for (int i = 0; i < cfg.geom.vol; ++i) {
    cfg.z[i] = { 0 };
    cfg.z[i][0] = 1.0;
  }
}

double SpinAction::link_action(const Spin& z, const Spin& zp) const {
  cdouble z2 = 0.0;
  for (int i = 0; i < NC; ++i) {
    z2 += z[i] * std::conj(zp[i]);
  }
  return beta * (1.0 - std::norm(z2));
}

double SpinAction::local_action(const Config& cfg, ull x) const {
  double S = 0.0;
  const Spin& z = cfg.z[x];
  for (int mu = 0; mu < ND; ++mu) {
    ull x_fwd = cfg.geom.shift_fwd(x, mu);
    const Spin& z_fwd = cfg.z[x_fwd];
    S += link_action(z, z_fwd);
    ull x_bwd = cfg.geom.shift_bwd(x, mu);
    const Spin& z_bwd = cfg.z[x_bwd];
    S += link_action(z, z_bwd);
  }
  return S;
}

double SpinAction::operator()(const Config& cfg) const {
  double S = 0.0;
  for (ull x = 0; x < cfg.geom.vol; ++x) {
    const Spin& z = cfg.z[x];
    for (int mu = 0; mu < ND; ++mu) {
      ull x_fwd = cfg.geom.shift_fwd(x, mu);
      const Spin& z_fwd = cfg.z[x_fwd];
      S += link_action(z, z_fwd);
      // ull x_bwd = cfg.geom.shift_bwd(x, mu);
      // const Spin& z_bwd = cfg.z[x_bwd];
      // S += link_action(z, z_bwd);
    }
  }
  return S;
}

}
