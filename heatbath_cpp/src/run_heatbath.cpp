#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

#include "config.h"
#include "cpn.h"
#include "metropolis.h"

void write_cfg(std::ostream& os, const cpn::Config& cfg) {
  for (ull x = 0; x < cfg.geom.vol; ++x) {
    os.write((char*)&cfg.z[x], sizeof(cfg.z[x]));
  }
}

int main(int argc, char** argv) {
  const LattGeom geom({2, 2});
  cpn::Config cfg(geom);
  cpn::init_unit(cfg);
  const double beta = 1.0;
  cpn::SpinAction action(beta);
  cpn::SpinAction action_b1(1.0);

  std::cout << "Initial action: " << action(cfg) << "\n";

  my_rand rng;

  constexpr double EPS = 0.5;
  std::uniform_int_distribution<int> nc_dist(0, NC-1);
  std::uniform_real_distribution<double> theta_dist(-EPS, EPS);
  auto proposal = [&](const cpn::Spin& old_z, my_rand& rng) {
    cpn::Spin z(old_z);
    // simple rotation on (ij) pair
    int i = nc_dist(rng);
    int j = nc_dist(rng);
    if (i == j) return z;
    double theta = theta_dist(rng);
    double alpha12 = theta_dist(rng);
    double alpha21 = theta_dist(rng);
    double alpha22 = theta_dist(rng);
    double alpha11 = alpha12 + alpha21 - alpha22;
    cdouble phase11 = std::exp(std::complex(0.0, alpha11));
    cdouble phase12 = std::exp(std::complex(0.0, alpha12));
    cdouble phase21 = std::exp(std::complex(0.0, alpha21));
    cdouble phase22 = std::exp(std::complex(0.0, alpha22));
    cdouble zi = z[i];
    cdouble zj = z[j];
    z[i] = phase11*zi*cos(theta) + phase12*zj*sin(theta);
    z[j] = phase22*zj*cos(theta) - phase21*zi*sin(theta);
    assert(std::abs(
        std::norm(z[i]) + std::norm(z[j])
        - std::norm(zi) - std::norm(zj)) < 1e-8);
    return z;
  };
  std::ofstream out_u("u.dat");
  std::ofstream out_ens("ens.dat");
  out_u << std::setprecision(18);
  int n_meas = 10;
  int n_save = 100;
  double acc = 0.0;
  for (int i = 0; i < 100000; ++i) {
    acc += metropolis_update(action, cfg, proposal, rng);
    if ((i+1) % n_meas == 0) {
      std::cout << "Iter " << i+1 << " energy: " << action_b1(cfg)/geom.vol << "\n";
      std::cout << "Acc " << (100*acc/(i+1)) << "%\n";
      out_u << action_b1(cfg)/geom.vol << "\n";
    }
    if ((i+1) % n_save == 0) {
      write_cfg(out_ens, cfg);
    }
  }
}
