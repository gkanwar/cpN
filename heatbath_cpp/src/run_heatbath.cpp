#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

#include "args.hxx"
#include "config.h"
#include "cpn.h"
#include "metropolis.h"

#ifndef NC
#error "Must define Nc in build configuration"
#endif

void write_cfg(std::ostream& os, const cpn::Config& cfg) {
  for (ull x = 0; x < cfg.geom.vol; ++x) {
    os.write((char*)&cfg.z[x], sizeof(cfg.z[x]));
  }
}

int main(int argc, char** argv) {
  args::ArgumentParser parser("Heatbath for CP(N)");
  args::HelpFlag help(
      parser, "help", "Display this help menu", {'h', "help"});
  args::ValueFlag<double> arg_beta(
      parser, "beta", "Gauge coupling", {'b', "beta"},
      args::Options::Required);
  args::ValueFlag<ull> arg_L(
      parser, "L", "Lattice size", {'L', "L"},
      args::Options::Required);
  args::ValueFlag<int> arg_n_iter(
      parser, "n_iter", "MCMC iterations", {'n', "n_iter"},
      args::Options::Required);
  args::ValueFlag<int> arg_n_therm(
      parser, "n_therm", "MCMC therm steps", {"n_therm"});
  args::ValueFlag<int> arg_n_meas(
      parser, "n_meas", "MCMC steps between measurements", {"n_meas"});
  args::ValueFlag<int> arg_n_save(
      parser, "n_save", "MCMC steps between writing cfgs", {"n_save"});
  args::ValueFlag<int> arg_n_block(
      parser, "n_block", "Number of RG blocking steps", {"n_block"});
  args::ValueFlag<double> arg_eps(
      parser, "eps", "Metropolis proposal eps", {"eps"});
  args::ValueFlag<std::string> arg_prefix(
      parser, "prefix", "Output prefix", {"prefix"},
      args::Options::Required);
  args::ValueFlag<unsigned long long> arg_seed(
      parser, "seed", "Random seed", {'s', "seed"},
      args::Options::Required);
  try {
    parser.ParseCLI(argc, argv);
  }
  catch (args::Help) {
    std::cout << parser;
    return 0;
  }
  catch (args::ParseError e) {
    std::cerr << e.what() << "\n";
    std::cerr << parser;
    return 1;
  }
  catch (args::ValidationError e) {
    std::cerr << e.what() << "\n";
    std::cerr << parser;
    return 1;
  }

  const int n_iter = args::get(arg_n_iter);
  const int n_therm = args::get(arg_n_therm);
  const int n_meas = arg_n_meas ? args::get(arg_n_meas) : 10;
  const int n_save = arg_n_save ? args::get(arg_n_save) : 100;
  const int n_block = arg_n_block ? args::get(arg_n_block) : 0;
  const std::string prefix = args::get(arg_prefix);
  const ull L = args::get(arg_L);
  const LattGeom geom({L, L});
  cpn::Config cfg(geom);
  cpn::init_unit(cfg);
  const double beta = args::get(arg_beta);
  cpn::SpinAction action(beta);
  cpn::SpinAction action_b1(1.0);

  std::vector<LattGeom> block_geoms = {geom};
  for (int i = 0; i < n_block; ++i) {
    block_geoms.push_back(block_geoms[i].coarsen());
  }

  auto make_blocked_cfg = [&](const cpn::Config& cfg) {
    cpn::Config fine = cfg;
    for (int i = 0; i < n_block; ++i) {
      cpn::Config coarse(block_geoms[i+1]);
      coarsen_config(coarse, fine);
      fine = coarse;
    }
    return fine;
  };

  std::cout << "Initial action: " << action(cfg) << "\n";

  my_rand rng;

  const double EPS = arg_eps ? args::get(arg_eps) : 0.5;
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

  {
    std::ofstream out_u(prefix + "_u.dat");
    std::ofstream out_ens(prefix + "_ens.dat", std::ios::binary);
    out_u << std::setprecision(18);
    double acc = 0.0;
    for (int i = -n_therm; i < n_iter; ++i) {
      acc += metropolis_update(action, cfg, proposal, rng);
      if ((i+1) % n_meas == 0) {
        std::cout << "Iter " << i+1 << " energy: " << action_b1(cfg)/geom.vol << "\n";
        std::cout << "Acc " << (100*acc/(i+1+n_therm)) << "%\n";
        if (i >= 0) {
          out_u << action_b1(cfg)/geom.vol << "\n";
        }
      }
      if (i >= 0 && (i+1) % n_save == 0) {
        cpn::Config cfg_blocked = make_blocked_cfg(cfg);
        write_cfg(out_ens, cfg_blocked);
      }
    }
  }
}
