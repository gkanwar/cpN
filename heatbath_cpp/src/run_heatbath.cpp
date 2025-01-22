#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <random>

#include "args.hxx"
#include "config.h"
#include "cpn.h"
#include "metropolis.h"

using namespace std;

void write_cfg(ostream& os, const cpn::Config& cfg) {
  for (ull x = 0; x < cfg.geom.vol; ++x) {
    os.write((char*)&cfg.z[x], sizeof(cfg.z[x]));
  }
}

bool read_cfg(istream& is, cpn::Config& cfg) {
  for (ull x = 0; x < cfg.geom.vol; ++x) {
    is.read((char*)&cfg.z[x], sizeof(cfg.z[x]));
  }
  return (bool)is;
}

string make_block_fname(const string& prefix, int i) {
  ostringstream ss(prefix, ios::ate);
  ss << "_block" << i << ".dat";
  return ss.str();
}

int main(int argc, char** argv) {
  args::ArgumentParser parser("CP(N) heatbath");
  args::ValueFlag<ull> arg_L(parser, "L", "Lattice size", {'L'}, args::Options::Required);
  args::ValueFlag<double> arg_beta(parser, "beta", "Inverse coupling", {'b'}, args::Options::Required);
  args::ValueFlag<double> arg_mcmc_eps(parser, "eps", "MCMC epsilon", {'e'});
  args::ValueFlag<int> arg_mcmc_n_iter(parser, "n_iter", "MCMC total iters", {'n'}, args::Options::Required);
  args::ValueFlag<int> arg_mcmc_n_meas(parser, "n_meas", "MCMC meas interval", {'m'}, args::Options::Required);
  args::ValueFlag<int> arg_mcmc_n_save(parser, "n_save", "MCMC save interval", {'x'}, args::Options::Required);
  args::ValueFlag<int> arg_mcmc_n_block(parser, "n_block", "MCMC save block size", {'k'});
  args::ValueFlag<int> arg_mcmc_init(
      parser, "init", "Block from which to start MCMC (if > 0 will load previous cfg)", {'i'});
  args::ValueFlag<int> arg_seed(parser, "seed", "RNG seed", {'s'});
  args::ValueFlag<string> arg_prefix(parser, "prefix", "Output prefix", {'f'}, args::Options::Required);
  
  try {
    parser.ParseCLI(argc, argv);
  }
  catch (args::Help) {
    cout << parser;
    return 0;
  }
  catch (args::ParseError e) {
    cerr << e.what() << "\n";
    cerr << parser;
    return 1;
  }
  catch (args::ValidationError e) {
    cerr << e.what() << "\n";
    cerr << parser;
    return 1;
  }

  const double beta = args::get(arg_beta);
  const ull L = args::get(arg_L);
  const double mcmc_eps = arg_mcmc_eps ? args::get(arg_mcmc_eps) : 0.5;
  const int mcmc_n_iter = args::get(arg_mcmc_n_iter);
  const int mcmc_n_meas = args::get(arg_mcmc_n_meas);
  const int mcmc_n_save = args::get(arg_mcmc_n_save);
  const int mcmc_n_block = arg_mcmc_n_block ? args::get(arg_mcmc_n_block) : 1000;
  const int mcmc_init = arg_mcmc_init ? args::get(arg_mcmc_init) : 0;
  auto seed_t = chrono::high_resolution_clock::now().time_since_epoch().count();
  const int seed = arg_seed ? args::get(arg_seed) : seed_t;
  const string prefix = args::get(arg_prefix);

  if (mcmc_n_iter % mcmc_n_block != 0) {
    cerr << "n_block must divide n_iter\n";
    cerr << parser;
    return 1;
  }

  // init lattice geom and config
  const LattGeom geom({L, L});
  cpn::Config cfg(geom);
  if (mcmc_init > 0) {
    string init_fname = make_block_fname(prefix, mcmc_init-1);
    cout << "Loading initial config from last block " << init_fname << "\n";
    ifstream in(init_fname, ios::binary | ios::ate);
    auto expected = mcmc_n_block * sizeof(cfg.z);
    if (in.tellg() != expected) {
      cerr << "Initial block size mismatch: expected " << expected << ", got " << in.tellg() << "\n";
      return 2;
    }
    in.seekg(sizeof(cfg.z), ios::end);
    bool ok = read_cfg(in, cfg);
    if (!ok) {
      cerr << "Failed to read initial cfg.\n";
      return 2;
    }
  }
  else {
    cout << "Initializing unit config.\n";
    cpn::init_unit(cfg);
  }

  // init action
  cpn::SpinAction action(beta);
  cpn::SpinAction action_b1(1.0);
  cout << "Initial action: " << action(cfg) << "\n";

  my_rand_arr rng;
  for (int i = 0; i < N_BLOCK; ++i) {
    rng[i].seed(seed+i);
  }
  

  auto proposal = [mcmc_eps](const cpn::Spin& old_z, my_rand& rng) {
    uniform_int_distribution<int> nc_dist(0, NC-1);
    uniform_real_distribution<double> theta_dist(-mcmc_eps, mcmc_eps);
    cpn::Spin z(old_z);
    // U(2) rotation on random (ij) pair
    int i = nc_dist(rng);
    int j = nc_dist(rng);
    if (i == j) return z;
    double theta = theta_dist(rng);
    double alpha12 = theta_dist(rng);
    double alpha21 = theta_dist(rng);
    double alpha22 = theta_dist(rng);
    double alpha11 = alpha12 + alpha21 - alpha22;
    cdouble phase11 = exp(complex(0.0, alpha11));
    cdouble phase12 = exp(complex(0.0, alpha12));
    cdouble phase21 = exp(complex(0.0, alpha21));
    cdouble phase22 = exp(complex(0.0, alpha22));
    cdouble zi = z[i];
    cdouble zj = z[j];
    z[i] = phase11*zi*cos(theta) + phase12*zj*sin(theta);
    z[j] = phase22*zj*cos(theta) - phase21*zi*sin(theta);
    assert(abs(
        norm(z[i]) + norm(z[j])
        - norm(zi) - norm(zj)) < 1e-8);
    return z;
  };

  // just used for debugging, report to a single file
  ostringstream fname_u(prefix, ios::ate);
  fname_u << "_u.txt";
  ofstream out_u(fname_u.str());
  out_u << setprecision(18);
  ofstream out_ens;
  double acc = 0.0;
  auto start = chrono::high_resolution_clock::now();
  for (int i = 0; i < mcmc_n_iter; ++i) {
    acc += metropolis_update(action, cfg, proposal, rng);
    if ((i+1) % mcmc_n_meas == 0) {
      double u = action_b1(cfg)/geom.vol;
      out_u << u << "\n";
      cout << "Iter " << i+1 << " energy: " << u << "\n";
      cout << "Acc " << (100*acc/(i+1)) << "%\n";
      auto now = chrono::high_resolution_clock::now();
      double t = 0.001 * chrono::duration_cast<chrono::milliseconds>(now - start).count();
      double exp_t = t * mcmc_n_iter / (double)(i+1);
      cout << "Time " << t << " of " << exp_t << "\n";
    }
    // update block ostream at the start of each block
    if (i % (mcmc_n_save * mcmc_n_block) == 0) {
      int bi = i / (mcmc_n_save * mcmc_n_block);
      if (out_ens.is_open()) {
        out_ens.close();
      }
      out_ens.open(make_block_fname(prefix, bi), ios::binary);
    }
    if ((i+1) % mcmc_n_save == 0) {
      write_cfg(out_ens, cfg);
    }
  }
  out_u.close();
  out_ens.close();
}
