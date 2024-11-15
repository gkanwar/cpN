#include "args.hxx"
#include "config.h"
#include "lattice.h"
#include <iostream>
#include <torch/torch.h>

struct Field {
  torch::Tensor inner;
  LattGeom geom;
};

Field make_zeros_field(const LattGeom& geom) {
  std::vector<int64_t> shape(geom.dims.begin(), geom.dims.end());
  shape.push_back(NC);
  return Field {
    .inner = torch::zeros(shape),
    .geom = geom
  };
}

int main(int argc, char** argv) {
  torch::NoGradGuard no_grad;
  std::cout << "Hello, world!\n";
}
