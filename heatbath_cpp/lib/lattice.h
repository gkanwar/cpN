#pragma once

#include <cassert>
#include <array>
#include "config.h"
#include "util.h"

struct LattGeom {
  std::array<ull, ND> dims;
  ull vol;

 private:
  std::array<ull, ND> blocks;
  std::array<ull, ND> strides;

 public:
  LattGeom(const std::array<ull, ND>& dims) : dims(dims) {
    vol = 1;
    for (int i = 0; i < dims.size(); ++i) {
      strides[i] = vol;
      vol *= dims[i];
      blocks[i] = vol;
    }
  }

  std::array<ull, ND> coord(ull idx) const {
    std::array<ull, ND> out;
    for (int mu = 0; mu < ND; ++mu) {
      out[mu] = (idx % blocks[mu]) / strides[mu];
    }
    return out;
  }

  ull get_idx(const std::array<ull, ND>& coord) const {
    ull idx = 0;
    for (int mu = 0; mu < ND; ++mu) {
      idx += coord[mu] * strides[mu];
    }
    return idx;
  }

  int shift_site_idx(int idx, int diff, int ax) const {
    if (diff < 0) {
      diff += dims[ax];
    }
    int full_block_idx = idx - (idx % blocks[ax]);
    return ((idx + diff*strides[ax]) % blocks[ax]) + full_block_idx;
  }

  int shift_fwd(int idx, int ax) const {
    return shift_site_idx(idx, 1, ax);
  }

  int shift_bwd(int idx, int ax) const {
    return shift_site_idx(idx, -1, ax);
  }

  LattGeom coarsen() const {
    std::array<ull, ND> coarse_dims;
    for (int mu = 0; mu < ND; ++mu) {
      assert(dims[mu] % 2 == 0);
      coarse_dims[mu] = dims[mu]/2;
    }
    return LattGeom(coarse_dims);
  }
};
