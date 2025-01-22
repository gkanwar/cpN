#pragma once

#include <array>
#include <complex>
#include <random>

using cdouble = std::complex<double>;
using ull = unsigned long long;

constexpr unsigned N_BLOCK = 256;
using my_rand = std::mt19937_64;
using my_rand_arr = std::array<my_rand, N_BLOCK>;
