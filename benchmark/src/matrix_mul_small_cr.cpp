#include "common_small.hpp"

#define BM(N, M, K) BENCH_TPL(small, N, M, K, C, C, R);

ALL_BM;
