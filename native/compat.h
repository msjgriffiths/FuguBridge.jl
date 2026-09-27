#pragma once
#if defined(__CUDACC__) && defined(PUFFER_ADMIRAL)
/* Admiral uses these two launch helpers from upstream pufferl.cu. */
#define BLOCK_SIZE 256
static inline int grid_size(int n) { return (n + BLOCK_SIZE - 1) / BLOCK_SIZE; }
#endif
/* Windows has no rand_r; reproduce glibc's seeded RNG used by upstream Linux. */
#ifdef _WIN32
#include <stdint.h>
#include <stdlib.h>
#undef RAND_MAX
#define RAND_MAX 2147483647
static int rand_r(unsigned int* seed) {
    uint32_t x=*seed, result=0;
    for(int i=0;i<3;i++) {
        x=1103515245u*x+12345u;
        result=(result<<10)|((x>>16)&(i==0?2047u:1023u));
    }
    *seed=x; return (int)result;
}
/* CRT rand() has a 15-bit range, incompatible with the RAND_MAX above.
   Reject environments using it rather than silently changing their simulation.
   A "poisoned rand/srand" error means this environment needs Linux. */
#pragma GCC poison rand srand
#endif
