#include <cuda_runtime.h>
/* Benchmark-only native host loop. Accepts the very same graph instance as Julia. */
extern "C" int fg_graph_replay(void* executable, void* stream, int count) {
    for(int i=0;i<count;i++) {
        cudaError_t err=cudaGraphLaunch((cudaGraphExec_t)executable,(cudaStream_t)stream);
        if(err!=cudaSuccess) return (int)err;
    }
    return (int)cudaGetLastError();
}
