#!/usr/bin/env bash
set -euo pipefail
cd /workspace/fugubridge-deps
if [ ! -d julia-1.12.7 ]; then
    curl -fL --retry 3 https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-1.12.7-linux-x86_64.tar.gz -o julia.tar.gz
    tar xf julia.tar.gz
fi
cd /workspace/FuguBridge.jl
export JULIA_NUM_PRECOMPILE_TASKS=4
export LD_LIBRARY_PATH=/workspace/fugubridge-deps/raylib-5.5_linux_amd64/lib
/workspace/fugubridge-deps/julia-1.12.7/bin/julia --startup-file=no --project=benchmark scripts/setup_overhead.jl
/usr/local/cuda/bin/nvcc -O3 -shared -Xcompiler=-fPIC --cudart=shared benchmark/graph_replay.cu -o benchmark/graph_replay.so
nvidia-smi -lgc 2040,2040 > /workspace/clocks.log 2>&1 || true
touch /workspace/overhead-setup-complete
