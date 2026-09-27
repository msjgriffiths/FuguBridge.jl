#!/usr/bin/env bash
set -euo pipefail
cd /workspace
mkdir -p fugubridge-deps
cd fugubridge-deps
if [ ! -d julia-1.12.7 ]; then
    curl -fL --retry 3 https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-1.12.7-linux-x86_64.tar.gz -o julia.tar.gz
    tar xf julia.tar.gz
fi
if [ ! -d PufferLib ]; then
    git clone --depth 1 --branch 5.0 https://github.com/PufferAI/PufferLib.git
fi
test "$(git -C PufferLib rev-parse HEAD)" = 6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2
if [ ! -d raylib-5.5_linux_amd64 ]; then
    curl -fL --retry 3 https://github.com/raysan5/raylib/releases/download/5.5/raylib-5.5_linux_amd64.tar.gz -o raylib.tar.gz
    tar xf raylib.tar.gz
fi
export LD_LIBRARY_PATH="$PWD/raylib-5.5_linux_amd64/lib:${LD_LIBRARY_PATH:-}"
export JULIA_NUM_PRECOMPILE_TASKS=4
cd /workspace/FuguBridge.jl
/workspace/fugubridge-deps/julia-1.12.7/bin/julia --startup-file=no --project=benchmark scripts/setup_gpu.jl
nvidia-smi
