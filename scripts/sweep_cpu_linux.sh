#!/usr/bin/env bash
set -euo pipefail
cd /workspace/FuguBridge.jl
export PUFFERLIB_SOURCE=/workspace/fugubridge-deps/PufferLib
export RAYLIB_ROOT=/workspace/fugubridge-deps/raylib-5.5_linux_amd64
export LD_LIBRARY_PATH=$RAYLIB_ROOT/lib
/workspace/fugubridge-deps/julia-1.12.7/bin/julia --startup-file=no --project=benchmark scripts/sweep_cpu.jl
