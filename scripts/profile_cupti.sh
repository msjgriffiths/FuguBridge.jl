#!/usr/bin/env bash
set -euo pipefail
cd /workspace/FuguBridge.jl
export LD_LIBRARY_PATH=/workspace/fugubridge-deps/raylib-5.5_linux_amd64/lib
export JULIA_NUM_THREADS=1
export FG_EXTERNAL_PROFILE=false
/workspace/fugubridge-deps/julia-1.12.7/bin/julia --startup-file=no --project=benchmark benchmark/trace.jl
