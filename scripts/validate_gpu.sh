#!/usr/bin/env bash
set -euo pipefail
cd /workspace/FuguBridge.jl
export LD_LIBRARY_PATH=/workspace/fugubridge-deps/raylib-5.5_linux_amd64/lib
export OMP_NUM_THREADS=4
julia=/workspace/fugubridge-deps/julia-1.12.7/bin/julia
if [ "${1:-}" != "bench-only" ]; then
    "$julia" --startup-file=no --project=benchmark test/gpu.jl > runs/gpu-tests.log 2>&1
fi
for env in breakout admiral robot_arm; do
    "$julia" --startup-file=no --project=benchmark benchmark/bench.jl "$(cat runs/$env-library.txt)" --gpu > "runs/$env-benchmark.csv" 2> "runs/$env-benchmark.log"
done
touch runs/validation-complete
