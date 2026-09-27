#!/usr/bin/env bash
set -euo pipefail
cd /workspace/FuguBridge.jl
export LD_LIBRARY_PATH=/workspace/fugubridge-deps/raylib-5.5_linux_amd64/lib
cp /workspace/{profile_cupti.sh,profile_overhead.sh,summarize_cupti.py} scripts/
mkdir -p examples
cp /workspace/gpu_graph.jl examples/
python3 scripts/summarize_cupti.py runs/overhead/cupti > runs/overhead/cupti-counts.log
/workspace/fugubridge-deps/julia-1.12.7/bin/julia --startup-file=no --project=benchmark \
    examples/gpu_graph.jl "$(cat runs/breakout-library.txt)" > runs/overhead/graph-example.log 2>&1
nvidia-smi > runs/overhead/nvidia-smi-final.txt
date -u > runs/overhead/completed-at.txt
cp /workspace/clocks.log runs/overhead/clock-lock-attempt.log
cp /workspace/nsight-install.log runs/overhead/nsight-install.log
cd /workspace
tar -czf fugubridge-overhead-evidence.tar.gz FuguBridge.jl
sha256sum fugubridge-overhead-evidence.tar.gz > fugubridge-overhead-evidence.sha256
cat fugubridge-overhead-evidence.sha256
