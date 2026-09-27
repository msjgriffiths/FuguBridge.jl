#!/usr/bin/env bash
# Diagnostic Nsight attempt: no CUDA activity was collected on the recorded pod.
# profile_cupti.sh is the successfully validated profiler entry point.
set -euo pipefail
cd /workspace/FuguBridge.jl
export LD_LIBRARY_PATH=/workspace/fugubridge-deps/raylib-5.5_linux_amd64/lib
export JULIA_NUM_THREADS=1
export FG_EXTERNAL_PROFILE=true
julia=/workspace/fugubridge-deps/julia-1.12.7/bin/julia
nsys=/opt/nvidia/nsight-systems/2026.3.2/target-linux-x64/nsys
# CUDA.jl selects a forward-compatible driver. Let Nsight load the same driver.
driver_dir=$("$julia" --startup-file=no --project=benchmark -e 'using CUDA; print(dirname(CUDA.libcuda))')
export LD_LIBRARY_PATH="$driver_dir:$LD_LIBRARY_PATH"
mkdir -p runs/overhead/traces
for env in breakout admiral robot_arm; do
    for mode in native julia graph_native graph_julia; do
        out=runs/overhead/traces/$env-$mode
        "$nsys" profile --trace=cuda --sample=none --cpuctxsw=none \
            --capture-range=cudaProfilerApi --capture-range-end=stop \
            --cuda-graph-trace=node --cuda-event-trace=false --force-overwrite=true -o "$out" \
            "$julia" --startup-file=no --project=benchmark benchmark/trace.jl "$env" "$mode" \
            > "$out.log" 2>&1
        "$nsys" stats --report cuda_api_sum,cuda_gpu_kern_sum --format csv "$out.nsys-rep" \
            > "$out-summary.csv" 2> "$out-stats.log"
    done
done
touch runs/overhead/profiling-complete
