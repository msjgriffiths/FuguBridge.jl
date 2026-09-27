# Performance and validation

Measured on 2026-09-27 using an NVIDIA L4 (24 GB), Linux, Julia 1.12.7 and CUDA.jl
5.8.5. Upstream: PufferLib `6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2`.

## Julia versus native stepping

At 4,096 agent slots, 24 randomized paired trials found no statistically resolved
Julia throughput penalty. Ratios below are means of within-pair throughput ratios;
the 95% intervals use 20,000 bootstrap resamples of whole pairs.

| Environment | Julia / native throughput | 95% interval | Julia graph gain over Julia direct |
|---|---:|---:|---:|
| Breakout | 100.094% | 99.753–100.431% | +7.026% [6.760–7.296%] |
| Admiral | 99.9896% | 99.9692–100.0105% | +0.3669% [0.3510–0.3821%] |
| Robot Arm | 99.9860% | 99.9654–100.0075% | +0.0182% [−0.0054–0.0421%] |

All direct Julia/native intervals include 100%. This bounds the overhead for these
workloads; it does not prove exactly zero overhead. Julia and C replaying the same
Julia-created graph also had intervals containing equal performance. Zero Julia heap
bytes were allocated per GPU step after warmup.

These are fixed-action simulation benchmarks without a model or training loop. Admiral
uses two agent slots per game. Different policies, batch sizes, or hardware can change
the result. The initial seven-pair measurements at 1–16,384 slots remain in
[measurements](measurements/); the stronger comparison is in
[overhead/raw.csv](measurements/overhead/raw.csv) and
[analysis.json](measurements/overhead/analysis.json).

## Method

Four modes are shuffled within each replicate: Julia and native direct loops, and
Julia and native CUDA graph replay. Both direct paths use the same compiled upstream
library and kernels. `step!` checks liveness, context, stream, and launch status every
step; the native reference `fg_steps` binds the stream and checks launch status once
per loop. The ordinary safeguards stay enabled.

Each trial creates a fresh batch with zero actions and advances 64 warmup steps.
Graphs contain 64 environment steps. Timed loops use 50,752 steps for Breakout, 1,024
for Admiral, and 64 for Robot Arm, calibrated toward half a second. Construction,
compilation, graph instantiation/upload, GC, and result copies are outside the timer.
Both paths synchronize after the loop. Wall time and CUDA event time are recorded.
Observations, rewards, and terminals agree exactly across all four modes after the
same 128 steps.

A real rollout must also evaluate its policy and save trajectories. The
[graph example](../examples/gpu_graph.jl) shows where policy evaluation belongs,
using a constant-action broadcast. Clock locking was unavailable; burn-in,
randomization, and paired analysis reduce drift effects without eliminating them.
The intervals describe this run's sampling variability, not variation across GPUs.

## Profiling

CUDA.jl's CUPTI profiler captured all twelve environment/mode combinations:

| Environment | Steps traced | Device kernels, every mode | Direct host kernel launches | Host graph launches |
|---|---:|---:|---:|---:|
| Breakout | 2,048 | 2,048 | 2,048 | 32 |
| Admiral | 128 | 384 | 384 | 2 |
| Robot Arm | 128 | 384 | 384 | 2 |

Graph replay reduces host launches while preserving device work, supporting launch
amortization as the explanation for the measured gains. There are no simulation
memory transfers. Each trace includes one 8-byte profiler warmup copy before the
first simulation kernel; the analysis checks it separately. See the
[trace summary](measurements/overhead/trace-summary.csv).

Profiling changes timing, so the performance table uses unprofiled trials. Nsight
Systems 2025.1.3 and 2026.3.2 failed to capture CUDA activity in this container;
the cause remains unresolved. Those empty reports are not performance evidence.

Native libraries used nvcc 12.8.93, `-O3 -arch=sm_89`. CUDA.jl used artifact
runtime/compiler 13.0 and user-mode driver 13.3 on kernel driver 570.195.03.
These versions refer to different components of the CUDA installation.

## Correctness and coverage

The L4 run passed 45 GPU integration checks across the three native GPU backends,
including exact trajectory parity, buffer types, duplicate-library singleton
protection, GC, stream rejection, lifecycle checks, and graph capture/replay. The
GPU policy and graph examples also ran successfully. The separate allocation checks
are now part of `test/gpu.jl`.

CPU integration tests cover CartPole, Tetris, byte observations, continuous actions,
multi-agent Admiral, and threaded native-loop parity. Regression tests also check
shared-resource cleanup and reject incompatible Windows random-number usage.

The original build/load sweep compiled 77/83 CPU entries on Linux and 74/83 on
Windows. These historical counts are not runtime coverage and predate the Windows
RNG guard. Six Linux failures required extra dependencies/assets: impulse_wars
(Box2D), matsci (LAMMPS), nethack (fast-nle), and the three osrs environments
(generated model data). See the coverage CSVs and [current platform limits](API.md).

## Reproduce

Run from the repository root. Pkg installs Julia dependencies, raylib, and pinned
PufferLib source. CPU tests still need a C11/OpenMP compiler; GPU tests need Linux,
nvcc, and supported NVIDIA hardware. Benchmark setup resolves a fresh Julia manifest;
the versions above identify the measured run.

```sh
julia --project -e 'using Pkg; Pkg.instantiate()'
FUGUBRIDGE_TEST_NATIVE=true OMP_NUM_THREADS=4 julia --project test/runtests.jl
julia --project test/build_sweep.jl       # Optional build/load coverage sweep
julia benchmark/setup.jl
julia --project=benchmark benchmark/build.jl
julia --project=benchmark test/gpu.jl
"${NVCC:-nvcc}" -O3 -shared -Xcompiler=-fPIC --cudart=shared benchmark/graph_replay.cu -o benchmark/graph_replay.so
julia --project=benchmark benchmark/overhead.jl
python3 benchmark/analyze.py runs/overhead/raw.csv
# Separate profiling run:
julia --project=benchmark benchmark/trace.jl
python3 benchmark/summarize_traces.py runs/overhead/cupti
```

`benchmark/build.jl` selects the current GPU's architecture and records library paths
under `runs/`. `benchmark/bench.jl` also accepts a library path for the initial
batch-size comparison (add `--gpu` for a GPU library). Set `CC`, `NVCC`,
`PUFFERLIB_SOURCE`, or `RAYLIB_ROOT` to override the defaults.
