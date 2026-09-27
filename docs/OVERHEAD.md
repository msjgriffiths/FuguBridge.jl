# Is the small gap fundamental?

The initial 0.04–0.12% difference did not survive a stronger experiment as a
statistically resolved Julia penalty. Both languages execute the same upstream
CUDA kernels on the same buffers and stream. There is host work in the wrapper,
but asynchronous execution can overlap it with GPU work; its existence does not
imply an equal penalty in end-to-end stepping throughput.

## Longer randomized comparison

NVIDIA L4, 4,096 agent slots, 24 randomized paired replicates per environment.
Reported ratios are the mean of within-replicate throughput ratios, with a 95%
percentile bootstrap interval (20,000 resamples of whole pairs).

| Environment | Julia / native throughput | 95% interval | Julia CUDA graph gain over Julia direct |
|---|---:|---:|---:|
| Breakout | 100.094% | 99.753–100.431% | +7.026% [6.760–7.296%] |
| Admiral | 99.9896% | 99.9692–100.0105% | +0.3669% [0.3510–0.3821%] |
| Robot Arm | 99.9860% | 99.9654–100.0075% | +0.0182% [−0.0054–0.0421%] |

All direct Julia/native intervals include 100%. This is evidence of negligible
overhead for these workloads, not proof of exactly zero overhead. Breakout's wider
interval in particular cannot resolve a difference of one tenth of a percent.
The graph improvement is resolved for Breakout and Admiral, not Robot Arm.

Julia and C replaying **the very same Julia-created graph** also have intervals
containing equal performance: Breakout 99.686–100.330%, Admiral 99.9876–100.0261%,
and Robot Arm 99.9753–100.0249%. Native graph replay uses a tiny C loop around
`cudaGraphLaunch`; Julia uses `CUDA.launch`.

## Method and controls

Four modes are shuffled within each replicate: native direct loop, Julia direct
loop, native graph replay, and Julia graph replay. Each trial starts a fresh batch,
uses fixed zero actions, and advances the same 64 warmup steps. Graphs contain 64
environment steps. Compilation, creation, graph instantiation/upload, garbage
collection, and result copies are outside the timed region. Both host wall time
and CUDA event time are recorded, and their conclusions agree.

Direct/native iteration counts were calibrated to about half a second and rounded
to multiples of 64: Breakout 50,752 steps, Admiral 1,024, Robot Arm 64. All four
modes were checked for exact observation/reward/terminal agreement after the same
128 steps. The test is deliberately simulation-only; a learned model, changing
actions, memory pressure, or another GPU can change the result. The 64-step graph
benchmark reuses fixed actions. A real rollout must capture policy evaluation and
trajectory writes as well; [the example](../examples/gpu_graph.jl) shows where the
policy belongs but uses a constant-action broadcast, not a learned model.

The native direct reference binds its stream and checks the CUDA launch status once
per loop. Julia's ordinary `step!` checks liveness, context, stream, and launch status
per step. These safeguards remain enabled. There is no reason in these measurements
to remove them for a speculative fraction of a percent.

Clock locking was denied by the provider. Trial randomization, burn-in, repeated
measurements, and paired analysis reduce drift sensitivity but do not eliminate
all possible clock/thermal/system effects. Bootstrap intervals describe this run's
sampling variability, not uncertainty across GPUs or policies.

## What the profiler established

CUDA.jl's integrated **CUPTI activity profiler** successfully captured all twelve
environment/mode combinations. This gives host CUDA API calls and GPU activity.
For this throughput question it is more directly useful than deterministic CPU
record/replay with rr: we need to see launch counts and device work together.

| Environment | Steps traced | Device kernels, every mode | Direct host kernel launches | Host graph launches |
|---|---:|---:|---:|---:|
| Breakout | 2,048 | 2,048 | 2,048 | 32 |
| Admiral | 128 | 384 | 384 | 2 |
| Robot Arm | 128 | 384 | 384 | 2 |

The Julia and native paths execute the same kernel counts. Graph replay replaces
many host launches with a few graph launches while preserving those device kernels.
This supports launch amortization as the explanation for the measured graph gains.
The traces show no simulation memory transfers. Each raw trace also contains an
8-byte host-to-device copy from the profiler's own `CuArray([1])` warmup, which occurs
before the first simulation kernel and is checked separately by the analysis script.

The direct Julia trace includes per-step `cudaGetLastError` calls; the native loop
checks once at the end. Host checks exist, but the unprofiled paired throughput
comparison does not resolve a penalty. Profiling changes timing, so the traces are
used to identify work, not to claim sub-percent performance differences.

Nsight Systems 2025.1.3 and 2026.3.2 did **not** capture CUDA activity in this container.
Full collection, API-delimited collection, and aligning the driver library search path
were tried. The specific cause remains unresolved; the initial suspicion of a simple
profiler/driver version mismatch was not sufficient. Those empty reports are not used
as performance evidence. The successful CUPTI traces preserve the actual benchmark
driver/runtime setup instead of changing that setup to accommodate Nsight.

References: [CUDA.jl profiling guide](https://cuda.juliagpu.org/dev/development/profiling/),
[Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/).

## Reproduce

`benchmark/overhead.jl` implements the four-way test. It expects library path files
under `runs/`, as created by `scripts/build_gpu.jl`, and `benchmark/graph_replay.so`:

```sh
nvcc -O3 -shared -Xcompiler=-fPIC --cudart=shared benchmark/graph_replay.cu -o benchmark/graph_replay.so
julia --project=benchmark benchmark/overhead.jl
python3 scripts/analyze_overhead.py runs/overhead/raw.csv
# Separate profiling run; never use profiled timings for the table above.
julia --project=benchmark benchmark/trace.jl
python3 scripts/summarize_cupti.py runs/overhead/cupti
```

The second pod restored the first run's exact compiled environment libraries and
Julia manifest. Native sources were compiled with nvcc 12.8.93 (`-O3 -arch=sm_89`).
Julia 1.12.7 / CUDA.jl 5.8.5 selected the artifact CUDA runtime/compiler 13.0 and
forward-compatible user-mode driver 13.3, on kernel driver 570.195.03. These are
different components; referring to all of them as merely "CUDA 12.8" is inaccurate.

The raw trials and analysis are in [measurements/overhead](measurements/overhead/).

## Evidence and budget

The follow-up pod `i5v4lio5evkwa4` ran from 16:29:44 to 17:02:05 UTC on 2026-09-27
at $0.49/hour, approximately 32.36 minutes. Its compute estimate is $0.2643.
Combined with the first pod, compute is **$0.4237**. Allowing $1/hour for both
compute and ephemeral storage gives a conservative total estimate of **$0.8647**,
within the authorized $5 limit. Itemized billing had not posted at the time of the
check; these are estimates, not invoiced charges.

Both test pods were explicitly deleted and their absence verified after downloading
and checksumming evidence. Independent local watchdogs enforced deadlines. The
pre-existing Tetris training pod was left running, and no persistent volumes were
created for these experiments.

Full host/device CSV traces, successful CUPTI summaries, failed Nsight attempts,
source, binaries, manifests, raw trials, and lifecycle records are saved locally at
`runs/overhead-20260927/`. The extracted copy is under `snapshot/FuguBridge.jl/`.
The compact analysis and trace summaries are included in this repository.

Follow-up archive SHA-256:
`702a402569396170818cefbda963153c02ad42f4d0458110f7fe1a6c0f4b10a1`.
