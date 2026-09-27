# Validation and native-speed comparison

Measured 2026-09-27 on a dedicated RunPod NVIDIA L4 (24 GB), Linux,
Julia 1.12.7, CUDA.jl 5.8.5, native nvcc compiler 12.8, kernel driver 570.195.03.
Julia uses artifact runtime/compiler 13.0 and a forward-compatible user-mode
driver 13.3; see the follow-up's recorded component versions.
Upstream: PufferLib `6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2`.

## GPU stepping at 4,096 agent slots

| Environment | Julia agent-steps/s | Native agent-steps/s | Julia / native throughput |
|---|---:|---:|---:|
| Breakout | 436,214,493 | 436,750,322 | 99.88% |
| Admiral | 8,343,472 | 8,352,269 | 99.89% |
| Robot Arm | 536,280 | 536,518 | 99.96% |

The ratio of median times differs by less than 0.13% at this batch size. The initial
seven paired trials do not establish a statistically resolved slowdown: exploratory
bootstrap intervals at 4,096 slots include equal performance for all three games.
See the follow-up investigation in [OVERHEAD.md](OVERHEAD.md). These are environment
microbenchmarks with fixed zero actions, **not complete RL training rates**. No model,
sampling, gradient calculation, or optimizer is timed. Admiral has two agent slots per
game, so its game-steps/s is half its agent-steps/s.

## Method

Both paths use the exact same compiled upstream environment and CUDA kernels.
The Julia path loops over `step!(env)`; the native reference loops over `puf_step`
inside `fg_steps`, entirely within C/CUDA. This isolates the Julia binding/dispatch cost.
Both paths synchronize after their loop. Construction, compilation, reset, GC, and output
copies are outside the timer. There are no host/device buffer transfers inside a timed step.

Each replicate creates a fresh deterministic batch and uses the same zero actions.
JIT warmup precedes measurement. Seven replicates per path alternate native/Julia order;
the table compares median elapsed times. Expensive cases choose a smaller common iteration
count from a pilot, targeting approximately 0.25 seconds per replicate. Breakout's short
iterations hit the 1,000-step cap. Small deviations above 100% are measurement noise.
Before timing each batch size, full observations, rewards, and terminals after 100 steps
are checked for exact equality between the Julia and native loops.

| Agent slots | Breakout Julia/native | Admiral Julia/native | Robot Arm Julia/native |
|---:|---:|---:|---:|
| 1 | 100.13% | Not a complete two-agent game | 100.00% |
| 256 | 99.97% | 100.35% | 100.01% |
| 4,096 | 99.88% | 99.89% | 99.96% |
| 16,384 | 100.00% | 100.00% | 99.94% |

Raw per-replicate timings and median CSVs are in [measurements](measurements/).
The results support negligible wrapper overhead on these kernels and this machine;
they do not establish learned-policy throughput or performance on other GPUs.

## Correctness and coverage

- **45 GPU checks passed**, across all three native GPU backends. Coverage includes
  exact native-loop trajectory parity, GPU buffer dtype/use, singleton protection across
  duplicate library loads, GC with a live batch, wrong-stream rejection, close/reset checks,
  action shape validation, and CUDA graph capture/replay.
- **Zero Julia heap bytes per GPU step** after warmup for all three environments.
- **71 Windows CPU checks passed**: package API plus integration of Tetris, CartPole,
  Chain MDP (byte observations), Squared Continuous, and multi-agent Admiral.
- **3 additional Linux threaded CPU parity checks passed** at 256 agent slots with four
  OpenMP threads (Tetris, CartPole, Admiral).
- The GPU policy-loop and CUDA-graph examples both ran successfully on Breakout.
- Generic build/load sweep: **77/83 CPU entries on Linux**, **74/83 on Windows**.
  These counts mean compiled and dynamically loaded; they do not imply every game was
  played or every asset/configuration was tested.

The six Linux build failures require additional upstream dependencies/assets: `impulse_wars`
(Box2D), `matsci` (LAMMPS), `nethack` (fast-nle), and `osrs_colosseum`, `osrs_inferno`,
`osrs_zulrah` (generated model data). Windows also fails `boxoban`, `chess`, and `drone`
because their source uses unavailable POSIX APIs/headers. See the coverage CSVs.

The separately maintained Tetris CUDA port was inspected but was not incorporated.
The general wrapper supports its upstream CPU environment. Converting other CPU
environments to CUDA still requires a GPU implementation of their simulation.

## Reproduction and cost

The follow-up [overhead investigation](OVERHEAD.md) includes both pods' final costs:
$0.4237 estimated compute, or $0.8647 with the conservative storage allowance.
Both pods have been deleted and their absence verified. The details below describe
the first run alone.

The scripts under `scripts/` record the dedicated cloud setup; `benchmark/bench.jl`
can benchmark any built library. All three use nvcc `-O3 -arch=sm_89` on the L4.
For a different GPU, select its architecture. Preserve the source commit and defaults.

Pod `jb9sh6akcd1eus` existed from 16:07:53 to 16:27:24 UTC (19.52 minutes), at
$0.49/hour. GPU compute estimate: **$0.1594**. Conservative compute-plus-storage
estimate at $1/hour: **$0.3253**, within the authorized $5 cap. The billing endpoint
had not posted itemized charges when checked, so these are estimates, not an invoice.
An independent local watchdog imposed a 150-minute deadline. The pod was explicitly
deleted after downloading evidence; its absence was verified. No persistent volume
was created. The pre-existing Tetris pod was left running.

Full evidence is saved locally under `runs/l4-20260927/`, including the tested source,
compiled libraries, upstream C/CUDA sources, manifests, configs, logs, raw timings,
hardware metadata, and lifecycle record. The archive retains Linux raylib symlinks;
the source/results subtree is also extracted for Windows browsing.

Archive SHA-256:
`faba3f45d727af6e9adfca1e7f0e59a58c5488e9156f342fb652c78cef1b5a4b`.
