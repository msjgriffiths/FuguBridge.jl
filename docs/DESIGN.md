# FuguBridge: investigation and implementation plan

Source inspected: PufferAI/PufferLib branch 5.0, commit
`6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2` (also the current remote head on 2026-09-27).

## What can actually run on a GPU?

PufferLib's trainer is CUDA, but this does not mean every simulation is CUDA.
`src/pufferenv.h` distinguishes per-environment CPU calls from whole-batch GPU calls.
At this revision only `breakout`, `admiral`, and `robot_arm` define `PUF_BACKEND PUF_GPU`.
Many other `.cu` files implement policy encoders, not GPU environments.
Python/Gym adapters from older PufferLib releases are outside this ABI.

The existing Tetris project has a custom CUDA port: it strips rendering, annotates
functions, replaces a variable-length stack array, and gives each GPU thread fixed
storage for the grid and deck. That transformation is specific to Tetris. A generated
Julia function cannot make arbitrary C pointers, allocation, I/O, and simulation logic
device-compatible. A generic source translator would be a separate compiler project.

## Architecture

1. Compile the selected upstream source into a shared library with one generic C shim.
2. Ask the shim for observation dtype, observation width, action heads, and backend.
3. Allocate typed Julia arrays or CUDA arrays and bind their addresses once.
4. Call `puf_step` directly through a cached function pointer for every batch step.
5. Keep policy evaluation in Julia on those same arrays. CUDA graphs can capture
   the caller's model and the native step together.

`Library{CPU,T}` / `Library{GPU,T}` and a concrete `Batch` specialize allocation,
pointer handling, and stream checks. A single `@generated ffi` turns an argument-type
tuple into the literal signature required by `ccall`. There is no runtime `eval`,
per-environment Julia wrapper, Python bridge, or observation copy on native GPU steps.
The C compiler owns C struct layouts; Julia never mirrors an `Env` or `Agent` struct.

## Correctness decisions

- Native action values remain zero-based Float32, including multidiscrete heads.
- Observations are feature × agent columns (same bytes as upstream agent-major rows).
- Buffers are owned by Julia and rooted by the batch; closing synchronizes GPU work.
- Native GPU code has a singleton global per loaded library. A second active batch
  is rejected. Calls to the same batch/library must not run concurrently on host threads.
- A batch stays on the CUDA context and stream that created it; mismatched stream use
  errors rather than racing a producer. Normal stepping does not synchronize.
- Upstream Breakout and Robot Arm reset kernels ignore their bound stream. Reset
  synchronizes before and after the upstream call to make this safe without patching it.
- GPU and custom-vector initializers own their seeds. Unsupported seed overrides error.
- Upstream handles auto-reset and terminal observations. There is no invented truncation
  field or separate terminal-observation buffer in the new C ABI.
- Configurations remain alive for the lifetime of the batch. Some upstream code retains
  string pointers. Custom vector initializers receive one vector buffer and one policy.
- Invalid upstream configurations can call `exit`/`assert` inside C. This experimental
  in-process interface is not a crash-isolating sandbox; use upstream-valid settings.

## Validation plan

- CPU: Tetris, CartPole, byte observations, continuous actions, and multi-agent Admiral.
- Compile sweep for all discovered CPU entries; separate build coverage from runtime coverage.
- GPU: all three native implementations; exact trajectory agreement with the native loop;
  singleton guard, close/GC behavior, stream mismatch, and CUDA graph capture.
- Performance: alternate Julia and native host loops, warm compilation first, use identical
  initial conditions and actions, synchronize both paths, and report medians. This measures
  environment stepping, not policy training or end-to-end PPO throughput.
- Cloud budget: dedicated pod, explicit $5 cap, independent local termination watchdog,
  download/checksum results and sources before deletion, then verify pod absence.

Executed results are in [BENCHMARKS.md](BENCHMARKS.md). The follow-up
[OVERHEAD.md](OVERHEAD.md) adds longer randomized trials and CUPTI traces; it finds
no statistically resolved Julia/native throughput gap and measures gains from
capturing multiple steps in a CUDA graph.

## Naming

FuguBridge is an independent project; it is not maintained or endorsed by PufferAI.
Fugu supplies the pufferfish reference and Bridge describes the external interface.
The package/module is `FuguBridge`; the repository is `FuguBridge.jl`. This follows
capitalized ASCII/UpperCamelCase naming and the current Julia General registry guidance
that external wrappers indicate that role in the name. An exact GitHub repository search
returned no `FuguBridge.jl` matches on 2026-09-27; this is not a trademark claim.

References: [PufferLib source](https://github.com/PufferAI/PufferLib/tree/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2),
[PufferLib documentation](https://puffer.ai/docs.html),
[Julia package naming](https://github.com/JuliaRegistries/General/blob/master/NAMING_GUIDELINES.md).
