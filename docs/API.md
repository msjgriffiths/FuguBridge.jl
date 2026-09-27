# API and build reference

- `Library(path)` opens a built environment library and queries its layout.
- `Batch(lib, agents; seed=0, config="")` creates and resets agent slots, not necessarily
  independent games. A multi-agent game consumes multiple slots.
- `env.obs`, `env.actions`, `env.rewards`, `env.terminals`, `env.action_mask` are persistent
  buffers. Copy observations before the next step when keeping a trajectory.
  The native GPU ABI has no action-mask output; its mask buffer remains all ones.
- `step!(env)` consumes actions already written into `env.actions` without copying.
- `step!(env, actions::AbstractArray{Float32})` checks the shape, then copies actions.
- `reset!` uses upstream reset semantics and **does not rewind RNG state**.
- `synchronize!` waits for the creating CUDA stream; CPU calls are already synchronous.
- `close` is idempotent. Prefer do blocks or explicit close; finalization is a fallback.

Actions are Float32 because that is the native ABI. Discrete values are **0-based**;
`lib.action_sizes` gives each head's size. Continuous settings can reinterpret the
upstream action heads; consult that environment's config. Values must be valid upstream
actions; the fast stepping method does not scan every action for bounds.

GPU buffers stay on their creating context/stream. Use `CUDA.stream!(env.token.stream) do
... end` when returning from another task/stream. Only one active GPU batch is allowed per
loaded library because upstream stores a global batch. GPU reset synchronizes to accommodate
upstream default-stream reset kernels. Steps remain asynchronous and can be CUDA-graph captured.

The supplied `config` file must be a complete upstream INI (start with the `defaults.ini`
next to the built library). Invalid settings may trigger upstream C assertions. No rendering,
PPO trainer, or older Python/Gym adapter is exposed. Environments with external assets or
special link dependencies still need those upstream dependencies and correct working directory.

## Build and validation

Tested against PufferLib commit `6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2`.
CPU: a C11 compiler, OpenMP, and raylib. GPU: nvcc, NVIDIA GPU, CUDA.jl, raylib.
`build` accepts `compiler`, `cflags`, `ldflags`, and `output`; all builds go into separate
directories so loaded libraries are never overwritten. Set `CC`, `NVCC`, or `RAYLIB_ROOT`
for local defaults. Windows CPU builds need MinGW; the compatibility header supplies
glibc-compatible `rand_r`. Linux is the validated GPU target.

```sh
# Package tests; set both variables to include native integration checks.
PUFFERLIB_SOURCE=/path/to/PufferLib RAYLIB_ROOT=/path/to/raylib julia --project test/runtests.jl
```

See [design and limits](DESIGN.md) and [benchmark results](BENCHMARKS.md).
