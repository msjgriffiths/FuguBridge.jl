# API and build reference

- `Library(path)` opens a built environment library and queries its layout.
- `build("cartpole")` compiles with the pinned PufferLib artifact and `Raylib_jll`.
- `environments()` lists the bundled source entries and their CPU/GPU availability.
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

The default source is PufferLib commit `6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2`.
`Artifacts.toml` pins its archive checksum and extracted tree hash. Julia downloads
it on first use and reuses the cached artifact. `Raylib_jll` supplies raylib 5.5,
its headers, and its shared-library dependencies through Pkg.

CPU builds require a C11/OpenMP compiler (`gcc` on Windows, `cc` on Linux).
GPU builds require nvcc, an NVIDIA GPU, and CUDA.jl. The compilers must be on `PATH`,
including any helper tools they invoke. Set `CC` or `NVCC` to select a compiler.
Julia manages the source and raylib downloads, not the compiler installation.

`build` accepts `compiler`, `cflags`, `ldflags`, and `output`; all builds go into separate
directories so loaded libraries are never overwritten. By default these directories
live in the package's Julia scratch space, keeping package and artifact directories
unchanged. Windows CPU builds need MinGW; Linux is the validated GPU target.

For local checkouts, `build("cartpole"; source="/path/to/PufferLib", raylib="/path/to/raylib")`
still works. `PUFFERLIB_SOURCE` and `RAYLIB_ROOT` also override the defaults. A local
raylib directory needs `include/` and `lib/` (a MinGW static library on Windows).

```sh
# Include native integration tests with the managed dependencies.
FUGUBRIDGE_TEST_NATIVE=true julia --project test/runtests.jl
```

See [design and limits](DESIGN.md) and [benchmark results](BENCHMARKS.md).
