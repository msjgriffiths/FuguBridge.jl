# FuguBridge.jl 🐡

**AI-generated:** This package and its documentation were generated using OpenAI Codex.

A small Julia interface to [PufferLib](https://github.com/PufferAI/PufferLib)'s native
reinforcement-learning environments. Multiple dispatch selects CPU or GPU buffers;
a generated FFI calls the upstream simulation directly. GPU observations, actions,
and rewards stay in `CuArray`s, ready for a Julia model.

*Fugu* is a pufferfish reference; *Bridge* identifies an independent wrapper.
Not affiliated with or endorsed by PufferAI.

## Performance

On an NVIDIA L4 with 4,096 agent slots, 24 randomized paired trials found **no
statistically resolved throughput gap from native C/CUDA** across Breakout, Admiral,
and Robot Arm. GPU steps allocate **zero Julia heap bytes** after warmup. Capturing
64 steps in a CUDA graph improved Breakout throughput by **7%** and Admiral by **0.37%**.

These measurements cover environment stepping with fixed actions, not full training.
See the [benchmarks](docs/BENCHMARKS.md) and [profiling analysis](docs/OVERHEAD.md).

## Usage

Requires Julia 1.10+, a [PufferLib 5.0 checkout](https://github.com/PufferAI/PufferLib/tree/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2),
[raylib 5.5](https://github.com/raysan5/raylib/releases/tag/5.5) with `include/` and `lib/`,
and a C11 compiler with OpenMP. Windows CPU builds require MinGW with its tools on
`PATH`; set `CC` if needed.

Install the unregistered package:

```julia
using Pkg
Pkg.add(url="https://github.com/msjgriffiths/FuguBridge.jl")
```

Run a batch of CartPole environments:

```julia
using FuguBridge

source = "/path/to/PufferLib"
raylib = "/path/to/raylib"
lib = Library(build("cartpole"; source, raylib))

Batch(lib, 256) do env
    for _ in 1:100
        env.actions .= 0f0          # Native actions are zero-based Float32
        step!(env)
    end
    synchronize!(env)               # Wait for GPU work; a no-op on CPU
    @show size(env.obs)             # (4, 256): features × agents
    @show sum(env.rewards)
end                                # Closes native resources
```

For GPU simulation, install CUDA.jl with `Pkg.add("CUDA")`, then replace the library
construction above with the following and use the same batch loop:

```julia
using CUDA
lib = Library(build("breakout"; source, raylib, backend=GPU()))
```

GPU builds require NVIDIA hardware and `nvcc` on `PATH` (or set `NVCC`); Linux is the
validated GPU platform. Upstream **Breakout, Admiral, and Robot Arm** have native GPU
backends. Other environments, including Tetris, use the CPU interface. Discover them
with `environments(source)`; some require additional upstream dependencies.

Buffers are reused: copy observations when saving a trajectory. GPU batches stay on
their creating CUDA stream, with one active batch per loaded library.
See the [API reference](docs/API.md) and [CUDA graph example](examples/gpu_graph.jl).
