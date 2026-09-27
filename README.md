# FuguBridge.jl

FuguBridge is a Julia wrapper for [PufferLib](https://github.com/PufferAI/PufferLib)'s
C and CUDA environments. It uses the existing simulations, with observations,
actions, and rewards exposed as Julia arrays (or `CuArray`s on GPU).

This package is AI-generated, written with OpenAI Codex. It is independent of PufferAI.
The name comes from fugu, a pufferfish.

## Usage

Requires Julia 1.10+ and a C compiler with OpenMP (`gcc` on Windows, `cc` on Linux).
Put the compiler's tools on `PATH`, or set `CC`. Julia installs raylib through
`Raylib_jll` and downloads a pinned PufferLib source artifact on first use.

```julia
using Pkg
Pkg.add(url="https://github.com/msjgriffiths/FuguBridge.jl")
```

```julia
using FuguBridge

lib = Library(build("cartpole"))
Batch(lib, 256) do env
    for _ in 1:100
        env.actions .= 0f0          # Actions are zero-based Float32 values
        step!(env)
    end
    @show size(env.obs)             # (4, 256): features × agents
    @show sum(env.rewards)
end
```

For GPU, install `CUDA.jl` with `Pkg.add("CUDA")` and build a GPU environment:

```julia
using CUDA, FuguBridge
lib = Library(build("breakout"; backend=GPU()))
```

The same batch loop works on GPU. It requires NVIDIA hardware and `nvcc` on `PATH`
(or set `NVCC`). Steps are asynchronous; call `synchronize!(env)` before and after a timed loop.
Linux is the tested GPU platform.

The bundled PufferLib version has three native GPU environments: Breakout, Admiral,
and Robot Arm. Other environments, including Tetris, run on CPU. `environments()` lists what's
available. Some games need additional assets or libraries.

## Performance

On an L4 with 4,096 agents, Julia and native C ran at the same speed within measurement
noise (24 paired trials per environment). GPU steps allocate no Julia heap memory
after warmup. Capturing 64 steps in a CUDA graph improved Breakout throughput by 7%
and Admiral by 0.37%.

These are fixed-action environment benchmarks, without a model or training loop.
See the [measurements and profiling](docs/OVERHEAD.md), [API notes](docs/API.md),
and [CUDA graph example](examples/gpu_graph.jl).
