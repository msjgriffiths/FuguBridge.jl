using CUDA, FuguBridge
CUDA.allowscalar(false)

# Build first: build("breakout"; source=..., raylib=..., backend=GPU()).
lib = Library(only(ARGS))
Batch(lib, 4096) do env
    # A tiny illustrative Julia policy: choose a valid action on the GPU.
    # Replace this broadcast with your model and categorical sampler.
    for t in 1:100
        env.actions .= Float32(t % first(lib.action_sizes))
        step!(env)
    end
    synchronize!(env)
    println("Observations: ", size(env.obs), "; total final-step reward: ", sum(env.rewards))
end
