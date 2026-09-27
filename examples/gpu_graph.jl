using CUDA, FuguBridge
CUDA.allowscalar(false)

lib = Library(only(ARGS))
Batch(lib, 4096) do env
    advance() = (env.actions .= 0f0; step!(env))
    advance()                         # Compile policy/broadcast before capture
    synchronize!(env)
    # Capture several policy + environment steps to amortize host graph launches.
    graph = CUDA.capture() do
        for _ in 1:64
            advance()
        end
    end
    executable = CUDA.instantiate(graph)
    for _ in 1:16                      # 1,024 captured steps
        CUDA.launch(executable)
    end
    synchronize!(env)
    println("Captured policy + environment; observations remain on the GPU.")
end
