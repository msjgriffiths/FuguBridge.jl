using CUDA, Test, FuguBridge
include("../benchmark/bench.jl")
CUDA.allowscalar(false)

@testset "Native GPU integration" begin
    for name in ("breakout","admiral","robot_arm")
        lib=Library(strip(read("runs/$(name)-library.txt",String)))
        @test lib isa Library{GPU}
        @test_throws ArgumentError Batch(lib,0)
        @test_throws ArgumentError Batch(lib,256;seed=3)
        env=Batch(lib,256)
        @test env.obs isa CuArray
        @test_throws ErrorException Batch(lib,256)
        @test_throws ErrorException Batch(Library(lib.path),256)
        @test_throws DimensionMismatch step!(env,CUDA.zeros(Float32,1,255))
        GC.gc()
        @test step!(env) === env
        synchronize!(env)
        @test all(isfinite,Array(env.obs))
        CUDA.stream!(CuStream()) do
            @test_throws ArgumentError step!(env)
        end
        @test reset!(env) === env
        close(env)
        @test close(env) === nothing
        @test_throws ArgumentError step!(env)
        # Exact equality to a native loop after a sufficiently long trajectory.
        a,b=measure(lib,256,100,julia!),measure(lib,256,100,native!)
        @test a[2:end]==b[2:end]
        # GPU-produced actions, no host copy; capture/replay the environment step.
        Batch(lib,256) do e
            e.actions .= 0f0
            step!(e); synchronize!(e)
            graph=CUDA.capture() do
                e.actions .= 0f0
                step!(e)
            end
            exec=CUDA.instantiate(graph)
            CUDA.launch(exec)
            synchronize!(e)
            @test all(isfinite,Array(e.obs))
        end
    end
end
