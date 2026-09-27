using Test, FuguBridge

@testset "Package API" begin
    @test CPU() isa FuguBridge.Backend
    @test GPU() isa FuguBridge.Backend
    @test_throws ErrorException FuguBridge.allocate(GPU(), Float32, 2)
end

if haskey(ENV, "PUFFERLIB_SOURCE") || get(ENV,"FUGUBRIDGE_TEST_NATIVE","false")=="true"
    source = get(ENV, "PUFFERLIB_SOURCE", nothing)
    raylib = get(ENV, "RAYLIB_ROOT", nothing)
    compiler = get(ENV, "CC", Sys.iswindows() ? "gcc" : "cc")
    @testset "Upstream CPU integration" begin
        catalog = source === nothing ? environments() : environments(source)
        @test only(filter(x -> x.name == "breakout", catalog)).gpu
        @test !only(filter(x -> x.name == "tetris", catalog)).gpu
        @test_throws ArgumentError build("tetris"; source, raylib, backend=GPU())
        for name in ("cartpole", "tetris", "chain_mdp", "squared_continuous", "admiral")
            lib = Library(build(name; source, raylib, compiler))
            @test lib isa Library{CPU}
            @test startswith(lib.path, FuguBridge.get_scratch!(FuguBridge,"build") * (Sys.iswindows() ? "\\" : "/"))
            n = 8
            seed=lib.seedable ? 73 : 0
            env = Batch(lib, n; seed)
            @test size(env.obs) == (lib.obs_size,n)
            @test size(env.actions) == (length(lib.action_sizes),n)
            @test all(isfinite, env.obs)
            env.actions .= 0
            step!(env)
            GC.gc()
            @test step!(env) === env
            @test reset!(env) === env
            @test_throws DimensionMismatch step!(env,zeros(Float32,1,n+1))
            # True trajectory equivalence: fresh batches, same initial RNG and action.
            close(env)
            function trajectory(native)
                Batch(lib,n; seed) do e
                    fill!(e.actions,0)
                    if native
                        GC.@preserve e FuguBridge.check(lib,FuguBridge.ffi(lib.api.steps,Cint,e.handle,Ptr{Cvoid}(0),Cint(100)))
                    else
                        for _ in 1:100; step!(e); end
                    end
                    (copy(e.obs),copy(e.rewards),copy(e.terminals))
                end
            end
            @test trajectory(false) == trajectory(true)
            @test !isopen(env)
            @test close(env) === nothing
            @test_throws ArgumentError step!(env)
            @test_throws ArgumentError reset!(env)
            @test_throws ArgumentError Batch(lib,0)
        end
    end
end
