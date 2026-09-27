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
        @test_throws "tetris has no native GPU backend" build("tetris"; source, raylib, backend=GPU())
        for name in ("cartpole", "tetris", "chain_mdp", "squared_continuous", "admiral")
            lib = Library(build(name; source, raylib, compiler))
            @test lib isa Library{CPU}
            @test startswith(lib.path, FuguBridge.get_scratch!(FuguBridge,"build") * (Sys.iswindows() ? "\\" : "/"))
            n = 8
            seed=lib.seedable ? 73 : 0
            env = Batch(lib, n; seed)
            if name == "admiral"
                @test_throws "only one live batch" Batch(lib,n)
                @test_throws "only one live batch" Batch(Library(lib.path),n)
            end
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
            function trajectory(native, slots=n)
                Batch(lib,slots; seed) do e
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
            if name in ("cartpole", "tetris", "admiral")
                slots = 256 * lib.agent_multiple # Enough independent games for OpenMP.
                @test trajectory(false,slots) == trajectory(true,slots)
            end
            @test !isopen(env)
            @test close(env) === nothing
            @test_throws ArgumentError step!(env)
            @test_throws ArgumentError reset!(env)
            @test_throws ArgumentError Batch(lib,0)
        end
        if Sys.iswindows()
            mktemp() do path, log
                redirect_stderr(log) do
                    @test_throws ProcessFailedException build("onestateworld"; source, raylib, compiler)
                end
                flush(log)
                @test occursin("poisoned", read(path,String))
            end
        elseif Sys.islinux()
            lib = Library(build("chess"; source, raylib, compiler))
            mktempdir() do dir
                cd(dir) do
                    mkpath("resources/chess")
                    config = abspath("chess.ini")
                    write(config, read(joinpath(dirname(lib.path),"defaults.ini"),String) * "\n[env]\nfen_curric_pct = 1\n")
                    write("resources/chess/fens.txt", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n")
                    first_obs = Batch(lib,2;config) do e
                        @test_throws "only one live batch" Batch(lib,2;config)
                        @test_throws "only one live batch" Batch(Library(lib.path),2;config)
                        copy(e.obs)
                    end
                    @test_throws "agent count must fit" Batch(lib,1;config)
                    write("resources/chess/fens.txt", "4k3/8/8/8/8/8/8/4K3 w - - 0 1\n")
                    Batch(lib,2;config) do e
                        @test e.obs != first_obs # Both close paths must release the old FEN cache.
                        @test step!(e) === e
                    end
                end
            end
        end
    end
    @testset "Native shared-resource cleanup" begin
        src = source === nothing ? FuguBridge.pufferlib_source() : source
        raylib_dir = raylib === nothing ? FuguBridge.Raylib_jll.artifact_dir : raylib
        mktempdir() do dir
            exe = joinpath(dir, Sys.iswindows() ? "vec_close.exe" : "vec_close")
            fixture = joinpath(@__DIR__, "vec_close.c")
            config = joinpath(dir, "test.ini")
            write(config, "[env]\n[vec]\n")
            run(`$compiler -std=gnu11 -I$(joinpath(src,"src")) -I$(joinpath(raylib_dir,"include")) $fixture -o $exe`)
            @test success(`$exe $config`)
        end
    end
end
