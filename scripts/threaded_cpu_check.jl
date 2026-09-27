using FuguBridge, Test
include("../benchmark/bench.jl")
rows=split.(readlines("runs/cpu-builds/results.tsv")[2:end], '\t')
@testset "Threaded native CPU trajectory parity" begin
    for name in ("tetris","cartpole","admiral")
        row=only(filter(r -> r[1]==name, rows))
        lib=Library(row[3])
        a,b=measure(lib,256,100,julia!),measure(lib,256,100,native!)
        @test a[2:end]==b[2:end]
    end
end
