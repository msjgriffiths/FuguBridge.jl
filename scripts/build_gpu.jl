using CUDA, FuguBridge
source = "/workspace/fugubridge-deps/PufferLib"
raylib = "/workspace/fugubridge-deps/raylib-5.5_linux_amd64"
mkpath("runs")
for name in ("breakout", "admiral", "robot_arm")
    println("Building ",name); flush(stdout)
    path=build(name;source,raylib,backend=GPU(),compiler="/usr/local/cuda/bin/nvcc",cflags=["-arch=sm_89"])
    write("runs/$(name)-library.txt",path)
    lib=Library(path)
    Batch(lib,256) do e
        step!(e)
        synchronize!(e)
        @assert all(isfinite,Array(e.obs))
        println(name," ",eltype(e.obs)," ",size(e.obs)," OK")
    end
end
