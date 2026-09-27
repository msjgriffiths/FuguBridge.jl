using CUDA, FuguBridge
cap = CUDA.capability(CUDA.device())
arch = "sm_$(cap.major)$(cap.minor)"
mkpath("runs")
for name in ("breakout", "admiral", "robot_arm")
    println("Building ",name); flush(stdout)
    path=build(name;backend=GPU(),cflags=["-arch=$arch"])
    write("runs/$(name)-library.txt",path)
    lib=Library(path)
    Batch(lib,256) do e
        step!(e)
        synchronize!(e)
        @assert all(isfinite,Array(e.obs))
        println(name," ",eltype(e.obs)," ",size(e.obs)," OK")
    end
end
