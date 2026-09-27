using CUDA, FuguBridge
CUDA.allowscalar(false)
allocated_step(e) = @allocated step!(e)
for name in ("breakout","admiral","robot_arm")
    lib=Library(strip(read("runs/$name-library.txt",String)))
    Batch(lib,256) do e
        step!(e); allocated_step(e); synchronize!(e)
        bytes=allocated_step(e)
        synchronize!(e)
        @assert bytes==0
        println(name,": ",bytes," Julia heap bytes per step")
    end
end
