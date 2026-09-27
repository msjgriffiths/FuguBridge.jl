using FuguBridge, Statistics, Printf

function native!(e, steps)
    s = FuguBridge.stream_pointer(FuguBridge.backend(e.lib), e.token)
    GC.@preserve e FuguBridge.check(e.lib,FuguBridge.ffi(e.lib.api.steps,Cint,e.handle,s,Cint(steps)))
    e
end
function julia!(e, steps)
    for _ in 1:steps; step!(e); end
    e
end
function measure(lib,n,steps,loop)
    Batch(lib,n) do e
        # Same deterministic initial conditions for both host loops.
        fill!(e.actions,0)
        synchronize!(e)
        GC.gc()
        t = @elapsed begin loop(e,steps); synchronize!(e); end
        (t, Array(e.obs), Array(e.rewards), Array(e.terminals))
    end
end
function benchmark(lib; batches=(1,256,4096,16384),steps=1000,repeats=7,raw=nothing)
    for n in batches
        n % lib.agent_multiple == 0 || continue
        for loop in (native!,julia!); measure(lib,n,5,loop); end
        a,b=measure(lib,n,100,julia!),measure(lib,n,100,native!)
        @assert a[2:end]==b[2:end] "trajectory mismatch at n=$n"
        # Bound expensive physics cases to roughly 0.25s per measured replicate.
        count=clamp(round(Int,0.25/max(a[1],b[1])*100),10,steps)
        jt,ct=Float64[],Float64[]
        for rep in 1:repeats
            for loop in (isodd(rep) ? (native!,julia!) : (julia!,native!))
                result=measure(lib,n,count,loop)
                push!(loop===native! ? ct : jt, result[1])
                if raw!==nothing
                    println(raw,n,",",count,",",rep,",",loop===native! ? "native" : "julia",",",result[1])
                    flush(raw)
                end
            end
        end
        j,c=median(jt),median(ct)
        @printf("%d,%d,%.9f,%.9f,%.6f,%.3f,%.3f\n",n,count,j,c,c/j,n*count/j,n*count/c)
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    if "--gpu" in ARGS
        @eval using CUDA
        CUDA.allowscalar(false)
    end
    lib=Library(ARGS[1])
    println("agents,steps,julia_seconds,native_seconds,native_fraction,julia_agent_steps_s,native_agent_steps_s")
    name=first(split(basename(dirname(lib.path)),"-gpu-"))
    mkpath("runs")
    open("runs/$(name)-raw.csv","w") do raw
        println(raw,"agents,steps,replicate,loop,seconds")
        benchmark(lib;raw)
    end
end
