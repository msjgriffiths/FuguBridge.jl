using CUDA, FuguBridge, Libdl, Random, Statistics, Printf
include("bench.jl")
CUDA.allowscalar(false)

const graphlib=Libdl.dlopen(joinpath(@__DIR__,"graph_replay.so"))
const replay_ptr=Libdl.dlsym(graphlib,:fg_graph_replay)

function graph_native!(e,exec,replays)
    stream=FuguBridge.stream_pointer(FuguBridge.backend(e.lib),e.token)
    p=Ptr{Cvoid}(Base.unsafe_convert(CUDA.CUgraphExec,exec))
    GC.@preserve e exec begin
        code=FuguBridge.ffi(replay_ptr,Cint,p,stream,Cint(replays))
        code==0 || error("native graph launch failed: $code")
    end
end
function graph_julia!(e,exec,replays)
    for _ in 1:replays; CUDA.launch(exec); end
end

function run_mode(e,mode,steps,exec,span)
    mode==:julia && return julia!(e,steps)
    mode==:native && return native!(e,steps)
    mode==:graph_julia && return graph_julia!(e,exec,steps÷span)
    mode==:graph_native && return graph_native!(e,exec,steps÷span)
    error("unknown mode")
end

function trial(lib,n,steps,mode;span=64,snapshot=false)
    Batch(lib,n) do e
        e.actions .= 0f0
        synchronize!(e)
        exec = if mode in (:graph_julia,:graph_native)
            graph=CUDA.capture() do
                for _ in 1:span; step!(e); end
            end
            CUDA.instantiate(graph)
        else
            nothing
        end
        # First graph launch may upload/prepare the executable. Give every mode
        # the same 64-step state advance before timing to exclude that one-off cost.
        run_mode(e,mode,span,exec,span)
        synchronize!(e)
        a,b=CuEvent(),CuEvent()
        GC.gc()
        CUDA.record(a)
        start=time_ns()
        run_mode(e,mode,steps,exec,span)
        submitted=time_ns()
        CUDA.record(b)
        synchronize!(e)
        finished=time_ns()
        result=(wall=(finished-start)/1e9, enqueue=(submitted-start)/1e9,
                device=CUDA.elapsed(a,b))
        snapshot ? (result,Array(e.obs),Array(e.rewards),Array(e.terminals)) : result
    end
end

function experiment(;repeats=24,agents=4096)
    rng=MersenneTwister(20260927)
    mkpath("runs/overhead")
    open("runs/overhead/raw.csv","w") do out
        println(out,"environment,agents,steps,replicate,order,mode,wall_seconds,enqueue_seconds,device_seconds")
        for name in ("breakout","admiral","robot_arm")
            lib=Library(strip(read("runs/$name-library.txt",String)))
            modes=[:native,:julia,:graph_native,:graph_julia]
            # Warm every specialization, including graph runtime/driver interop.
            for m in modes; trial(lib,agents,64,m); end
            snapshots=[trial(lib,agents,128,m;snapshot=true) for m in modes]
            @assert all(x -> x[2:end]==snapshots[1][2:end],snapshots) "graph trajectory mismatch"
            pilot=trial(lib,agents,64,:native)
            steps=64*clamp(round(Int,0.5/pilot.wall),1,2000)
            # Burn in the GPU before randomizing measured trials.
            for _ in 1:4; trial(lib,agents,steps,:native); end
            println(name," agents=",agents," steps=",steps," repeats=",repeats); flush(stdout)
            for rep in 1:repeats
                for (order,mode) in enumerate(shuffle(rng,modes))
                    r=trial(lib,agents,steps,mode)
                    @printf(out,"%s,%d,%d,%d,%d,%s,%.9f,%.9f,%.9f\n",name,agents,steps,rep,order,mode,r.wall,r.enqueue,r.device)
                    flush(out)
                end
            end
        end
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    experiment()
end
