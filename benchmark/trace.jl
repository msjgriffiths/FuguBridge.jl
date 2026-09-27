using CUDA, FuguBridge
include("overhead.jl")
function trace_run(name,mode)
lib=Library(strip(read("runs/$name-library.txt",String)))
steps=name=="breakout" ? 2048 : 128

Batch(lib,4096) do e
    e.actions .= 0f0
    synchronize!(e)
    graph=CUDA.capture() do
        for _ in 1:64; step!(e); end
    end
    executable=CUDA.instantiate(graph)
    # Compile each host path and upload graph before the capture range.
    for m in (:native,:julia,:graph_native,:graph_julia)
        run_mode(e,m,64,executable,64)
    end
    synchronize!(e)
    if get(ENV,"FG_EXTERNAL_PROFILE","false")=="true"
        CUDA.@profile external=true begin
            run_mode(e,mode,steps,executable,64)
            synchronize!(e)
        end
    else
        results = CUDA.@profile external=false begin
            run_mode(e,mode,steps,executable,64)
            synchronize!(e)
        end
        out="runs/overhead/cupti/$name-$mode"
        mkpath(dirname(out))
        for kind in (:host,:device)
            df=getproperty(results,kind)
            open("$out-$kind.csv","w") do io
                println(io,join(propertynames(df),','))
                for row in eachrow(df)
                    println(io,join(("\""*replace(string(v),"\""=>"\"\"")*"\"" for v in row),','))
                end
            end
        end
        open("$out-summary.txt","w") do io
            show(io,MIME"text/plain"(),results)
        end
        println(name," ",mode,": ",size(results.host,1)," host records, ",size(results.device,1)," device records")
        flush(stdout)
    end
end
end

if isempty(ARGS)
    for name in ("breakout","admiral","robot_arm"), mode in (:native,:julia,:graph_native,:graph_julia)
        trace_run(name,mode)
    end
else
    trace_run(ARGS[1],Symbol(ARGS[2]))
end
