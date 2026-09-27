using FuguBridge
source=get(() -> FuguBridge.pufferlib_source(), ENV, "PUFFERLIB_SOURCE")
names=Set(ARGS)  # Optional subset; no arguments checks every CPU source entry.
mkpath("runs/cpu-builds")
open("runs/cpu-builds/results.tsv","w") do summary
    println(summary,"environment\tstatus\tlibrary")
    for e in environments(source)
        e.cpu || continue
        isempty(names) || e.name in names || continue
        path=""
        status="failed"
        open("runs/cpu-builds/$(e.name).log","w") do log
            redirect_stdout(log) do
                redirect_stderr(log) do
                    try
                        path=build(e.name;source)
                        Library(path)
                        status="compiled"
                    catch err
                        showerror(log,err); println(log)
                    end
                end
            end
        end
        println(summary,"$(e.name)\t$status\t$path"); flush(summary)
        println(e.name," ",status); flush(stdout)
    end
end
