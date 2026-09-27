"""Discover source entries. `gpu=true` means a native PUF_GPU source, not just a policy encoder."""
function environments(source::AbstractString)
    ocean = joinpath(source, "ocean")
    [begin
        h, cu = (joinpath(ocean, name, name * ext) for ext in (".h", ".cu"))
        gpu = isfile(cu) && occursin(r"#define\s+PUF_BACKEND\s+PUF_GPU", read(cu, String))
        (; name, cpu=isfile(h) && name != "robot_arm", gpu)
    end for name in sort(readdir(ocean)) if isdir(joinpath(ocean, name))]
end

"""Compile one environment against an explicit PufferLib 5 source checkout.
Requires C11 (CPU) or nvcc (GPU), and a raylib installation with include/ and lib/.
Extra include/link requirements can be supplied in `cflags`/`ldflags`.
Each build gets its own directory, so rebuilding never overwrites a loaded library.
"""
function build(name::AbstractString; source, backend::Backend=CPU(),
               raylib=get(ENV, "RAYLIB_ROOT", ""),
               compiler=get(ENV, backend isa GPU ? "NVCC" : "CC", backend isa GPU ? "nvcc" : "cc"),
               cflags=String[], ldflags=String[], output=joinpath(dirname(@__DIR__), "build"))
    isempty(raylib) && throw(ArgumentError("provide raylib=... or set RAYLIB_ROOT"))
    source, raylib = realpath(source), realpath(raylib)
    match(r"^[a-z][a-z0-9_]*$", name) === nothing && throw(ArgumentError("invalid environment name"))
    entries = filter(x -> x.name == name, environments(source))
    isempty(entries) && throw(ArgumentError("environment not found: $name"))
    entry = only(entries)
    gpu = backend isa GPU
    (gpu ? entry.gpu : entry.cpu) || throw(ArgumentError("$name has no $(gpu ? "native GPU" : "CPU") backend"))
    hdr = joinpath(source, "ocean", name, name * (gpu ? ".cu" : ".h"))
    config = joinpath(source, "config", name * ".ini")
    isfile(config) || throw(ArgumentError("missing environment config: $config"))
    mkpath(output)
    out = mktempdir(output; prefix=name * (gpu ? "-gpu-" : "-cpu-"), cleanup=false)
    write(joinpath(out, "defaults.ini"), read(joinpath(source,"config","default.ini"), String) * "\n" * read(config, String))
    # Include a single upstream source, without rewriting or porting its logic.
    shim = joinpath(dirname(@__DIR__), "native", "bridge.c")
    unit = joinpath(out, gpu ? "bridge.cu" : "bridge.c")
    compat = joinpath(dirname(shim), "compat.h")
    write(unit, join(["#include \"" * replace(p, '\\'=>'/') * "\"" for p in (compat,hdr,shim)], "\n") * "\n")
    lib = joinpath(out, "libfugubridge." * Libdl.dlext)
    includes = ["-I" * p for p in (source, joinpath(source,"src"), dirname(hdr), joinpath(source,"vendor"), joinpath(raylib,"include"))]
    links = Sys.iswindows() ? [joinpath(raylib,"lib","libraylib.a"), "-lopengl32", "-lgdi32", "-lwinmm"] :
        ["-L" * joinpath(raylib,"lib"), "-lraylib", "-Wl,-rpath," * joinpath(raylib,"lib"), "-lm", "-ldl", "-lpthread"]
    opts = gpu ? ["-std=c++17", "-O3", "-shared", "-Xcompiler=-fPIC,-fvisibility=hidden", "--cudart=shared"] :
        ["-std=gnu11", "-O3", "-shared", "-fopenmp", "-fvisibility=hidden"]
    !gpu && !Sys.iswindows() && push!(opts, "-fPIC")
    Sys.iswindows() && append!(opts, ["-static-libgcc", "-static"])
    # nvcc passes host linker flags through explicitly.
    if gpu && !Sys.iswindows()
        filter!(x -> !startswith(x, "-Wl,"), links)
        append!(links, ["-Xlinker=-rpath", "-Xlinker=" * joinpath(raylib,"lib")])
    end
    cmd = Cmd([compiler; opts; includes; "-DPLATFORM_DESKTOP"; "-DPUFFER_" * uppercase(name); cflags; unit; links; ldflags; "-o"; lib])
    write(joinpath(out, "build-command.txt"), string(cmd) * "\n")
    write(joinpath(out, "source.txt"), "source=$source\nheader=$hdr\nsha256=$(bytes2hex(sha256(read(hdr))))\n")
    run(cmd)
    lib
end
