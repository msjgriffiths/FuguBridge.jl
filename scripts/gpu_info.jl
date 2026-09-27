using CUDA, Libdl
CUDA.versioninfo()
println("Loaded CUDA libraries:")
foreach(println,filter(x->occursin("cuda",lowercase(x))||occursin("cupti",lowercase(x)),Libdl.dllist()))
