using Pkg
Pkg.develop(path=dirname(@__DIR__))
Pkg.add(PackageSpec(name="CUDA", version="5.8.5"))
Pkg.add("Statistics")
using CUDA, FuguBridge
CUDA.versioninfo()
