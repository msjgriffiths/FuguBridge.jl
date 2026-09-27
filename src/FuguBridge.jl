module FuguBridge

using Libdl, SHA, BFloat16s, LazyArtifacts, Scratch
import Raylib_jll
export CPU, GPU, Library, Batch, build, environments, reset!, step!, synchronize!

abstract type Backend end
struct CPU <: Backend end
struct GPU <: Backend end

# ccall requires literal argument types: specialize once, with no eval or runtime lookup.
@generated function ffi(f::Ptr{Cvoid}, ::Type{R}, args::Vararg{Any,N}) where {R,N}
    :(ccall(f, $R, $(Expr(:tuple, args...)), $([:(args[$i]) for i in 1:N]...)))
end

const symbols = (:info, :create, :reset, :step, :close, :steps, :error)
struct Library{B<:Backend,T}
    path::String
    handle::Ptr{Cvoid}  # Kept loaded for the process; generated calls may outlive an instance.
    api::NamedTuple{symbols,NTuple{7,Ptr{Cvoid}}}
    obs_size::Int
    action_sizes::Vector{Int}
    mask_size::Int
    agent_multiple::Int
    seedable::Bool
end

function Library(path::AbstractString)
    path = realpath(path)
    handle = Libdl.dlopen(path, Libdl.RTLD_LOCAL | Libdl.RTLD_NOW)
    api = NamedTuple{symbols}(map(s -> Libdl.dlsym(handle, "fg_" * String(s)), symbols))
    info = (i, j=0) -> Int(ffi(api.info, Cint, Cint(i), Cint(j)))
    info(0) == 1 || error("Unsupported FuguBridge ABI")
    B = info(1) == 1 ? GPU : CPU
    T = (UInt8, BFloat16, Float32, Float64)[info(2)]
    Library{B,T}(path, handle, api, info(3), [info(5,i-1) for i in 1:info(4)], info(6), info(7), info(8)==1)
end

mutable struct Batch{L,O,A,R,M,S}
    lib::L
    handle::Ptr{Cvoid}
    obs::O
    actions::A
    rewards::R
    terminals::R
    action_mask::M
    token::S
end

allocate(::CPU, T, dims...) = zeros(T, dims...)
allocate(::Backend, T, dims...) = error("Load CUDA first: using CUDA, FuguBridge")
token(::CPU) = nothing
stream_pointer(::CPU, ::Nothing) = Ptr{Cvoid}(0)
wait_backend(::CPU, ::Nothing) = nothing
destroy(f, ::CPU, ::Nothing) = f()
rawpointer(x::Array) = Ptr{Cvoid}(pointer(x))
backend(::Library{B}) where B = B()

function check(lib, code)
    code == 0 || error(unsafe_string(ffi(lib.api.error, Ptr{UInt8})))
    nothing
end
function live(env)
    isopen(env) || throw(ArgumentError("environment is closed"))
    env.handle
end
Base.isopen(env::Batch) = env.handle != C_NULL

"""Create a batch of agent slots. Actions use native zero-based Float32 values.
Buffers are features × agents, owned by Julia and updated in place. Use `close` or a do block.
GPU batches use the creating CUDA context/stream and upstream's built-in seeds.
`config` is a complete INI path; omitted uses the defaults captured at build time.
"""
function Batch(lib::Library{B,T}, n::Integer; seed=0, config="") where {B,T}
    0 < n <= typemax(Cint) || throw(ArgumentError("n must fit a positive Cint"))
    n % lib.agent_multiple == 0 || throw(ArgumentError("n must be a multiple of $(lib.agent_multiple)"))
    0 <= seed <= typemax(Cuint) || throw(ArgumentError("invalid seed"))
    !lib.seedable && seed != 0 && throw(ArgumentError("upstream batch initializer owns its seeds; omit seed"))
    config = realpath(isempty(config) ? joinpath(dirname(lib.path), "defaults.ini") : config)
    b = B()
    obs = allocate(b, T, lib.obs_size, n)
    actions = allocate(b, Float32, length(lib.action_sizes), n)
    rewards, terminals = (allocate(b, Float32, n) for _ in 1:2)
    mask = allocate(b, UInt8, lib.mask_size, n)
    fill!(mask, 0x01)
    env = Batch(lib, C_NULL, obs, actions, rewards, terminals, mask, token(b))
    # Native constructors can use the default stream, so complete buffer initialization.
    wait_backend(b, env.token)
    GC.@preserve env config begin
        env.handle = ffi(lib.api.create, Ptr{Cvoid}, Cint(n), Cuint(seed),
            rawpointer(obs), rawpointer(actions), rawpointer(rewards), rawpointer(terminals),
            rawpointer(mask), pointer(config))
    end
    env.handle == C_NULL && check(lib, -1)
    # CUDA cleanup may yield: defer it out of Julia's finalizer context.
    finalizer(e -> (@async try close(e) catch err; @warn "FuguBridge cleanup failed" exception=err; end), env)
    try
        reset!(env)
    catch
        close(env)
        rethrow()
    end
    env
end

function Batch(f::Function, args...; kwargs...)
    env = Batch(args...; kwargs...)
    try f(env) finally close(env) end
end

"""Reset using the environment's upstream semantics, without separately reseeding it."""
function reset!(env::Batch)
    h = live(env)
    s = stream_pointer(backend(env.lib), env.token)
    GC.@preserve env check(env.lib, ffi(env.lib.api.reset, Cint, h, s))
    env
end

"""Advance all environments using `env.actions`; no copies or synchronization on GPU."""
function step!(env::Batch)
    h = live(env)
    s = stream_pointer(backend(env.lib), env.token)
    GC.@preserve env check(env.lib, ffi(env.lib.api.step, Cint, h, s))
    env
end

function step!(env::Batch, actions::AbstractArray{Float32})
    live(env)
    size(actions) == size(env.actions) || throw(DimensionMismatch("actions must match $(size(env.actions))"))
    stream_pointer(backend(env.lib), env.token)
    copyto!(env.actions, actions)
    step!(env)
end

function synchronize!(env::Batch)
    live(env)
    wait_backend(backend(env.lib), env.token)
    env
end

function Base.close(env::Batch)
    isopen(env) || return nothing
    destroy(backend(env.lib), env.token) do
        GC.@preserve env check(env.lib, ffi(env.lib.api.close, Cint, env.handle))
        env.handle = C_NULL
    end
    nothing
end

include("build.jl")
end
