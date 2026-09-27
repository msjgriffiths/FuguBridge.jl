module FuguBridgeCUDAExt
using FuguBridge, CUDA
import FuguBridge: allocate, token, stream_pointer, wait_backend, destroy, rawpointer, GPU

allocate(::GPU, T, dims...) = CUDA.zeros(T, dims...)
token(::GPU) = (; context=CUDA.context(), stream=CUDA.stream())
rawpointer(x::CuArray) = Ptr{Cvoid}(UInt(pointer(x)))
function stream_pointer(::GPU, t)
    CUDA.context() == t.context || throw(ArgumentError("use the batch's creating CUDA context"))
    CUDA.stream() == t.stream || throw(ArgumentError("use CUDA.stream!(batch.token.stream) do ... end"))
    Ptr{Cvoid}(Base.unsafe_convert(CUDA.CUstream, t.stream))
end
wait_backend(::GPU, t) = CUDA.context!(() -> CUDA.synchronize(t.stream), t.context)
function destroy(f, ::GPU, t)
    CUDA.context!(t.context) do
        CUDA.synchronize(t.stream)
        f()
    end
end
end
