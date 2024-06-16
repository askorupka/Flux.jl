# ==============================================
#          inspired by avik-pal's Lux.jl
# ==============================================

module DistributedUtils

using ChainRulesCore: ChainRulesCore
using ConcreteStructs: @concrete
using ..Flux: AbstractFluxDistributedBackend, MPIBackend, NCCLBackend, get_device
### TODO check get_device. was using FluxDeviceUtils: get_device
### see https://fluxml.ai/Flux.jl/stable/gpu/

const CRC = ChainRulesCore

const NCCL_Initialized = Ref(false)
const MPI_Initialized = Ref(false)

"""
    initialized(backend::Type{<:AbstractFluxDistributedBackend})

Check if the given backend is initialized.
"""
initialized(::Type{<:MPIBackend}) = MPI_Initialized[]
initialized(::Type{<:NCCLBackend}) = NCCL_Initialized[]

"""
    initialize(backend::Type{<:AbstractFluxDistributedBackend}; kwargs...)

Initialize the given backend. Users can supply `cuda_devices` and `amdgpu_devices` to
initialize the backend with the given devices. These can be set to `missing` to prevent
initialization of the given device type. If set to `nothing`, and the backend is functional
we assign GPUs in a round-robin fashion. Finally, a list of integers can be supplied to
initialize the backend with the given devices.

Possible values for `backend` are:

  - `MPIBackend`: MPI backend for distributed training. Requires `MPI.jl` to be installed.
  - `NCCLBackend`: NCCL backend for CUDA distributed training. Requires `CUDA.jl`,
    `MPI.jl`, and `NCCL.jl` to be installed. This also wraps `MPI` backend for non-CUDA
    communications.
"""
function initialize(backend::Type{<:AbstractFluxDistributedBackend}; kwargs...)
    initialized(backend) && return
    __initialize(backend; kwargs...)
    return
end

function __initialize end

"""
    get_distributed_backend(backend::Type{<:AbstractFluxDistributedBackend})

Get the distributed backend for the given backend type. Possible values are:

  - `MPIBackend`: MPI backend for distributed training. Requires `MPI.jl` to be installed.
  - `NCCLBackend`: NCCL backend for CUDA distributed training. Requires `CUDA.jl`,
    `MPI.jl`, and `NCCL.jl` to be installed. This also wraps `MPI` backend for non-CUDA
    communications.

!!! danger

    `initialize(backend; kwargs...)` must be called before calling this function.
"""
function get_distributed_backend(backend::Type{<:AbstractFluxDistributedBackend})
    initialized(backend) ||
        error("Backend `$(backend)` is not initialized. Call `DistributedUtils.initialize` first.")
    return __get_distributed_backend(backend)
end

function __get_distributed_backend end

CRC.@non_differentiable get_distributed_backend(::Any...)

"""
    local_rank(backend::AbstractFluxDistributedBackend)

Get the local rank for the given backend.
"""
function local_rank end

CRC.@non_differentiable local_rank(::Any...)

"""
    total_workers(backend::AbstractFluxDistributedBackend)

Get the total number of workers for the given backend.
"""
function total_workers end

CRC.@non_differentiable total_workers(::Any...)

"""
    bcast!(backend::AbstractFluxDistributedBackend, sendrecvbuf; root::Int=0)
    bcast!(backend::AbstractFluxDistributedBackend, sendbuf, recvbuf; root::Int=0)

Backend Agnostic API to broadcast the given buffer `sendrecvbuf` or `sendbuf` to all
workers into `recvbuf`. The value at `root` will be broadcasted to all other workers.
"""
function bcast!(backend::AbstractFluxDistributedBackend, sendrecvbuf; root::Int=0)
    return __bcast!(backend, sendrecvbuf, get_device(sendrecvbuf); root)
end

function bcast!(backend::AbstractFluxDistributedBackend, sendbuf, recvbuf; root::Int=0)
    send_dev = get_device(sendbuf)
    recv_dev = get_device(recvbuf)
    if send_dev === recv_dev
        return __bcast!(backend, sendbuf, recvbuf, send_dev; root)
    else
        sendbuf_ = sendbuf |> recv_dev
        @warn "`sendbuf` and `recvbuf` are on different devices." maxlog=1
        __bcast!(backend, sendbuf_, recvbuf, recv_dev; root)
        return recvbuf
    end
end

function __bcast! end

CRC.@non_differentiable bcast!(::Any...)

function avg end

"""
    allreduce!(backend::AbstractFluxDistributedBackend, sendrecvbuf, op)
    allreduce!(backend::AbstractFluxDistributedBackend, sendbuf, recvbuf, op)

Backend Agnostic API to perform an allreduce operation on the given buffer `sendrecvbuf` or
`sendbuf` and store the result in `recvbuf`.

`op` allows a special `DistributedUtils.avg` operation that averages the result across all
workers.
"""
function allreduce!(backend::AbstractFluxDistributedBackend, sendrecvbuf, op::F) where {F}
    return __allreduce!(backend, sendrecvbuf, op, get_device(sendrecvbuf))
end

function allreduce!(
        backend::AbstractFluxDistributedBackend, sendbuf, recvbuf, op::F) where {F}
    send_dev = get_device(sendbuf)
    recv_dev = get_device(recvbuf)
    if send_dev === recv_dev
        return __allreduce!(backend, sendbuf, recvbuf, op, send_dev)
    else
        sendbuf_ = sendbuf |> recv_dev
        @warn "`sendbuf` and `recvbuf` are on different devices." maxlog=1
        __allreduce!(backend, sendbuf_, recvbuf, op, recv_dev)
        return recvbuf
    end
end

function __allreduce! end

CRC.@non_differentiable allreduce!(::Any...)

"""
    reduce!(backend::AbstractFluxDistributedBackend, sendrecvbuf, op; root::Int=0)
    reduce!(backend::AbstractFluxDistributedBackend, sendbuf, recvbuf, op; root::Int=0)

Backend Agnostic API to perform a reduce operation on the given buffer `sendrecvbuf` or
`sendbuf` and store the result in `recvbuf`.

`op` allows a special `DistributedUtils.avg` operation that averages the result across all
workers.
"""
function reduce!(
        backend::AbstractFluxDistributedBackend, sendrecvbuf, op::F; root::Int=0) where {F}
    return __reduce!(backend, sendrecvbuf, op, get_device(sendrecvbuf); root)
end

function reduce!(backend::AbstractFluxDistributedBackend,
        sendbuf, recvbuf, op::F; root::Int=0) where {F}
    send_dev = get_device(sendbuf)
    recv_dev = get_device(recvbuf)
    if send_dev === recv_dev
        return __reduce!(backend, sendbuf, recvbuf, op, send_dev; root)
    else
        sendbuf_ = sendbuf |> recv_dev
        @warn "`sendbuf` and `recvbuf` are on different devices." maxlog=1
        __reduce!(backend, sendbuf_, recvbuf, op, recv_dev; root)
        return recvbuf
    end
end

function __reduce! end

CRC.@non_differentiable reduce!(::Any...)

# synchronize!
"""
    synchronize!!(backend::AbstractFluxDistributedBackend, ps; root::Int=0)

Synchronize the given structure `ps` using the given backend. The value at `root` will be
broadcasted to all other workers.
"""

### TODO doesn't have to be chain
function synchronize!!(backend::AbstractFluxDistributedBackend, ps::Chain; root::Int=0)
    length(ps) == 0 && return ps
    return fmap(x -> synchronize!!(backend, x; root), ps.model)
end

function synchronize!!(backend::AbstractFluxDistributedBackend, ps::Tuple; root::Int=0)
    length(ps) == 0 && return ps
    return map(x -> synchronize!!(backend, x; root), ps)
end

function synchronize!!(backend::AbstractFluxDistributedBackend,
        ps::NamedTuple{fields}; root::Int=0) where {fields}
    length(ps) == 0 && return ps
    return NamedTuple{fields}(map(x -> synchronize!!(backend, x; root), values(ps)))
end

function synchronize!!(
        backend::AbstractFluxDistributedBackend, ps::AbstractArray{T}; root::Int=0) where {T}
    if isbitstype(T)
        bcast!(backend, ps; root)
        return ps
    end
    return map(x -> synchronize!!(backend, x; root), ps)
end

function synchronize!!(backend::AbstractFluxDistributedBackend, ps::T; root::Int=0) where {T}
    isbitstype(T) && return bcast!(backend, [ps]; root)[]
    return ps # If we don't know how to synchronize, just return the value. For ex, Symbol, String, etc.
end

# data container
"""
    DistributedDataContainer(backend::AbstractFluxDistributedBackend, data)

`data` must be compatible with `MLUtils` interface. The returned container is compatible
with `MLUtils` interface and is used to partition the dataset across the available
processes.

!!! danger

    `MLUtils.jl` must be installed and loaded before using this.
"""
@concrete struct DistributedDataContainer
    data
    idxs
end

function DistributedDataContainer(backend::AbstractFluxDistributedBackend, data)
    if Base.get_extension(@__MODULE__, :FluxMLUtilsExt) === nothing
        error("`MLUtils.jl` must be installed and loaded before using \
               `DistributedDataContainer`.")
    end
    return __construct_distributed_data_container(backend, data)
end

function __construct_distributed_data_container end

Base.length(ddc::DistributedDataContainer) = length(ddc.idxs)

Base.getindex(ddc::DistributedDataContainer, i) = getindex(ddc.data, ddc.idxs[i])

# Distributed Optimizer
"""
    DistributedOptimizer(backend::AbstractFluxDistributedBacked, optimizer)

Wrap the `optimizer` in a `DistributedOptimizer`. Before updating the parameters, this
averages the gradients across the processes using Allreduce.

## Arguments

  - `optimizer`: An Optimizer compatible with the Optimisers.jl package

!!! danger

    `Optimisers.jl` must be installed and loaded before using this.
"""
function DistributedOptimizer(backend::AbstractFluxDistributedBackend, opt)
    mod = Base.get_extension(@__MODULE__, :FluxOptimisersExt)
    mod === nothing &&
        error("`Optimisers.jl` must be installed and loaded before using this.")
    return getproperty(mod, :DistributedOptimizer)(backend, opt)
end

end