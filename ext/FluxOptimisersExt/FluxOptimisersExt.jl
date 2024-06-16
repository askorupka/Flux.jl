module FluxOptimisersExt

using ConcreteStructs: @concrete
using Flux: Flux, DistributedUtils, AbstractDevice
using .DistributedUtils: AbstractFluxDistributedBackend
###using FluxDeviceUtils: AbstractFluxDevice, gpu_device
using Optimisers: Optimisers, AbstractRule, Leaf
using Random: Random
using Setfield: @set!

# DistributedUtils
@concrete struct DistributedOptimizer{B <: AbstractFluxDistributedBackend} <: AbstractRule
    backend::B
    opt
end

function Optimisers.apply!(opt::DistributedOptimizer, state, x, y)
    y_avg = DistributedUtils.allreduce!(opt.backend, y, DistributedUtils.avg)
    return Optimisers.apply!(opt.opt, state, x, y_avg)
end

Optimisers.init(opt::DistributedOptimizer, x::AbstractArray) = Optimisers.init(opt.opt, x)

function Optimisers._adjust(opt::DistributedOptimizer, nt::NamedTuple)
    return DistributedOptimizer(opt.backend, Optimisers._adjust(opt.opt, nt))
end

function DistributedUtils.synchronize!!(
        backend::AbstractFluxDistributedBackend, ps::Leaf; root::Int=0)
    @set! ps.state = DistributedUtils.synchronize!!(backend, ps.state; root)
    return ps
end

end