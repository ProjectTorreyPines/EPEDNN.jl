module EPEDNN

import Flux
import Flux: NNlib
import Dates
import Memoize
import BSON

#= ===================================== =#
#  structs/constructors for the EPEDmodel
#= ===================================== =#
# EPEDmodel abstract type, since we could have different models
abstract type EPEDmodel end

# EPED1NN
struct EPED1NNmodel <: EPEDmodel
    fluxmodel::Flux.Chain
    name::String
    date::Dates.DateTime
    xnames::Vector{String}
    ynames::Vector{String}
    xm::Vector{Float32}
    xσ::Vector{Float32}
    ym::Vector{Float32}
    yσ::Vector{Float32}
    xbounds::Array{Float32}
    ybounds::Array{Float32}
    yp::Array{Float64}
end

# constructor that always converts to the correct types
function EPED1NNmodel(fluxmodel::Flux.Chain, name, date, xnames, ynames, xm, xσ, ym, yσ, xbounds, ybounds, yp)
    return EPED1NNmodel(
        fluxmodel,
        String(name),
        date,
        String.(xnames),
        String.(ynames),
        Float32.(reshape(xm, length(xm))),
        Float32.(reshape(xσ, length(xσ))),
        Float32.(reshape(ym, length(ym))),
        Float32.(reshape(yσ, length(yσ))),
        Float32.(xbounds),
        Float32.(ybounds),
        yp
    )
end

# constructor where the date is always filled out
function EPED1NNmodel(fluxmodel::Flux.Chain, name, xnames, ynames, xm, xσ, ym, yσ, xbounds, ybounds, yp)
    date = Dates.now()
    return EPED1NNmodel(fluxmodel, name, date, xnames, ynames, xm, xσ, ym, yσ, xbounds, ybounds, yp)
end

#= ========================================== =#
#  functions for saving/loading the EPEDmodel
#= ========================================== =#
function savemodel(model::EPEDmodel, filename::String)
    savedict = Dict()
    for name in fieldnames(EPED1NNmodel)
        savedict[name] = getproperty(model, name)
    end
    fullpath = dirname(dirname(@__FILE__)) * "/data/" * filename
    BSON.bson(fullpath, savedict)
    return fullpath
end

Memoize.@memoize function loadmodelonce(filename::String)
    return loadmodel(filename)
end

function loadmodel(filename::String)
    savedict = BSON.load(dirname(dirname(@__FILE__)) * "/data/" * filename, @__MODULE__)
    args = []
    for name in fieldnames(EPED1NNmodel)
        push!(args, savedict[name])
    end
    return EPED1NNmodel(args...)
end

#= ====================================== =#
#  functions to get the pedestal solution
#= ====================================== =#
function pedestal_array(pedmodel::EPED1NNmodel, x::AbstractMatrix{<:Real}; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=true)
    return hcat(collect(map(x0 -> pedestal_array(pedmodel, x0; only_powerlaw, warn_nn_train_bounds), eachslice(x; dims=2)))...)
end

function pedestal_array(pedmodel::EPED1NNmodel, x::AbstractVector{<:Real}; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=true)
    if eltype(x) <: Float32
        x32 = copy(x)
    else
        x32 = Float32.(x)
    end

    if warn_nn_train_bounds # training bounds are on the original data
        for ix in eachindex(x32)
            if any(x32[ix] .< pedmodel.xbounds[ix, 1])
                @warn("Extrapolation warning on $(pedmodel.xnames[ix])=$(minimum(x32[ix])) is below bound of $(pedmodel.xbounds[ix,1])")
            elseif any(x32[ix] .> pedmodel.xbounds[ix, 2])
                @warn("Extrapolation warning on $(pedmodel.xnames[ix])=$(maximum(x32[ix])) is above bound of $(pedmodel.xbounds[ix,2])")
            end
        end
    end

    x32[4] += 1.0 # delta + 1
    x32 .= abs.(x32) # to make Bt and Ip always positive
    y0 = power_law_fit_eval(pedmodel.yp, x32)
    if !only_powerlaw
        xn = (x32 .- pedmodel.xm) ./ pedmodel.xσ
        xn = Float32.(xn)
        yn = pedmodel.fluxmodel(xn)
        y1 = yn .* pedmodel.yσ .+ pedmodel.ym
        y = y0 .+ y1
    else
        y = y0
    end
    y .^= 2 # quare of the outputs
    y[1:9] .*= [x32[8] for k in 1:9] # multiply by density
    return y
end

function pedestal_array(
    pedmodel::EPED1NNmodel,
    a::T,
    betan::T,
    bt::T,
    delta::T,
    ip::T,
    kappa::T,
    m::T,
    neped::T,
    r::T,
    zeffped::T;
    only_powerlaw::Bool=false,
    warn_nn_train_bounds::Bool=true
) where {T<:Real}
    x = [a, betan, bt, delta, ip, kappa, m, neped, r, zeffped]
    return pedestal_array(pedmodel, x; only_powerlaw, warn_nn_train_bounds)
end

#= ================================================== =#
#  structs/constructors to interpret PedestalSolution
#= ================================================== =#
struct ModeSolution
    H
    meta
    superH
end

struct DiamagneticSolution
    GH::ModeSolution
    G::ModeSolution
    H::ModeSolution
end

struct PedestalSolution
    pressure::DiamagneticSolution
    width::DiamagneticSolution
end

function Base.Dict(pedsol::PedestalSolution)
    out = Dict()
    for field1 in fieldnames(PedestalSolution)
        out[field1] = Dict()
        for field2 in fieldnames(DiamagneticSolution)
            out[field1][field2] = Dict()
            for field3 in fieldnames(ModeSolution)
                out[field1][field2][field3] = getproperty(getproperty(getproperty(pedsol, field1), field2), field3)
            end
        end
    end
    return out
end

function PedestalSolution(
    pedmodel::EPED1NNmodel,
    a::Real,
    betan::Real,
    bt::Real,
    delta::Real,
    ip::Real,
    kappa::Real,
    m::Real,
    neped::Real,
    r::Real,
    zeffped::Real;
    only_powerlaw::Bool=false,
    warn_nn_train_bounds::Bool=true
)
    a, betan, bt, delta, ip, kappa, m, neped, r, zeffped = promote(a, betan, bt, delta, ip, kappa, m, neped, r, zeffped)
    x = [a, betan, bt, delta, ip, kappa, m, neped, r, zeffped]
    return PedestalSolution(pedmodel, x; only_powerlaw, warn_nn_train_bounds)
end

function PedestalSolution(pedmodel::EPED1NNmodel, x::AbstractVector; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=true)
    y = pedestal_array(pedmodel, x; only_powerlaw, warn_nn_train_bounds)
    return PedestalSolution(
        DiamagneticSolution(
            ModeSolution(y[1], y[2], y[3]),
            ModeSolution(y[4], y[5], y[6]),
            ModeSolution(y[7], y[8], y[9])
        ),
        DiamagneticSolution(
            ModeSolution(y[10], y[11], y[12]),
            ModeSolution(y[13], y[14], y[15]),
            ModeSolution(y[16], y[17], y[18])
        )
    )
end

#= ================================= =#
#  functors for EPED1NNmodel objects
#= ================================= =#
function (pedmodel::EPED1NNmodel)(x::Array; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=true)
    return pedestal_array(pedmodel, x; only_powerlaw, warn_nn_train_bounds)
end

function (pedmodel::EPED1NNmodel)(a, betan, bt, delta, ip, kappa, m, neped, r, zeffped; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=true)
    return PedestalSolution(pedmodel, a, betan, bt, delta, ip, kappa, m, neped, r, zeffped; only_powerlaw, warn_nn_train_bounds)
end

mutable struct InputEPED{T<:Real}
    a::Union{T,Missing}
    betan::Union{T,Missing}
    bt::Union{T,Missing}
    delta::Union{T,Missing}
    ip::Union{T,Missing}
    kappa::Union{T,Missing}
    m::Union{T,Missing}
    neped::Union{T,Missing}
    r::Union{T,Missing}
    zeffped::Union{T,Missing}

    function InputEPED()
        return InputEPED{Float64}()
    end
    function InputEPED{T}() where {T<:Real}
        return new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    end
end

function Base.show(io::IO, input::InputEPED)
    return print(io,
        "\n" *
        "           a : $(input.a)\n" *
        "       betan : $(input.betan)\n" *
        "          bt : $(input.bt)\n" *
        "       delta : $(input.delta)\n" *
        "          ip : $(input.ip)\n" *
        "       kappa : $(input.kappa)\n" *
        "           m : $(input.m)\n" *
        "       neped : $(input.neped)\n" *
        "           r : $(input.r)\n" *
        "     zeffped : $(input.zeffped)")
end

function (pedmodel::EPED1NNmodel)(input::InputEPED; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=true)
    return PedestalSolution(
        pedmodel,
        input.a,
        input.betan,
        input.bt,
        input.delta,
        input.ip,
        input.kappa,
        input.m,
        input.neped,
        input.r,
        input.zeffped;
        only_powerlaw,
        warn_nn_train_bounds
    )
end

"""
    run_epednn(input_eped::InputEPED; model_filename::String="EPED1NNmodel.bson", warn_nn_train_bounds::Bool)

Run EPEDNN starting from a InputEPED, using a specific `model_filename`.

The warn_nn_train_bounds checks against the standard deviation of the inputs to warn if evaluation is likely outside of training bounds.

Returns a `PedestalSolution` structure
"""
function run_epednn(input_eped::InputEPED; model_filename::String="EPED1NNmodel.bson", warn_nn_train_bounds::Bool)
    epedmod = EPEDNN.loadmodelonce(model_filename)
    return epedmod(input_eped...; warn_nn_train_bounds)
end

export run_epednn

#= ============= =#
#  power law fit
#= ============= =#
function power_law_fit(A, b, λ=0)
    A = vcat(transpose(b .* 0.0 .+ 1), log10.(abs.(A)))
    b = log10.(abs.(b))
    if λ > 0
        A = transpose(A)
        reg_solve(A, b, λ) = inv(A' * A + λ * I) * A' * b
        p = reg_solve(A, b, λ)
    else
        b = transpose(b)
        p = transpose(b / A)
    end
    return p
end

function power_law_fit_eval(py::AbstractMatrix, x::AbstractMatrix)
    yy = zeros(size(py)[1], size(x)[2])
    for k in 1:size(py)[1]
        yy[k, :] .= power_law_fit_eval(py[k, :], x)[1, :]
    end
    return yy
end

function power_law_fit_eval(py::AbstractVector, x::AbstractMatrix)
    return hcat(collect(map(x0 -> power_law_fit_eval(py, x0), eachslice(x; dims=2)))...)
end

function power_law_fit_eval(py::AbstractMatrix, x0::AbstractVector)
    yy = zeros(eltype(x0), size(py)[1])
    for k in 1:size(py)[1]
        yy[k, :] .= power_law_fit_eval(py[k, :], x0)
    end
    return yy
end

function power_law_fit_eval(p::AbstractVector, x0::AbstractVector)
    y = p[1]
    for i in eachindex(x0)
        y += (p[i+1] * log10(abs(x0[i])))
    end
    return 10.0^y
end

"""
    effective_triangularity(tri_lo::T, tri_up::T) where {T<:Real}a

Effective triangularity to be used as an EPED input. Defined as:
tri_eff = (2/3)*tri_min + (1/3)*tri_max
where tri_min is the minimum of upper and lower triangularity, and tri_max is the maximum
"""
function effective_triangularity(tri_lo::T, tri_up::T) where {T<:Real}
    tri_min = min(tri_lo, tri_up)
    tri_max = max(tri_lo, tri_up)
    return (2.0 / 3.0) * tri_min + (1.0 / 3.0) * tri_max
end

const document = Dict()
document[Symbol(@__MODULE__)] = [name for name in Base.names(@__MODULE__; all=false, imported=false) if name != Symbol(@__MODULE__)]

end # module
