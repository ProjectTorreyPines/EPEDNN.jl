module EPEDNN

import Flux
import Dates
import Memoize
import BSON
import SpecialFunctions: erf

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
    xm::Vector{Float64}
    xσ::Vector{Float64}
    ym::Vector{Float64}
    yσ::Vector{Float64}
    xbounds::Array{Float64}
    ybounds::Array{Float64}
    yp::Array{Float64}
end

# EPED1NN deep ensemble
#
# Same power-law + neural-network-residual formulation as `EPED1NNmodel`, but with N independently
# trained networks instead of one: the spread of their predictions is a measure of the uncertainty
# of the prediction (deep ensemble). See `ensemble_predict` and `ensemble_uncertainty`.
#
# The shipped model (`EPED1NNensemble.bson`) takes 12 inputs (the 10 inputs of `EPED1NNmodel` plus
# `nesep_ratio` and `tesep`) and predicts the pedestal pressure `p_E1` only (omega-star/`dmagH`
# diamagnetic model, first solution). The pedestal width is not a network output: it follows the
# analytic EPED1 width law, see `pedestal_width`.
struct EPED1NNensemble <: EPEDmodel
    fluxmodels::Vector{Flux.Chain}
    name::String
    date::Dates.DateTime
    xnames::Vector{String}
    ynames::Vector{String}
    xm::Vector{Float64}
    xσ::Vector{Float64}
    ym::Vector{Float64}
    yσ::Vector{Float64}
    xbounds::Array{Float64}
    ybounds::Array{Float64}
    yp::Array{Float64}
end

#= ========================================== =#
#  functions for saving/loading the EPEDmodel
#= ========================================== =#
"""
    savemodel(model::EPEDmodel, filename::String)

Save an `EPEDmodel` as `data/<filename>`

Flux models are saved as `Flux.state(...)` (plain arrays) and not as objects, so that saved models
keep loading across Flux versions: a single network is stored under `:fluxstate`, an ensemble of
networks under `:fluxstates`
"""
function savemodel(model::EPEDmodel, filename::String)
    savedict = Dict()
    for name in fieldnames(typeof(model))
        value = getproperty(model, name)
        if isa(value, Flux.Chain)
            savedict[:fluxstate] = Flux.state(value)
        elseif isa(value, AbstractVector{<:Flux.Chain})
            savedict[:fluxstates] = [Flux.state(fluxmodel) for fluxmodel in value]
        else
            savedict[name] = value
        end
    end
    fullpath = dirname(dirname(@__FILE__)) * "/data/" * filename
    BSON.bson(fullpath, savedict)
    return fullpath
end

Memoize.@memoize function loadmodelonce(filename::String)
    return loadmodel(filename)
end

"""
    loadmodel(filename::String)

Load an `EPEDmodel` from `data/<filename>`

Returns an `EPED1NNmodel` for a single network and an `EPED1NNensemble` for an ensemble of networks
"""
function loadmodel(filename::String)
    savedict = BSON.load(dirname(dirname(@__FILE__)) * "/data/" * filename, @__MODULE__)
    if haskey(savedict, :fluxstates)
        # the ensemble was trained in PyTorch, with the exact (error function) GELU activation
        fluxmodels = [chain_from_state(fluxstate, gelu_erf) for fluxstate in savedict[:fluxstates]]
        args = Any[name == :fluxmodels ? fluxmodels : savedict[name] for name in fieldnames(EPED1NNensemble)]
        return EPED1NNensemble(args...)
    else
        fluxmodel = chain_from_state(savedict[:fluxstate], Flux.gelu)
        args = Any[name == :fluxmodel ? fluxmodel : savedict[name] for name in fieldnames(EPED1NNmodel)]
        return EPED1NNmodel(args...)
    end
end

"""
    chain_from_state(fluxstate, activation)

Rebuild a 64 bit dense `Flux.Chain` from a `Flux.state(...)` named tuple

The number of layers and their sizes are taken from the stored weight matrices, `activation` is
applied to all layers but the last one
"""
function chain_from_state(fluxstate, activation)
    nlayers = length(fluxstate.layers)
    layers = [Flux.Dense(reverse(size(layer.weight))..., k == nlayers ? identity : activation) for (k, layer) in enumerate(fluxstate.layers)]
    fluxmodel = Flux.Chain(layers...) |> Flux.f64 # use a 64bit model
    Flux.loadmodel!(fluxmodel, fluxstate)
    return fluxmodel
end

"""
    gelu_erf(x)

Exact (error function) GELU activation: `x·Φ(x) = 0.5·x·(1 + erf(x/√2))`

This is what `torch.nn.GELU()` does, and thus what the `EPED1NNensemble` networks were trained with.
NNlib's `gelu` is the cheaper `tanh` approximation of the same function, which differs by up to
~0.5% on the predicted pedestal height.
"""
gelu_erf(x) = 0.5 * x * (1.0 + erf(x / sqrt(2.0)))

"""
    warn_train_bounds(pedmodel::EPEDmodel, x::AbstractVector{<:Real})

Raise a warning for each input that lies outside of the training bounds of the model
(bounds are on the original data, ie. before the internal preprocessing of the inputs)
"""
function warn_train_bounds(pedmodel::EPEDmodel, x::AbstractVector{<:Real})
    for ix in eachindex(x)
        if any(x[ix] .< pedmodel.xbounds[ix, 1])
            @warn("Extrapolation warning on $(pedmodel.xnames[ix])=$(minimum(x[ix])) is below bound of $(pedmodel.xbounds[ix,1])")
        elseif any(x[ix] .> pedmodel.xbounds[ix, 2])
            @warn("Extrapolation warning on $(pedmodel.xnames[ix])=$(maximum(x[ix])) is above bound of $(pedmodel.xbounds[ix,2])")
        end
    end
end

#= ====================================== =#
#  functions to get the pedestal solution
#= ====================================== =#
function pedestal_array(pedmodel::EPED1NNmodel, x::AbstractMatrix{<:Real}; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=true)
    return hcat(collect(map(x0 -> pedestal_array(pedmodel, x0; only_powerlaw, warn_nn_train_bounds), eachslice(x; dims=2)))...)
end

function pedestal_array(pedmodel::EPED1NNmodel, x::AbstractVector{<:Real}; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=true)
    xx = deepcopy(x)

    if warn_nn_train_bounds # training bounds are on the original data
        warn_train_bounds(pedmodel, xx)
    end

    xx[4] += 1.0 # delta + 1
    xx .= abs.(xx) # to make Bt and Ip always positive
    y0 = power_law_fit_eval(pedmodel.yp, xx)
    if !only_powerlaw
        xn = (xx .- pedmodel.xm) ./ pedmodel.xσ
        yn = pedmodel.fluxmodel(xn)
        y1 = yn .* pedmodel.yσ .+ pedmodel.ym
        y = y0 .+ y1
    else
        y = y0
    end
    y .^= 2 # quare of the outputs
    y[1:9] .*= [xx[8] for k in 1:9] # multiply by density
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
        # pressure
        DiamagneticSolution(
            ModeSolution(y[1], y[2], y[3]),
            ModeSolution(y[4], y[5], y[6]),
            ModeSolution(y[7], y[8], y[9])
        ),
        # width
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
    # additional inputs of the EPED1NNensemble model
    # (defaulted to the values that the EPED1NNmodel training set was generated with)
    nesep_ratio::Union{T,Missing} # ne_sep / ne_ped
    tesep::Union{T,Missing} # separatrix electron temperature [eV]

    function InputEPED()
        return InputEPED{Float64}()
    end
    function InputEPED{T}() where {T<:Real}
        return new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 75.0)
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
        "     zeffped : $(input.zeffped)\n" *
        " nesep_ratio : $(input.nesep_ratio)\n" *
        "       tesep : $(input.tesep)")
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
    epedmod = loadmodelonce(model_filename)
    return run_epednn(epedmod, input_eped; warn_nn_train_bounds)
end

export run_epednn

#= ================================================= =#
#  functions to get the pedestal solution of ensembles
#= ================================================= =#
"""
    pedestal_members(pedmodel::EPED1NNensemble, x::AbstractVector{<:Real}; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=true)

Pedestal solution of each individual member of the ensemble, as a `(n_outputs, n_members)` matrix

These are the samples that `pedestal_array`, `ensemble_predict` and `ensemble_uncertainty` reduce
"""
function pedestal_members(pedmodel::EPED1NNensemble, x::AbstractVector{<:Real}; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=true)
    xx = deepcopy(x)

    if warn_nn_train_bounds # training bounds are on the original data
        warn_train_bounds(pedmodel, xx)
    end

    xx[findfirst(==("delta"), pedmodel.xnames)] += 1.0 # delta + 1
    xx .= abs.(xx) # to make Bt and Ip always positive

    y0 = power_law_fit_eval(pedmodel.yp, xx)
    members = repeat(y0, 1, length(pedmodel.fluxmodels))
    if !only_powerlaw
        xn = (xx .- pedmodel.xm) ./ pedmodel.xσ
        for (k, fluxmodel) in enumerate(pedmodel.fluxmodels)
            members[:, k] .+= fluxmodel(xn) .* pedmodel.yσ .+ pedmodel.ym
        end
    end
    members .^= 2 # square of the outputs
    members .*= xx[findfirst(==("neped"), pedmodel.xnames)] # multiply by density
    return members
end

"""
    pedestal_array(pedmodel::EPED1NNensemble, x::AbstractVector{<:Real}; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=true)

Ensemble average of the pedestal solution (same layout as `pedestal_array` of a `EPED1NNmodel`)
"""
function pedestal_array(pedmodel::EPED1NNensemble, x::AbstractVector{<:Real}; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=true)
    members = pedestal_members(pedmodel, x; only_powerlaw, warn_nn_train_bounds)
    return vec(sum(members; dims=2)) ./ size(members, 2)
end

function pedestal_array(pedmodel::EPED1NNensemble, x::AbstractMatrix{<:Real}; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=true)
    return hcat(collect(map(x0 -> pedestal_array(pedmodel, x0; only_powerlaw, warn_nn_train_bounds), eachslice(x; dims=2)))...)
end

"""
    ensemble_predict(pedmodel::EPED1NNensemble, x::AbstractVector{<:Real}; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=false)

Pedestal height (MPa) predicted by the ensemble, as a `(mean, std)` named tuple

`std` is the disagreement between the members of the ensemble, which is the uncertainty of the prediction

NOTE: the shipped ensemble predicts the pedestal height only; the pedestal width is not a network
output, it follows the analytic EPED1 width law, see `pedestal_width`

Extrapolation warnings are off by default here, since `ensemble_uncertainty` reports the very same
information as a number
"""
function ensemble_predict(pedmodel::EPED1NNensemble, x::AbstractVector{<:Real}; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=false)
    members = pedestal_members(pedmodel, x; only_powerlaw, warn_nn_train_bounds)[1, :]
    μ = sum(members) / length(members)
    σ = sqrt(sum((members .- μ) .^ 2) / length(members)) # population standard deviation
    return (mean=μ, std=σ)
end

function ensemble_predict(pedmodel::EPED1NNensemble, input::InputEPED; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=false)
    return ensemble_predict(pedmodel, input_vector(pedmodel, input); only_powerlaw, warn_nn_train_bounds)
end

"""
    ensemble_uncertainty(
        pedmodel::EPED1NNensemble,
        x::AbstractVector{<:Real};
        sigma_threshold::Real=0.05,
        extrapolation_threshold::Real=0.0,
        only_powerlaw::Bool=false,
        warn_nn_train_bounds::Bool=false)

Pedestal height with its uncertainty, and whether the query point is in the distribution that the
ensemble was trained on. Returns a named tuple with:
  - `height`, `sigma`     : ensemble mean and standard deviation of the pedestal height [MPa]
  - `sigma_frac`          : `sigma/height`, the fractional uncertainty of the prediction
  - `extrapolation`       : `extrapolation_distance(...).max_distance` (0.0 = within training bounds)
  - `sigma_frac_combined` : `max(sigma_frac, extrapolation)`
  - `in_distribution`     : `sigma_frac < sigma_threshold && extrapolation <= extrapolation_threshold`

The two metrics are complementary: `sigma_frac` catches points that are within the training bounds
of each individual input, but away from the (thin) manifold where the training data actually lives;
`extrapolation` catches points far outside of the training bounds, where all members of the ensemble
fall back onto the same power law and thus agree with each other for the wrong reason.

`sigma_threshold` defaults to 5%, the 99th percentile of `sigma_frac` over the training set, and
`extrapolation_threshold` to 0, ie. any input outside of the training bounds counts. Raise the latter
to tolerate query points that sit just outside of the box (`extrapolation` is a fraction of the
training range of the input that is furthest out).
"""
function ensemble_uncertainty(
    pedmodel::EPED1NNensemble,
    x::AbstractVector{<:Real};
    sigma_threshold::Real=0.05,
    extrapolation_threshold::Real=0.0,
    only_powerlaw::Bool=false,
    warn_nn_train_bounds::Bool=false)
    prediction = ensemble_predict(pedmodel, x; only_powerlaw, warn_nn_train_bounds)
    extrapolation = extrapolation_distance(pedmodel, x).max_distance
    sigma_frac = prediction.std / max(prediction.mean, 1e-30)
    return (
        height=prediction.mean,
        sigma=prediction.std,
        sigma_frac=sigma_frac,
        extrapolation=extrapolation,
        sigma_frac_combined=max(sigma_frac, extrapolation),
        in_distribution=(sigma_frac < sigma_threshold && extrapolation <= extrapolation_threshold))
end

function ensemble_uncertainty(
    pedmodel::EPED1NNensemble,
    input::InputEPED;
    sigma_threshold::Real=0.05,
    extrapolation_threshold::Real=0.0,
    only_powerlaw::Bool=false,
    warn_nn_train_bounds::Bool=false)
    return ensemble_uncertainty(pedmodel, input_vector(pedmodel, input); sigma_threshold, extrapolation_threshold, only_powerlaw, warn_nn_train_bounds)
end

#= =================================== =#
#  functors for EPED1NNensemble objects
#= =================================== =#
function (pedmodel::EPED1NNensemble)(
    x::AbstractVector{<:Real};
    sigma_threshold::Real=0.05,
    extrapolation_threshold::Real=0.0,
    only_powerlaw::Bool=false,
    warn_nn_train_bounds::Bool=false)
    return ensemble_uncertainty(pedmodel, x; sigma_threshold, extrapolation_threshold, only_powerlaw, warn_nn_train_bounds)
end

function (pedmodel::EPED1NNensemble)(
    input::InputEPED;
    sigma_threshold::Real=0.05,
    extrapolation_threshold::Real=0.0,
    only_powerlaw::Bool=false,
    warn_nn_train_bounds::Bool=false)
    return ensemble_uncertainty(pedmodel, input; sigma_threshold, extrapolation_threshold, only_powerlaw, warn_nn_train_bounds)
end

"""
    pedestal_width(βpol_ped::Real)

EPED1 analytic pedestal width law `w_ped = 0.076·√βₚ,ped` (half width, as a fraction of ψ_N)

The `EPED1NNensemble` model predicts the pedestal height only: the width of an EPED solution comes
out of this analytic law (R²=1.0, MAPE 0.05% on the 1477 EPED runs of the ITER scan that the model
was trained on), so there is nothing left for a network to learn.
"""
pedestal_width(βpol_ped::Real) = 0.076 * sqrt(βpol_ped)

#= ========================================================= =#
#  the pedestal of a solution, for any of the EPEDNN models
#= ========================================================= =#
# These make it possible to run either model, and get the pedestal out of its solution, without
# having to know which model it was.
"""
    run_epednn(pedmodel::EPEDmodel, input_eped::InputEPED; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=false)

Run `pedmodel` on `input_eped`: returns a `PedestalSolution` for a `EPED1NNmodel`, and the height
with its uncertainty for a `EPED1NNensemble`

Pass the solution to `pedestal_height` and `pedestal_width` to get the pedestal out of it
"""
function run_epednn(pedmodel::EPED1NNmodel, input_eped::InputEPED; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=false)
    return pedmodel(input_eped; only_powerlaw, warn_nn_train_bounds)
end

function run_epednn(pedmodel::EPED1NNensemble, input_eped::InputEPED; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=false)
    # `only_powerlaw` does not apply to an ensemble: dropping the network correction would also drop
    # the disagreement between the networks, which is what the uncertainty of the prediction is made of
    return pedmodel(input_eped; warn_nn_train_bounds)
end

"""
    pedestal_height(pedmodel::EPEDmodel, input_eped::InputEPED, sol)

Pedestal pressure [MPa] of the solution `sol` that `pedmodel` returned for `input_eped`, and the
uncertainty of that pressure as a fraction of itself

Warns (once) when the query point is outside of the distribution that the model was trained on
"""
function pedestal_height(pedmodel::EPED1NNmodel, input_eped::InputEPED, sol::PedestalSolution)
    # a single network carries no uncertainty of its own: all that is known about its prediction is
    # how far the query point sits outside of the training bounds
    extrapolation = extrapolation_distance(pedmodel, input_eped)
    if extrapolation.max_distance > 0.0
        details = join(["$name=$(round(distance; digits=2))" for (name, distance) in extrapolation.per_input if distance > 0.0], ", ")
        @warn "EPED-NN extrapolation: max_distance=$(round(extrapolation.max_distance; digits=2)) on $(extrapolation.worst_input) [$details]" maxlog = 1
    end
    return sol.pressure.GH.H, extrapolation.max_distance
end

function pedestal_height(pedmodel::EPED1NNensemble, input_eped::InputEPED, sol::NamedTuple)
    if !sol.in_distribution
        @warn "EPED-NN ensemble out of distribution: σ=$(round(sol.sigma_frac*100; digits=1))% of the height, extrapolation=$(round(sol.extrapolation; digits=2))" maxlog = 1
    end
    return sol.height, sol.sigma_frac_combined
end

"""
    pedestal_width(pedmodel::EPEDmodel, sol, βpol_ped::Real)

Pedestal width of the solution `sol` (EPED definition: 1/2 width as a fraction of psi_norm)

`βpol_ped` is the poloidal beta of the pedestal pressure, and is only used by the models that predict
the pedestal height alone: those take their width from the analytic EPED1 width law
"""
function pedestal_width(pedmodel::EPED1NNmodel, sol::PedestalSolution, βpol_ped::Real)
    return sol.width.GH.H
end

function pedestal_width(pedmodel::EPED1NNensemble, sol::NamedTuple, βpol_ped::Real)
    return pedestal_width(βpol_ped)
end

export ensemble_predict, ensemble_uncertainty, pedestal_height, pedestal_width

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
    input_vector(pedmodel::EPEDmodel, input::InputEPED)

Vector of the inputs that `pedmodel` takes, in the order that the model expects them (`pedmodel.xnames`)
"""
function input_vector(pedmodel::EPEDmodel, input::InputEPED)
    return [getproperty(input, Symbol(name)) for name in pedmodel.xnames]
end

# An input that sits exactly on a training bound can land a floating point epsilon outside of it
# (a 15 MA ITER against training data that tops out at ip = 15.0, say). That is not extrapolation:
# distances below this fraction of the training range are taken to be zero.
const EXTRAPOLATION_TOLERANCE = 1e-6

"""
    extrapolation_distance(pedmodel::EPEDmodel, x::AbstractVector{<:Real})

Compute a normalized extrapolation distance for each input relative to the training bounds.

Returns a named tuple with:
  - `per_input`: Dict mapping input name to its normalized distance (0 = within bounds)
  - `max_distance`: worst-case distance across all inputs
  - `worst_input`: name of the input furthest outside bounds

Distance is measured as fraction of the training range: `(x - bound) / (bound_max - bound_min)`.
A value of 0.5 means the input is half a training-range-width outside the bounds. Distances below
`EXTRAPOLATION_TOLERANCE` are reported as 0, so that inputs sitting exactly on a training bound do
not register as extrapolation.
"""
function extrapolation_distance(pedmodel::EPEDmodel, x::AbstractVector{<:Real})
    per_input = Dict{String,Float64}()
    max_dist = 0.0
    worst = ""
    for ix in eachindex(x)
        xmin = pedmodel.xbounds[ix, 1]
        xmax = pedmodel.xbounds[ix, 2]
        range_ix = xmax - xmin
        if range_ix <= 0.0
            continue
        end
        dist = max(0.0, (xmin - x[ix]) / range_ix, (x[ix] - xmax) / range_ix)
        if dist < EXTRAPOLATION_TOLERANCE
            dist = 0.0
        end
        per_input[pedmodel.xnames[ix]] = dist
        if dist > max_dist
            max_dist = dist
            worst = pedmodel.xnames[ix]
        end
    end
    return (per_input=per_input, max_distance=max_dist, worst_input=worst)
end

function extrapolation_distance(pedmodel::EPEDmodel, input::InputEPED)
    return extrapolation_distance(pedmodel, input_vector(pedmodel, input))
end

export extrapolation_distance

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
