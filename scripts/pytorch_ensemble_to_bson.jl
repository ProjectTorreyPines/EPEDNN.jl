#!/usr/bin/env julia
#
# Package a PyTorch-trained EPED-NN deep ensemble as an EPEDNN.jl `EPED1NNensemble` BSON model.
#
# Usage:
#
#     julia scripts/pytorch_ensemble_to_bson.jl <trained_models_dir> [output_filename]
#
# where `<trained_models_dir>` is what the training script exports:
#
#     preprocessing.json      normalizations, power-law coefficients, training bounds, architecture
#     model_<k>_weights.json  weights/biases of member `k` (PyTorch `named_parameters()` dump)
#
# and `[output_filename]` (default `EPED1NNensemble.bson`) is written to `EPEDNN.jl/data/`.
#
# The conversion is a repackaging only: preprocessing, power law and network weights are the ones
# that came out of the training, and are checked to reproduce the PyTorch predictions.
#
# NOTE: the training pipeline itself matches the (Flux) pipeline of the original EPED1NN model:
# `delta+1`, `abs()` of the inputs, pressure divided by `neped`, `sqrt()` of the outputs, power-law
# baseline fitted in log space and network trained on the residual of the power law, z-scored.
# The only formatting differences handled here are:
#   - the power law is fitted in natural log, EPEDNN.jl evaluates it in log10 => rescale the constant
#   - the training bounds are stored after preprocessing => undo the `delta+1` shift, since
#     EPEDNN.jl checks the bounds against the original (un-preprocessed) inputs

import Pkg
Pkg.activate(; temp=true)
Pkg.develop(; path=dirname(@__DIR__))
Pkg.add(["JSON", "Flux", "BSON"])

import JSON
import Flux
import Dates
import EPEDNN

function pytorch_ensemble_to_bson(dirname::String, filename::String="EPED1NNensemble.bson")
    pp = JSON.parsefile(joinpath(dirname, "preprocessing.json"))

    xnames = String.(pp["input_cols"])
    # the trained target is the pedestal pressure of the `diamagnetic_model` at the first (lowest
    # pedestal beta) crossing of the marginal stability boundary => name it like the EPED1NN outputs
    ynames = [name == "p_E1" ? "OUT_p_E1_dmag$(pp["diamagnetic_model"])_sol0" : name for name in pp["output_cols"]]

    # power law: exp(p[1] + Σ p[i+1] * log(x[i])) => 10^(p[1]/log(10) + Σ p[i+1] * log10(x[i]))
    yp = rows_to_matrix(pp["yp"])
    yp[:, 1] ./= log(10.0)

    # training bounds are stored on the preprocessed inputs, EPEDNN.jl wants them on the original ones
    xbounds = rows_to_matrix(pp["xbounds"])
    xbounds[findfirst(==("delta"), xnames), :] .-= 1.0

    fluxmodels = [load_pytorch_chain(joinpath(dirname, "model_$(k)_weights.json")) for k in 0:pp["n_ensemble"]-1]

    ensemble = EPEDNN.EPED1NNensemble(
        fluxmodels,
        "delta_ne_sqrt_power_ensemble",
        Dates.DateTime(pp["timestamp"], Dates.dateformat"yyyymmdd_HHMMSS"),
        xnames,
        ynames,
        Float64.(pp["xm"]),
        Float64.(pp["xs"]),
        Float64.(pp["ym"]),
        Float64.(pp["ys"]),
        xbounds,
        rows_to_matrix(pp["ybounds"]),
        yp)

    fullpath = EPEDNN.savemodel(ensemble, filename)
    println("$(length(fluxmodels)) x $(pp["n_inputs"])=>$(pp["n_outputs"]) networks saved to $(fullpath)")
    return fullpath
end

"""
    load_pytorch_chain(filename::String)

Build a `Flux.Chain` from a PyTorch `nn.Sequential` of `nn.Linear`/`nn.GELU` layers dumped as JSON

PyTorch stores weights as `[out][in]` (row major), Flux wants `(out, in)`. The activation is the
exact (error function) GELU, which is what `torch.nn.GELU()` uses.
"""
function load_pytorch_chain(filename::String)
    weights = JSON.parsefile(filename)
    ilayers = sort([parse(Int, split(key, ".")[2]) for key in keys(weights) if endswith(key, ".weight")])
    layers = Flux.Dense[]
    for (k, ilayer) in enumerate(ilayers)
        W = rows_to_matrix(weights["net.$(ilayer).weight"])
        b = Float64.(weights["net.$(ilayer).bias"])
        push!(layers, Flux.Dense(W, b, k == length(ilayers) ? identity : EPEDNN.gelu_erf))
    end
    return Flux.Chain(layers...)
end

rows_to_matrix(rows) = reduce(vcat, [permutedims(Float64.(row)) for row in rows])

if abspath(PROGRAM_FILE) == @__FILE__
    @assert length(ARGS) >= 1 "usage: julia scripts/pytorch_ensemble_to_bson.jl <trained_models_dir> [output_filename]"
    pytorch_ensemble_to_bson(ARGS...)
end
