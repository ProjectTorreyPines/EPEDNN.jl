# EPEDNN.jl

Runs the EPEDNN pedestal model.

## Models

Both models are shipped in `data/` and are loaded with `EPEDNN.loadmodelonce(<filename>)`:

| model | inputs | outputs |
| --- | --- | --- |
| `EPED1NNmodel.bson` | 10: `a`, `betan`, `bt`, `delta`, `ip`, `kappa`, `m`, `neped`, `r`, `zeffped` | 18: pedestal pressure and width, for 3 diamagnetic models × 3 solutions |
| `EPED1NNensemble.bson` | 12: the same, plus `nesep_ratio` (`ne_sep/ne_ped`) and `tesep` | 1: pedestal pressure (`dmagH` diamagnetic model, first solution), **with an uncertainty** |

Both are a power law fitted in log space plus a neural-network correction on top of it, so that
predictions revert to the physics-shaped power-law trend where the training data runs out.

```julia
input = EPEDNN.InputEPED()
input.a, input.betan, input.bt, input.delta, input.ip = 2.0, 2.0, 5.3, 0.49, 15.0
input.kappa, input.m, input.neped, input.r, input.zeffped = 1.85, 2.5, 7.0, 6.2, 1.5
input.nesep_ratio, input.tesep = 0.25, 75.0 # only used by the ensemble

epedmod = EPEDNN.loadmodelonce("EPED1NNmodel.bson")
solution = epedmod(input) # pedestal pressure [MPa] and width [ψ_N] of each solution
solution.pressure.GH.H, solution.width.GH.H

ensemble = EPEDNN.loadmodelonce("EPED1NNensemble.bson")
ensemble(input) # (height, sigma, sigma_frac, extrapolation, sigma_frac_combined, in_distribution)
```

The same three calls work for either model, so that a code can switch between them without knowing
which one it is holding:

```julia
sol = EPEDNN.run_epednn(pedmodel, input)
pressure, uncertainty = EPEDNN.pedestal_height(pedmodel, input, sol) # [MPa], and as a fraction of it
width = EPEDNN.pedestal_width(pedmodel, sol, βpol_ped)               # [ψ_N] 1/2 width
```

`βpol_ped` is the poloidal beta of the pedestal pressure (`IMAS.pedestal_poloidal_beta`), and is only
used by the models that predict the height alone.

### `EPED1NNensemble`: uncertainty and out of distribution detection

`EPED1NNensemble` is a deep ensemble: several networks trained independently on the same data, whose
disagreement measures the uncertainty of the prediction. `ensemble_uncertainty` combines it with the
distance from the training bounds (`extrapolation_distance`) into an `in_distribution` flag. The two
are complementary: the ensemble spread catches points that are within the bounds of each individual
input but away from the (thin) manifold where the training data actually lives, while the distance
catches points far outside of the bounds, where all the members fall back onto the same power law and
so agree with each other for the wrong reason.

The shipped ensemble predicts the pedestal **height** only: the width of an EPED solution follows the
analytic EPED1 width law `w_ped = 0.076·√βₚ,ped` (`pedestal_width`), so there is nothing left for a
network to learn. It was trained on the multi-machine EPED database used for `EPED1NNmodel` plus new
ITER-scale EPED runs, which extend the training set to low pedestal density and to separatrix
conditions (`nesep_ratio`, `tesep`) that the original database sampled at a single value.

`scripts/pytorch_ensemble_to_bson.jl` packages a trained ensemble into the `data/*.bson` shipped here.

## Online documentation
For more details, see the [online documentation](https://projecttorreypines.github.io/EPEDNN.jl/dev).

![Docs](https://github.com/ProjectTorreyPines/EPEDNN.jl/actions/workflows/make_docs.yml/badge.svg)
