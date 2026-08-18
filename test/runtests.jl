using EPEDNN
using Test

# EPED1NNmodel inputs: a, betan, bt, delta, ip, kappa, m, neped, r, zeffped
const ITER_15MA = (a=2.0, betan=2.0, bt=5.3, delta=0.49, ip=15.0, kappa=1.85, m=2.5, neped=7.0, r=6.2, zeffped=1.5)
const DIIID_LIKE = (a=0.6, betan=2.0, bt=2.0, delta=0.30, ip=1.5, kappa=1.8, m=2.0, neped=4.0, r=1.70, zeffped=1.8)
const JET_LIKE = (a=0.9, betan=2.0, bt=2.5, delta=0.30, ip=2.5, kappa=1.7, m=2.0, neped=5.0, r=2.90, zeffped=1.8)

function input_eped(point; kw...)
    input = EPEDNN.InputEPED()
    for (field, value) in pairs(merge(point, kw))
        setproperty!(input, field, value)
    end
    return input
end

@testset "EPEDNN.jl" begin

    epedmod = EPEDNN.loadmodelonce("EPED1NNmodel.bson")
    ensemble = EPEDNN.loadmodelonce("EPED1NNensemble.bson")

    @testset "EPED1NNmodel" begin
        @test epedmod isa EPEDNN.EPED1NNmodel
        @test epedmod.xnames == ["a", "betan", "bt", "delta", "ip", "kappa", "m", "neped", "r", "zeffped"]
        @test length(epedmod.ynames) == 18

        # regression values of the shipped model (10 inputs, 18 outputs)
        for (point, p_GH_H, p_H_H, w_GH_H, powerlaw_p_GH_H) in (
            (ITER_15MA, 0.07717931845709872, 0.07902655299181356, 0.03263137598834528, 0.07414049323424252),
            (DIIID_LIKE, 0.010140433798324673, 0.009509841209859787, 0.03431218216342435, 0.010317751713337635),
            (JET_LIKE, 0.013637552682254207, 0.013187359519673682, 0.0349190373841225, 0.01328473979140864))
            sol = epedmod(input_eped(point); warn_nn_train_bounds=false)
            @test sol.pressure.GH.H ≈ p_GH_H rtol = 1e-12
            @test sol.pressure.H.H ≈ p_H_H rtol = 1e-12
            @test sol.width.GH.H ≈ w_GH_H rtol = 1e-12
            y = EPEDNN.pedestal_array(epedmod, EPEDNN.input_vector(epedmod, input_eped(point)); only_powerlaw=true, warn_nn_train_bounds=false)
            @test y[1] ≈ powerlaw_p_GH_H rtol = 1e-12
        end
    end

    @testset "EPED1NNensemble" begin
        @test ensemble isa EPEDNN.EPED1NNensemble
        @test ensemble.xnames == ["a", "betan", "bt", "delta", "ip", "kappa", "m", "neped", "nesep_ratio", "r", "tesep", "zeffped"]
        @test ensemble.ynames == ["OUT_p_E1_dmagH_sol0"]
        @test length(ensemble.fluxmodels) == 5

        # the ensemble is trained in PyTorch: these are the predictions of the PyTorch model, which
        # the Julia model must reproduce to machine precision (see scripts/pytorch_ensemble_to_bson.jl)
        for (point, kw, height, sigma) in (
            (ITER_15MA, (;), 0.0780282222986313, 0.0005203043630241124),
            (ITER_15MA, (nesep_ratio=0.55, tesep=200.0), 0.09557675009989561, 0.000598272917773384),
            (DIIID_LIKE, (;), 0.010733736463853543, 0.00029870202898008355),
            (JET_LIKE, (;), 0.01281946568022067, 0.0002897405318576672))
            prediction = EPEDNN.ensemble_predict(ensemble, input_eped(point; kw...))
            @test prediction.mean ≈ height rtol = 1e-10
            @test prediction.std ≈ sigma rtol = 1e-10
        end

        # power law only (no network correction): all members fall back onto the same power law
        x = EPEDNN.input_vector(ensemble, input_eped(ITER_15MA))
        powerlaw = EPEDNN.ensemble_predict(ensemble, x; only_powerlaw=true)
        @test powerlaw.mean ≈ 0.07530872382305742 rtol = 1e-10
        @test powerlaw.std == 0.0

        # the members of the ensemble are what the mean and the standard deviation are taken over
        members = EPEDNN.pedestal_members(ensemble, x)
        @test size(members) == (1, 5)
        @test sum(members) / length(members) ≈ EPEDNN.ensemble_predict(ensemble, x).mean
        @test EPEDNN.pedestal_array(ensemble, x) ≈ vec(sum(members; dims=2)) ./ 5
        @test EPEDNN.pedestal_array(ensemble, hcat(x, x)) ≈ hcat(EPEDNN.pedestal_array(ensemble, x), EPEDNN.pedestal_array(ensemble, x))

        # in distribution: small ensemble spread and within the training bounds
        uncertainty = EPEDNN.ensemble_uncertainty(ensemble, input_eped(ITER_15MA))
        @test uncertainty.height ≈ 0.0780282222986313 rtol = 1e-10
        @test uncertainty.sigma_frac ≈ uncertainty.sigma / uncertainty.height
        @test uncertainty.extrapolation == 0.0
        @test uncertainty.sigma_frac < 0.01
        @test uncertainty.in_distribution

        # out of distribution: Ip well beyond the training bounds
        uncertainty = EPEDNN.ensemble_uncertainty(ensemble, input_eped(ITER_15MA; ip=25.0))
        @test uncertainty.height ≈ 0.10550315950598824 rtol = 1e-10
        @test uncertainty.extrapolation > 0.5
        @test uncertainty.sigma_frac_combined == uncertainty.extrapolation
        @test !uncertainty.in_distribution
        @test EPEDNN.extrapolation_distance(ensemble, input_eped(ITER_15MA; ip=25.0)).worst_input == "ip"

        # an input sitting exactly on a training bound (or a floating point epsilon past it) is not
        # extrapolation: ip of a 15 MA ITER is exactly the maximum ip of the training set
        ip_max = ensemble.xbounds[findfirst(==("ip"), ensemble.xnames), 2]
        @test ip_max == 15.0
        @test EPEDNN.extrapolation_distance(ensemble, input_eped(ITER_15MA; ip=nextfloat(ip_max))).max_distance == 0.0
        @test EPEDNN.ensemble_uncertainty(ensemble, input_eped(ITER_15MA; ip=ip_max * (1 + 1e-9))).in_distribution

        # how far outside of the box still counts as in distribution is tunable
        outside = input_eped(ITER_15MA; ip=1.05 * ip_max)
        @test !EPEDNN.ensemble_uncertainty(ensemble, outside).in_distribution
        @test EPEDNN.ensemble_uncertainty(ensemble, outside; extrapolation_threshold=0.1).in_distribution

        # calling the model is the same as asking for its uncertainty
        @test ensemble(input_eped(ITER_15MA)) == EPEDNN.ensemble_uncertainty(ensemble, input_eped(ITER_15MA))
        @test ensemble(x) == EPEDNN.ensemble_uncertainty(ensemble, x)
    end

    @testset "EPED1NNmodel vs EPED1NNensemble" begin
        # both models predict the same quantity (omega-star `dmagH` pedestal pressure, first solution)
        # and must agree where they are both in distribution: conventional tokamaks, not ITER scale
        # (the ensemble was trained with ITER scale data that the single network never saw)
        for point in (DIIID_LIKE, JET_LIKE)
            single = epedmod(input_eped(point); warn_nn_train_bounds=false).pressure.H.H
            ens = EPEDNN.ensemble_predict(ensemble, input_eped(point)).mean
            @test abs(ens - single) / single < 0.20
        end
    end

    @testset "InputEPED" begin
        input = EPEDNN.InputEPED()
        # the two extra inputs of the ensemble default to the values of the EPED1NNmodel training set
        @test input.nesep_ratio == 0.25
        @test input.tesep == 75.0
        @test EPEDNN.input_vector(epedmod, input_eped(ITER_15MA)) == [2.0, 2.0, 5.3, 0.49, 15.0, 1.85, 2.5, 7.0, 6.2, 1.5]
        @test EPEDNN.input_vector(ensemble, input_eped(ITER_15MA)) == [2.0, 2.0, 5.3, 0.49, 15.0, 1.85, 2.5, 7.0, 0.25, 6.2, 75.0, 1.5]
    end

    @testset "pedestal of a solution (model agnostic)" begin
        # run either model and get the pedestal out of its solution, without knowing which model it
        # is: this is what FUSE uses to be able to switch between them
        βpol_ped = 0.19 # ITER 15MA
        input = input_eped(ITER_15MA)
        for pedmodel in (epedmod, ensemble)
            sol = EPEDNN.run_epednn(pedmodel, input)
            height, sigma_frac = EPEDNN.pedestal_height(pedmodel, input, sol)
            width = EPEDNN.pedestal_width(pedmodel, sol, βpol_ped)
            @test 0.05 < height < 0.10   # MPa
            @test 0.0 <= sigma_frac < 0.05
            @test 0.02 < width < 0.05    # psi_norm
        end

        # the single network takes its width from the network itself, the ensemble from the EPED1 law
        @test EPEDNN.pedestal_width(epedmod, EPEDNN.run_epednn(epedmod, input), βpol_ped) ==
              EPEDNN.run_epednn(epedmod, input).width.GH.H
        @test EPEDNN.pedestal_width(ensemble, EPEDNN.run_epednn(ensemble, input), βpol_ped) == EPEDNN.pedestal_width(βpol_ped)
        @test EPEDNN.pedestal_height(ensemble, input, EPEDNN.run_epednn(ensemble, input))[1] ≈ 0.0780282222986313 rtol = 1e-10

        # `only_powerlaw` drops the network correction of a single network, but is a no-op on an
        # ensemble (it would leave it without the disagreement that its uncertainty is made of)
        @test EPEDNN.run_epednn(epedmod, input; only_powerlaw=true).pressure.GH.H !=
              EPEDNN.run_epednn(epedmod, input).pressure.GH.H
        @test EPEDNN.run_epednn(ensemble, input; only_powerlaw=true) == EPEDNN.run_epednn(ensemble, input)

        # run_epednn(input) loads the model it names
        @test EPEDNN.run_epednn(input; warn_nn_train_bounds=false).pressure.GH.H == EPEDNN.run_epednn(epedmod, input).pressure.GH.H
        @test EPEDNN.run_epednn(input; model_filename="EPED1NNensemble.bson", warn_nn_train_bounds=false).height ==
              EPEDNN.run_epednn(ensemble, input).height
    end

    @testset "pedestal_width" begin
        # EPED1 analytic width law that goes with the height-only ensemble
        @test EPEDNN.pedestal_width(0.0) == 0.0
        @test EPEDNN.pedestal_width(1.0) == 0.076
        @test EPEDNN.pedestal_width(4.0) == 2 * EPEDNN.pedestal_width(1.0)
    end

    @testset "save/load" begin
        datadir = joinpath(dirname(dirname(pathof(EPEDNN))), "data")
        if (uperm(datadir) & 0x02) != 0 # skip when the package is installed read only
            for (model, x) in ((epedmod, EPEDNN.input_vector(epedmod, input_eped(ITER_15MA))),
                (ensemble, EPEDNN.input_vector(ensemble, input_eped(ITER_15MA))))
                filename = "test_roundtrip.bson"
                fullpath = EPEDNN.savemodel(model, filename)
                try
                    reloaded = EPEDNN.loadmodel(filename)
                    @test typeof(reloaded) == typeof(model)
                    @test EPEDNN.pedestal_array(reloaded, x; warn_nn_train_bounds=false) ==
                          EPEDNN.pedestal_array(model, x; warn_nn_train_bounds=false)
                finally
                    rm(fullpath; force=true)
                end
            end
        end
    end

end
