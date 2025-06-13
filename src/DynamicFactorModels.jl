#=
DynamicFactorModels.jl

    Provides a collection of tools for working with dynamic factor models, such
    as estimation, w/ and w/o regularization, forecasting, filtering, and
    smoothing.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/09/15
=#

module DynamicFactorModels

using Distributed

using LinearAlgebra
using FillArrays
using Random
using Distributions: MvNormal

using StatsAPI: StatisticalModel

using MultivariateStats: PCA, fit, projection, predict

using IrrationalConstants: log2π

using Distances: evaluate, Chebyshev

using LogExpFunctions: logistic, logit
using Optim
using FiniteDiff
using DifferentiationInterface: AutoFiniteDiff
using ProximalCore
using ProximalOperators: NormL1, NormL21, TotalVariation1D
using ProximalAlgorithms
using TreeParzen

using ProgressMeter
using Logging

using StatsAPI: aic, aicc, bic

import Statistics: mean, var, cov
import StatsAPI: fit!, loglikelihood, dof, nobs
import ProximalOperators: prox!

export
    # constructors
    DynamicFactorModel,                                     # main
    UnrestrictedStationary, UnrestrictedUnitRoot,           # unrestricted factor processes
    NelsonSiegelStationary, NelsonSiegelUnitRoot,           # Nelson-Siegel factor processes
    ZeroMean, Exogenous,                                    # mean specifications
    Simple, SpatialAutoregression, SpatialMovingAverage,    # error models
    NormL21Weighted, NormL1plusL21, TotalVariation1D,       # regularizers

    # interface methods
    ## getters
    data, mean, errors, process, resid, # general
    factors, loadings, dynamics, decay, # factors
    slopes, regressors,                 # mean
    cov, var,                           # distribution
    spatial, weights,                   # spatial

    # simulate
    simulate,

    # fit
    fit!,
    model_tuning_ic!, model_tuning_cv!,
    loglikelihood,
    dof, nobs, nfactors, aic, aicc, bic,
    HP,                                     # re-export from TreeParzen for convenience

    # forecast
    forecast,

    # irf
    irf

include("types.jl")
include("show.jl")
include("interface.jl")
include("utilities.jl")
include("fit/utilities.jl")
include("fit/regularization.jl")
include("fit/solver.jl")

end
