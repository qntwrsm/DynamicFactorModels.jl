#=
DynamicFactorModels.jl

    Provides a collection of tools for working with dynamic factor models, such 
    as estimation, w/ and w/o regularization, forecasting, filtering, and 
    smoothing. 

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/09/15
=#

module DynamicFactorModels

using 
    LinearAlgebra,
    FillArrays,
    Random,
    Distributions

using StatsAPI: StatisticalModel

using MultivariateStats

using IrrationalConstants: log2π

using LogExpFunctions: logistic, logit
using Optim
using ProximalOperators: NormL1, NormL21, TotalVariation1D
using ProximalAlgorithms: FastForwardBackward

using StatsAPI: aic, aicc, bic

import Base: show, size, copy
import Statistics: mean, var, cov
import StatsAPI: params, params!, fit!, loglikelihood, dof, nobs
import ProximalOperators: prox!

export
    # constructors
    DynamicFactorModel,                                     # main
    ZeroMean, Exogenous,                                    # mean specifications
    Simple, SpatialAutoregression, SpatialMovingAverage,    # error models
    NormL21Weighted, NormL1plusL21, TotalVariation1D,       # regularizers  

    # interface methods
    ## getters
    data, mean, errors, process, resid, # general
    factors, loadings, dynamics,        # factors
    slopes, regressors,                 # mean
    cov, var,                           # distribution
    spatial, weights,                   # spatial

    # simulate
    simulate,

    # fit
    fit!,
    loglikelihood,
    dof, nobs, aic, aicc, bic

include("types.jl")
include("show.jl")
include("interface.jl")
include("utilities.jl")
include("fit/utilities.jl")
include("fit/regularization.jl")
include("fit/solver.jl")

end
