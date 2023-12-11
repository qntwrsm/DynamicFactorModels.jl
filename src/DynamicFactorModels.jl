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
    Random,
    FillArrays,
    Distributions,
    MultivariateStats

import Base: show, size, copy
import Statistics: mean, var, cov

export
    # constructors
    DynamicFactorModel,                                     # main
    ZeroMean, Exogenous,                                    # mean specifications
    Simple, SpatialAutoregression, SpatialMovingAverage,    # error models 

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
    fit!

include("types.jl")
include("show.jl")
include("interface.jl")
include("utilities.jl")
include("fit/utilities.jl")
include("fit/solver.jl")

end
