#=
interface.jl

    Provides a collection of interface tools for working with dynamic factor 
    models, such as estimation (w/ and w/o penalization), simulation, 
    forecasting, and variance decomposition. 

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/09/29
=#

"""
    DynamicFactorModel(y, R, μ, ε) -> model

Construct a dynamic factor model with data `y`, mean specification `μ`, error
model `ε`, `R` dynamic factors.
"""
function DynamicFactorModel(
    y::AbstractMatrix,
    R::Integer,
    μ::AbstractMeanSpecification,
    ε::AbstractErrorModel
)
    (n, T) = size(y)

    # check model specification
    R > n && throw(ArgumentError("R must be less than or equal to n."))

    # instantiate factor process
    Λ = zeros(n, R)
    F = FactorProcess(
        Diagonal{Float64}(undef, R),
        Matrix{Float64}(undef, R, T),
        MvNormal(zeros(R), I(R))
    )

    return DynamicFactorModel(y, μ, ε, Λ, F)
end

"""
    DynamicFactorModel(dims, μ, ε) -> model

Construct a dynamic factor model with dimensions `dims` (n, T, R), mean
specification `μ`, and error model `ε`.
"""
function DynamicFactorModel(
    dims::Dims,
    μ::AbstractMeanSpecification,
    ε::AbstractErrorModel
)
    (n, T, R) = dims

    # check model specification
    R > n && throw(ArgumentError("R must be less than or equal to n."))

    # instantiate factor process
    Λ = zeros(n, R)
    F = FactorProcess(
        Diagonal{Float64}(undef, R),
        Matrix{Float64}(undef, R, T),
        MvNormal(zeros(R), I(R))
    )

    return DynamicFactorModel(Matrix{Float64}(undef, n, T), μ, ε, Λ, F)
end

"""
    Exogenous(X, n) -> μ

Construct a mean specification with exogenous regressors `X` and `n` time
series.
"""
Exogenous(X::AbstractMatrix, n::Integer) = Exogenous(X, similar(X, (n, size(X, 1))))

"""
    Simple(n, T) -> ε

Construct a simple error model with `n` time series and `T` observations.
"""
Simple(n::Integer, T::Integer) = Simple(Array{Float64}(undef, n, T), MvNormal(Zeros(n), I(n)))

"""
    SpatialAutoregression(n, T, W, spatial=:homo) -> ε

Construct a spatial autoregression error model with `n` time series, `T`
observations, spatial weight matrix `W`, and spatial dependence of type
`spatial`.
"""
function SpatialAutoregression(n::Integer, T::Integer, W::AbstractMatrix, spatial::Symbol=:homo)
    # spatial dependence
    ρ_max = max(inv(opnorm(W, 1)), inv(opnorm(W, Inf)))
    if spatial == :homo
        ρ = zeros(1)
    elseif spatial == :hetero
        ρ = zeros(n)
    else
        throw(ArgumentError("spatial dependence type $spatial not supported."))
    end

    return SpatialAutoregression(Array{Float64}(undef, n, T), MvNormal(Zeros(n), I(n)), ρ, ρ_max, W)
end

"""
    SpatialMovingAverage(n, T, W, spatial=:homo) -> ε

Construct a spatial moving average error model with `n` time series, `T`
observations, spatial weight matrix `W`, and spatial dependence of type
`spatial`.
"""
function SpatialMovingAverage(n::Integer, T::Integer, W::AbstractMatrix, spatial::Symbol=:homo)
    # spatial dependence
    if spatial == :homo
        ρ = zeros(1)
    elseif spatial == :hetero
        ρ = zeros(n)
    else
        throw(ArgumentError("spatial dependence type $spatial not supported."))
    end

    return SpatialMovingAverage(Array{Float64}(undef, n, T), MvNormal(Zeros(n), I(n)), ρ, W)
end

