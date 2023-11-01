#=
types.jl

    Provides a collection of types for working with dynamic factor models, such 
    as mean specifications and error distributions, as well as the main 
    dynamic factor model itself. 

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/09/15
=#

# mean specifications
"""
    AbstractMeanSpecification

Abstract type for mean specification. 
"""
abstract type AbstractMeanSpecification end

"""
    ZeroMean <: AbstractMeanSpecification

Mean specification with constant zero mean of `type`, used purely for dispatch.
"""
struct ZeroMean{T<:DataType} <: AbstractMeanSpecification
    type::T
end

"""
    Exogenous <: AbstractMeanSpecification

Mean specification with exogenous regressors `X` and slopes `β`.
"""
struct Exogenous{
    Reg<:AbstractMatrix,
    Slopes<:AbstractMatrix
} <: AbstractMeanSpecification
    X::Reg
    β::Slopes
    function Exogenous(X::AbstractMatrix, β::AbstractMatrix)
        size(X, 1) == size(β, 2) || throw(DimensionMismatch("βX must be defined."))

        return new{typeof(X), typeof(β)}(X, β)
    end
end

# methods
slopes(μ::Exogenous) = μ.β
regressors(μ::Exogenous) = μ.X
mean(μ::ZeroMean) = zero(μ.type)
mean(μ::Exogenous) = slopes(μ) * regressors(μ)

# error models
"""
    AbstractErrorModel

Abstract type for error model.
"""
abstract type AbstractErrorModel end

"""
    Simple <: AbstractErrorModel

Simple error model with errors `ε` and multivariate normal distribution `dist`.
"""
struct Simple{Error<:AbstractMatrix, Dist<:AbstractMvNormal} <: AbstractErrorModel
    ε::Error
    dist::Dist
    function Simple(ε::AbstractMatrix, dist::AbstractMvNormal)
        size(ε, 1) == length(mean(dist)) || throw(DimensionMismatch("ε and mean of dist must have the same number of rows."))
        size(ε, 1) == size(cov(dist), 1) || throw(DimensionMismatch("ε and covariance of dist must have the same number of rows."))

        return new{typeof(ε), typeof(dist)}(ε, dist)
    end
end

"""
    SpatialAutoregression <: AbstractErrorModel

Spatial autoregressive error model with errors `ε`, multivariate normal
distribution `dist`, spatial dependence `ρ`, maximum spatial dependence `ρ_max`,
and spatial weights `W`.
"""
struct SpatialAutoregression{
    Error<:AbstractMatrix,
    Dist<:AbstractMvNormal,
    SpatialDep<:AbstractVector,
    SpatialMax<:Real,
    SpatialWeights<:AbstractMatrix
} <: AbstractErrorModel 
    ε::Error
    dist::Dist
    ρ::SpatialDep
    ρ_max::SpatialMax
    W::SpatialWeights
    function SpatialAutoregression(
        ε::AbstractMatrix,
        dist::AbstractMvNormal,
        ρ::AbstractVector,
        ρ_max::Real,
        W::AbstractMatrix
    )
        size(ε, 1) == length(mean(dist)) || throw(DimensionMismatch("ε and mean of dist must have the same number of rows."))
        size(ε, 1) == size(cov(dist), 1) || throw(DimensionMismatch("ε and covariance of dist must have the same number of rows."))
        size(ε, 1) == size(W, 1) || throw(DimensionMismatch("ε and W must have the same number of rows."))
        (length(ρ) == size(W, 1) || length(ρ) == 1) || throw(DimensionMismatch("ρ and W must have the same number of rows or ρ must be a single element vector."))
        all(ρi -> abs(ρi) < ρ_max, ρ)  || throw(DimensionMismatch("|ρ| < ρ_max."))
        size(W, 1) == size(W, 2) || throw(DimensionMismatch("W must be square."))

        return new{typeof(ε), typeof(dist), typeof(ρ), typeof(ρ_max), typeof(W)}(ε, dist, ρ, ρ_max, W)
    end
end

"""
    SpatialMovingAverage <: AbstractErrorModel

Spatial moving average error model with errors `ε`, multivariate normal
distribution `dist`, spatial dependence `ρ`, and spatial weights `W`.
"""
struct SpatialMovingAverage{
    Error<:AbstractMatrix,
    Dist<:AbstractMvNormal,
    SpatialDep<:AbstractVector,
    SpatialWeights<:AbstractMatrix
} <: AbstractErrorModel
    ε::Error
    dist::Dist
    ρ::SpatialDep
    W::SpatialWeights
    function SpatialMovingAverage(
        ε::AbstractMatrix,
        dist::AbstractMvNormal,
        ρ::AbstractVector,
        W::AbstractMatrix
    )
        size(ε, 1) == length(mean(dist)) || throw(DimensionMismatch("ε and mean of dist must have the same number of rows."))
        size(ε, 1) == size(cov(dist), 1) || throw(DimensionMismatch("ε and covariance of dist must have the same number of rows."))
        (length(ρ) == size(W, 1) || length(ρ) == 1) || throw(DimensionMismatch("ρ and W must have the same number of rows or ρ must be a single element vector."))
        size(W, 1) == size(W, 2) || throw(DimensionMismatch("W must be square."))

        return new{typeof(ε), typeof(dist), typeof(ρ), typeof(W)}(ε, dist, ρ, W)
    end
end

# methods
resid(ε::AbstractErrorModel) = ε.ε
dist(ε::AbstractErrorModel) = ε.dist
mean(ε::AbstractErrorModel) = mean(dist(ε))
cov(ε::AbstractErrorModel) = cov(dist(ε))
var(ε::AbstractErrorModel) = var(dist(ε))
spatial(ε::SpatialAutoregression) = ε.ρ
spatial(ε::SpatialMovingAverage) = ε.ρ
weights(ε::SpatialAutoregression) = ε.W
weights(ε::SpatialMovingAverage) = ε.W
function poly(ε::SpatialAutoregression)
    if length(spatial(ε)) == 1
        return I - spatial(ε) .* weights(ε)
    else
        return I - Diagonal(spatial(ε)) * weights(ε)
    end
end
function poly(ε::SpatialMovingAverage)
    if length(spatial(ε)) == 1
        return I + spatial(ε) .* weights(ε)
    else
        return I + Diagonal(spatial(ε)) * weights(ε)
    end
end

# factor process
struct FactorProcess{
    Dynamics<:AbstractMatrix,
    Factors<:AbstractMatrix,
    Dist<:AbstractMvNormal
}
    ϕ::Dynamics
    f::Factors
    dist::Dist
    function FactorProcess(ϕ::AbstractMatrix, f::AbstractMatrix, dist::AbstractMvNormal)
        size(ϕ, 1) == size(ϕ, 2) || throw(DimensionMismatch("ϕ must be square."))
        size(ϕ, 1) == size(f, 1) || throw(DimensionMismatch("ϕ and f must have the same number of rows."))
        size(f, 1) == length(mean(dist)) || throw(DimensionMismatch("f and mean of dist must have the same size."))
        size(f, 1) == size(cov(dist), 1) || throw(DimensionMismatch("f and covariance of dist must have the same number of rows."))

        return new{typeof(ϕ), typeof(f), typeof(dist)}(ϕ, f, dist)
    end
end

# methods
dynamics(F::FactorProcess) = F.ϕ
factors(F::FactorProcess) = F.f
dist(F::FactorProcess) = F.dist
cov(F::FactorProcess) = cov(dist(F))

# dynamic factor model
"""
    DynamicFactorModel

Dynamic factor model with mean specification `μ`, error model `ε`,
factor loadings `Λ`, and factor process `f`.
"""
struct DynamicFactorModel{
    Data<:AbstractMatrix,
    Mean<:AbstractMeanSpecification,
    Error<:AbstractErrorModel,
    Loadings<:AbstractMatrix,
    Factors<:FactorProcess,
}
    y::Data
    μ::Mean
    ε::Error
    Λ::Loadings
    F::Factors
    function DynamicFactorModel(
        y::AbstractMatrix, 
        μ::AbstractMeanSpecification,
        ε::AbstractErrorModel,
        Λ::AbstractMatrix,
        F::FactorProcess
    )
        size(y, 1) == size(Λ, 1) || throw(DimensionMismatch("y and Λ must have the same number of rows."))
        size(y) == size(resid(ε)) || throw(DimensionMismatch("y and residuals must have the same dimensions."))
        size(y, 2) == size(factors(F), 2) || throw(DimensionMismatch("y and factors must have the same number of observations."))
        size(Λ, 2) == size(factors(F), 1) || throw(DimensionMismatch("multiplication of loadings and factors must be defined."))

        return new{typeof(y), typeof(μ), typeof(ε), typeof(Λ), typeof(F)}(y, μ, ε, Λ, F)
    end
end

# methods
data(model::DynamicFactorModel) = model.y
mean(model::DynamicFactorModel) = model.μ
errors(model::DynamicFactorModel) = model.ε
resid(model::DynamicFactorModel) = resid(errors(model))
loadings(model::DynamicFactorModel) = model.Λ
process(model::DynamicFactorModel) = model.F
factors(model::DynamicFactorModel) = factors(process(model))
dynamics(model::DynamicFactorModel) = dynamics(process(model.F))