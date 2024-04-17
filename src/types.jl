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

Mean specification with constant zero mean of `type` and size `n`, used purely
for dispatch.
"""
struct ZeroMean{T<:DataType} <: AbstractMeanSpecification
    type::T
    n::Integer
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
mean(μ::ZeroMean) = Zeros(μ.type, μ.n)
mean(μ::Exogenous) = slopes(μ) * regressors(μ)
copy(μ::ZeroMean) = ZeroMean(μ.type, μ.n)
copy(μ::Exogenous) = Exogenous(copy(regressors(μ)), copy(slopes(μ)))

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
struct Simple{Error<:AbstractMatrix, Dist<:ZeroMeanDiagNormal} <: AbstractErrorModel
    ε::Error
    dist::Dist
    function Simple(ε::AbstractMatrix, dist::ZeroMeanDiagNormal)
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
    Dist<:ZeroMeanDiagNormal,
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
        dist::ZeroMeanDiagNormal,
        ρ::AbstractVector,
        ρ_max::Real,
        W::AbstractMatrix
    )
        size(ε, 1) == size(cov(dist), 1) || throw(DimensionMismatch("ε and covariance of dist must have the same number of rows."))
        size(ε, 1) == size(W, 1) || throw(DimensionMismatch("ε and W must have the same number of rows."))
        (length(ρ) == size(W, 1) || length(ρ) == 1) || throw(DimensionMismatch("ρ and W must have the same number of rows or ρ must be a single element vector."))
        all(ρi -> abs(ρi) < ρ_max, ρ)  || throw(DomainError("|ρ| < ρ_max."))
        size(W, 1) == size(W, 2) || throw(DimensionMismatch("W must be square."))

        return new{typeof(ε), typeof(dist), typeof(ρ), typeof(ρ_max), typeof(W)}(ε, dist, ρ, ρ_max, W)
    end
end

"""
    SpatialMovingAverage <: AbstractErrorModel

Spatial moving average error model with errors `ε`, multivariate normal
distribution `dist`, spatial dependence `ρ`, maximum spatial dependence `ρ_max`, 
and spatial weights `W`.
"""
struct SpatialMovingAverage{
    Error<:AbstractMatrix,
    Dist<:ZeroMeanDiagNormal,
    SpatialDep<:AbstractVector,
    SpatialMax<:Real,
    SpatialWeights<:AbstractMatrix
} <: AbstractErrorModel
    ε::Error
    dist::Dist
    ρ::SpatialDep
    ρ_max::SpatialMax
    W::SpatialWeights
    function SpatialMovingAverage(
        ε::AbstractMatrix,
        dist::ZeroMeanDiagNormal,
        ρ::AbstractVector,
        ρ_max::Real,
        W::AbstractMatrix
    )
        size(ε, 1) == size(cov(dist), 1) || throw(DimensionMismatch("ε and covariance of dist must have the same number of rows."))
        size(ε, 1) == size(W, 1) || throw(DimensionMismatch("ε and W must have the same number of rows."))
        (length(ρ) == size(W, 1) || length(ρ) == 1) || throw(DimensionMismatch("ρ and W must have the same number of rows or ρ must be a single element vector."))
        all(ρi -> abs(ρi) < ρ_max, ρ)  || throw(DomainError("|ρ| < ρ_max."))
        size(W, 1) == size(W, 2) || throw(DimensionMismatch("W must be square."))

        return new{typeof(ε), typeof(dist), typeof(ρ), typeof(ρ_max), typeof(W)}(ε, dist, ρ, ρ_max, W)
    end
end

# methods
resid(ε::AbstractErrorModel) = ε.ε
dist(ε::AbstractErrorModel) = ε.dist
mean(ε::AbstractErrorModel) = mean(dist(ε))
cov(ε::AbstractErrorModel; full::Bool=false) = Distributions._cov(dist(ε))
function cov(ε::SpatialAutoregression; full::Bool=false)
    if full
        return poly(ε) \ Distributions._cov(dist(ε)) / poly(ε)'
    else
        return Distributions._cov(dist(ε))
    end
end
function cov(ε::SpatialMovingAverage; full::Bool=false)
    if full
        return poly(ε) * Distributions._cov(dist(ε)) * poly(ε)'
    else
        return Distributions._cov(dist(ε))
    end
end
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
"""
    AbstractFactorProcess

Abstract type for factor process. 
"""
abstract type AbstractFactorProcess end

"""
    AbstractUnrestrictedFactorProcess

Abstract type for factor process with unrestricted loadings
"""
abstract type AbstractUnrestrictedFactorProcess <: AbstractFactorProcess end

"""
    AbstractNelsonSiegelFactorProcess

Abstract type for factor process with Nelson-Siegel loadings
"""
abstract type AbstractNelsonSiegelFactorProcess <: AbstractFactorProcess end

"""
    UnrestrictedStationary <: AbstractUrestrictedFactorProcess

Stationary factor process with unrestricted loadings `Λ`, dynamics `ϕ`, factors
`f`, and standard multivariate normal distribution `dist`.
"""
struct UnrestrictedStationary{
    Loadings<:AbstractMatrix, 
    Dynamics<:Diagonal, 
    Factors<:AbstractMatrix,
    Dist<:ZeroMeanIsoNormal
} <: AbstractUnrestrictedFactorProcess
    Λ::Loadings
    ϕ::Dynamics
    f::Factors
    dist::Dist
    function UnrestrictedStationary(Λ::AbstractMatrix, ϕ::Diagonal, f::AbstractMatrix, dist::ZeroMeanIsoNormal)
        size(Λ, 1) >= size(Λ, 2) || throw(ArgumentError("R must be less than or equal to n."))
        size(Λ, 2) == size(f, 1) || throw(DimensionMismatch("multiplication of loadings and factors must be defined."))
        size(ϕ, 1) == size(ϕ, 2) || throw(DimensionMismatch("ϕ must be square."))
        size(ϕ, 1) == size(f, 1) || throw(DimensionMismatch("ϕ and f must have the same number of rows."))
        size(cov(dist), 1) == size(f, 1) || throw(DimensionMismatch("covariance of dist and f must have the same number of rows."))

        return new{typeof(Λ), typeof(ϕ), typeof(f), typeof(dist)}(Λ, ϕ, f, dist)
    end
end

"""
    UnrestrictedUnitRoot <: AbstractUnrestrictedFactorProcess

Unit-root factor process with unrestricted loadings `Λ`, factors `f`, and zero
mean diagonal multivariate normal distribution `dist`.
"""
struct UnrestrictedUnitRoot{
    Loadings<:AbstractMatrix, 
    Factors<:AbstractMatrix, 
    Dist<:ZeroMeanDiagNormal
} <: AbstractUnrestrictedFactorProcess
    Λ::Loadings
    f::Factors
    dist::Dist
    function UnrestrictedUnitRoot(Λ::AbstractMatrix, f::AbstractMatrix, dist::ZeroMeanDiagNormal)
        size(Λ, 1) >= size(Λ, 2) || throw(ArgumentError("R must be less than or equal to n."))
        size(Λ, 2) == size(f, 1) || throw(DimensionMismatch("multiplication of loadings and factors must be defined."))
        size(cov(dist), 1) == size(f, 1) || throw(DimensionMismatch("covariance of dist and f must have the same number of rows."))

        return new{typeof(Λ), typeof(f), typeof(dist)}(Λ, f, dist)
    end
end

"""
    NelsonSiegelStationary <: AbstractNelsonSiegelFactorProcess

Stationary Nelson-Siegel factor process with decay parameter `λ`, dynamics `ϕ`,
factors `f`, and zero mean multivariate normal distribution `dist` for
maturities `τ`.
"""
mutable struct NelsonSiegelStationary{
    Decay<:Real, 
    Maturities<:AbstractVector,
    Dynamics<:AbstractMatrix, 
    Factors<:AbstractMatrix, 
    Dist<:ZeroMeanFullNormal
} <: AbstractNelsonSiegelFactorProcess
    λ::Decay
    τ::Maturities
    ϕ::Dynamics
    f::Factors
    dist::Dist
    function NelsonSiegelStationary(λ::Real, τ::AbstractVector, ϕ::AbstractMatrix, f::AbstractMatrix, dist::ZeroMeanFullNormal)
        λ > 0 || throw(DomainError("λ must be positive"))
        minimum(τ) > 0 || throw(DomainError("maturities must be positive"))
        length(τ) >= 3 || throw(ArgumentError("R must be less than or equal to n."))
        size(ϕ, 1) == size(ϕ, 2) || throw(DimensionMismatch("ϕ must be square."))
        size(f, 1) == 3 || throw(DimensionMismatch("multiplication of loadings and factors must be defined."))
        size(cov(dist), 1) == size(f, 1) || throw(DimensionMismatch("covariance of dist and f must have the same number of rows."))

        return new{typeof(λ), typeof(τ), typeof(ϕ), typeof(f), typeof(dist)}(λ, τ, ϕ, f, dist)
    end
end

"""
    NelsonSiegelUnitRoot <: AbstractNelsonSiegelFactorProcess

Unit-root Nelson-Siegel factor process with decay parameter `λ`, factors `f`,
and zero mean diagonal multivariate normal distribution `dist` for maturities
`τ`.
"""
mutable struct NelsonSiegelUnitRoot{
    Decay<:Real, 
    Maturities<:AbstractVector,
    Factors<:AbstractMatrix, 
    Dist<:ZeroMeanDiagNormal
} <: AbstractNelsonSiegelFactorProcess
    λ::Decay
    τ::Maturities
    f::Factors
    dist::Dist
    function NelsonSiegelUnitRoot(λ::Real, τ::AbstractVector, f::AbstractMatrix, dist::ZeroMeanDiagNormal)
        λ > 0 || throw(DomainError("λ must be positive"))
        minimum(τ) > 0 || throw(DomainError("maturities must be positive"))
        length(τ) >= 3 || throw(ArgumentError("R must be less than or equal to n."))
        size(f, 1) == 3 || throw(DimensionMismatch("multiplication of loadings and factors must be defined."))
        size(cov(dist), 1) == size(f, 1) || throw(DimensionMismatch("covariance of dist and f must have the same number of rows."))

        return new{typeof(λ), typeof(τ), typeof(f), typeof(dist)}(λ, τ, f, dist)
    end
end

# methods
decay(F::AbstractNelsonSiegelFactorProcess) = F.λ
maturities(F::AbstractNelsonSiegelFactorProcess) = F.τ
loadings(F::AbstractUnrestrictedFactorProcess) = F.Λ
function loadings(F::AbstractNelsonSiegelFactorProcess)
    n = length(maturities(F))
    Λ = ones(n, 3)
    for (i, τ) ∈ pairs(maturities(F))
        z = exp(-decay(F) * τ)
        Λ[i,2:end] .= (1 - z) / (decay(F) * τ)
        Λ[i,end] -= z
    end

    return Λ
end
factors(F::AbstractFactorProcess) = F.f
size(F::AbstractFactorProcess) = size(factors(F), 1)
dynamics(F::AbstractFactorProcess) = F.ϕ
dynamics(F::UnrestrictedUnitRoot) = I
dynamics(F::NelsonSiegelUnitRoot) = I
dist(F::AbstractFactorProcess) = F.dist
cov(F::AbstractFactorProcess) = Distributions._cov(dist(F))
function copy(F::UnrestrictedStationary)
    Λ = copy(loadings(F))
    ϕ = copy(dynamics(F))
    f = copy(factors(F))
    type = eltype(dist(F))

    return UnrestrictedStationary(Λ, ϕ, f, MvNormal(Zeros{type}(size(F)), one(type)I))
end
function copy(F::UnrestrictedUnitRoot)
    Λ = copy(loadings(F))
    f = copy(factors(F))
    dist = MvNormal(Diagonal(var(dist(F))))

    return UnrestrictedUnitRoot(Λ, f, dist)
end
function copy(F::NelsonSiegelStationary)
    τ = copy(maturities(F))
    ϕ = copy(dynamics(F))
    f = copy(factors(F))

    return NelsonSiegelStationary(decay(F), τ, ϕ, f, MvNormal(cov(dist)))
end
function copy(F::NelsonSiegelUnitRoot)
    τ = copy(maturities(F))
    f = copy(factors(F))

    return NelsonSiegelUnitRoot(decay(F), τ, f, MvNormal(cov(dist)))
end

# dynamic factor model
"""
    DynamicFactorModel <: StatisticalModel

Dynamic factor model with mean specification `μ`, error model `ε`, and factor
process `F`.

The dynamic factor model is defined as

```math
yₜ = μₜ + Λfₜ + εₜ, εₜ ∼ N(0, Σε),
fₜ = ϕfₜ₋₁ + ηₜ, ηₜ ∼ N(0, Ση),
```

where ``yₜ`` is a ``n × 1`` vector of observations, ``μₜ`` is a ``n × 1`` vector
of time-varying means, ``Λ`` is a ``n × R`` matrix of factor loadings, ``fₜ`` is
a ``R × 1`` vector of factors, and ``εₜ`` is a ``n × 1`` vector of errors. 

When the loading matrix is unrestricted the factors are assumed to be
independent for identification purpose, i.e. the factor process has diagonal
autoregressive dynamics and the disturbances follow a multivariate normal
distribution with diagonal covariance matrix. In the case of a stationary
process the covariance of the error term is an identity matrix (``ηₜ ∼ N(0,
I)``). Moreover, the dynamics of the independent factors are assumed to be
idiosyncratic, i.e. ``ϕᵢ ≠ ϕⱼ`` for ``i ≠ j``.
"""
struct DynamicFactorModel{
    Data<:AbstractMatrix,
    Mean<:AbstractMeanSpecification,
    Error<:AbstractErrorModel,
    Factors<:AbstractFactorProcess,
} <: StatisticalModel
    y::Data
    μ::Mean
    ε::Error
    F::Factors
    function DynamicFactorModel(
        y::AbstractMatrix, 
        μ::AbstractMeanSpecification,
        ε::AbstractErrorModel,
        F::AbstractFactorProcess
    )
        size(y) == size(resid(ε)) || throw(DimensionMismatch("y and residuals must have the same dimensions."))
        size(y, 2) == size(factors(F), 2) || throw(DimensionMismatch("y and factors must have the same number of observations."))

        return new{typeof(y), typeof(μ), typeof(ε), typeof(F)}(y, μ, ε, F)
    end
end

# methods
data(model::DynamicFactorModel) = model.y
mean(model::DynamicFactorModel) = model.μ
errors(model::DynamicFactorModel) = model.ε
resid(model::DynamicFactorModel) = resid(errors(model))
cov(model::DynamicFactorModel) = cov(errors(model), full=true)
process(model::DynamicFactorModel) = model.F
loadings(model::DynamicFactorModel) = loadings(process(model))
factors(model::DynamicFactorModel) = factors(process(model))
dynamics(model::DynamicFactorModel) = dynamics(process(model))
size(model::DynamicFactorModel) = (size(data(model))..., size(process(model)))
nobs(model::DynamicFactorModel) = size(data(model), 2)