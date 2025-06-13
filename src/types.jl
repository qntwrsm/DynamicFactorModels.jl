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
struct ZeroMean{T <: DataType} <: AbstractMeanSpecification
    type::T
    n::Int
end

"""
    Exogenous <: AbstractMeanSpecification

Mean specification with exogenous regressors `X` and slopes `β`.
"""
struct Exogenous{TX <: AbstractMatrix, Tβ <: AbstractMatrix} <: AbstractMeanSpecification
    X::TX
    β::Tβ
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
Base.copy(μ::ZeroMean) = ZeroMean(μ.type, μ.n)
Base.copy(μ::Exogenous) = Exogenous(copy(regressors(μ)), copy(slopes(μ)))

# error models
"""
    AbstractErrorModel

Abstract type for error model.
"""
abstract type AbstractErrorModel end

"""
    Simple <: AbstractErrorModel

Simple error model of errors with zero mean and diagonal covariance matrix `Σ` multivariate
normal distribution.
"""
struct Simple{Cov <: Diagonal} <: AbstractErrorModel
    Σ::Cov
end

"""
    SpatialAutoregression <: AbstractErrorModel

Spatial autoregressive error model with spatial dependence `ρ`, maximum spatial dependence
`ρmax`, spatial weights `W`, and covariance matrix `Σ`.
"""
struct SpatialAutoregression{Tρ <: AbstractVector, Tρmax <: Real, TW <: AbstractMatrix,
                             TΣ <: Diagonal} <: AbstractErrorModel
    ρ::Tρ
    ρmax::Tρmax
    W::TW
    Σ::TΣ
    function SpatialAutoregression(ρ::AbstractVector, ρmax::Real, W::AbstractMatrix,
                                   Σ::Diagonal)
        size(W, 1) == size(Σ, 1) ||
            throw(DimensionMismatch("W and Σ must have the same number of rows."))
        (length(ρ) == size(W, 1) || length(ρ) == 1) ||
            throw(DimensionMismatch("ρ and W must have the same number of rows or ρ must be a single element vector."))
        all(ρi -> abs(ρi) < ρmax, ρ) || throw(DomainError("|ρ| < ρ_max."))
        size(W, 1) == size(W, 2) || throw(DimensionMismatch("W must be square."))

        return new{typeof(ρ), typeof(ρmax), typeof(W), typeof(Σ)}(ρ, ρmax, W, Σ)
    end
end

"""
    SpatialMovingAverage <: AbstractErrorModel

Spatial moving average error model with spatial dependence `ρ`, maximum spatial dependence
`ρmax`, spatial weights `W`, and covariance matrix Σ.
"""
struct SpatialMovingAverage{Tρ <: AbstractVector, Tρmax <: Real, TW <: AbstractMatrix,
                            TΣ <: Diagonal} <: AbstractErrorModel
    ρ::Tρ
    ρmax::Tρmax
    W::TW
    Σ::TΣ
    function SpatialMovingAverage(ρ::AbstractVector, ρmax::Real, W::AbstractMatrix,
                                  Σ::Diagonal)
        size(W, 1) == size(Σ, 1) ||
            throw(DimensionMismatch("W and Σ must have the same number of rows."))
        (length(ρ) == size(W, 1) || length(ρ) == 1) ||
            throw(DimensionMismatch("ρ and W must have the same number of rows or ρ must be a single element vector."))
        all(ρi -> abs(ρi) < ρmax, ρ) || throw(DomainError("|ρ| < ρ_max."))
        size(W, 1) == size(W, 2) || throw(DimensionMismatch("W must be square."))

        return new{typeof(ρ), typeof(ρmax), typeof(W), typeof(Σ)}(ρ, ρmax, W, Σ)
    end
end

# methods
cov(ε::AbstractErrorModel; full::Bool = false) = ε.Σ
function cov(ε::SpatialAutoregression; full::Bool = false)
    if full
        return poly(ε) \ ε.Σ / poly(ε)'
    else
        return ε.Σ
    end
end
function cov(ε::SpatialMovingAverage; full::Bool = false)
    if full
        return poly(ε) * ε.Σ * poly(ε)'
    else
        return ε.Σ
    end
end
var(ε::AbstractErrorModel) = cov(ε).diag
spatial(ε::SpatialAutoregression) = ε.ρ
spatial(ε::SpatialMovingAverage) = ε.ρ
weights(ε::SpatialAutoregression) = ε.W
weights(ε::SpatialMovingAverage) = ε.W
poly(ε::SpatialAutoregression) = poly(spatial(ε), weights(ε), :ar)
poly(ε::SpatialMovingAverage) = poly(spatial(ε), weights(ε), :ma)
function poly(ρ::AbstractVector, W::AbstractMatrix, type::Symbol)
    common = length(ρ) == 1 ? ρ .* W : Diagonal(ρ) * W

    return type == :ar ? I - common : I + common
end
Base.copy(ε::Simple) = Simple(copy(cov(ε)))
function Base.copy(ε::SpatialAutoregression)
    SpatialAutoregression(copy(spatial(ε)), ε.ρmax, copy(weights(ε)), copy(cov(ε)))
end
function Base.copy(ε::SpatialMovingAverage)
    SpatialMovingAverage(copy(spatial(ε)), ε.ρmax, copy(weights(ε)), copy(cov(ε)))
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
    UnrestrictedStationaryIdentified <: AbstractUrestrictedFactorProcess

Identified stationary factor process with unrestricted loadings `Λ`, factors `f`, an
dynamics `ϕ`, by imposing a standard multivariate normal distribution.
"""
struct UnrestrictedStationaryIdentified{TΛ <: AbstractMatrix, Tf <: AbstractMatrix,
                                        Tϕ <: Diagonal} <: AbstractUnrestrictedFactorProcess
    Λ::TΛ
    f::Tf
    ϕ::Tϕ
    function UnrestrictedStationaryIdentified(Λ::AbstractMatrix, f::AbstractMatrix,
                                              ϕ::Diagonal)
        size(Λ, 1) >= size(Λ, 2) ||
            throw(ArgumentError("R must be less than or equal to n."))
        size(Λ, 2) == size(f, 1) ||
            throw(DimensionMismatch("multiplication of loadings and factors must be defined."))
        size(ϕ, 1) == size(ϕ, 2) || throw(DimensionMismatch("ϕ must be square."))
        size(ϕ, 1) == size(f, 1) ||
            throw(DimensionMismatch("ϕ and f must have the same number of rows."))

        return new{typeof(Λ), typeof(f), typeof(ϕ)}(Λ, f, ϕ)
    end
end

"""
    UnrestrictedStationaryFull <: AbstractUrestrictedFactorProcess

Dependent stationary factor process with unrestricted loadings `Λ`, factors `f`, an
dynamics `ϕ`, and covariance matrix `Σ`.
"""
struct UnrestrictedStationaryFull{TΛ <: AbstractMatrix, Tf <: AbstractMatrix,
                                  Tϕ <: AbstractMatrix, TΣ <: Symmetric} <:
       AbstractUnrestrictedFactorProcess
    Λ::TΛ
    f::Tf
    ϕ::Tϕ
    Σ::TΣ
    function UnrestrictedStationaryFull(Λ::AbstractMatrix, f::AbstractMatrix,
                                        ϕ::AbstractMatrix, Σ::Symmetric)
        size(Λ, 1) >= size(Λ, 2) ||
            throw(ArgumentError("R must be less than or equal to n."))
        size(Λ, 2) == size(f, 1) ||
            throw(DimensionMismatch("multiplication of loadings and factors must be defined."))
        size(ϕ, 1) == size(ϕ, 2) || throw(DimensionMismatch("ϕ must be square."))
        size(ϕ, 1) == size(f, 1) ||
            throw(DimensionMismatch("ϕ and f must have the same number of rows."))
        size(Σ, 1) == size(f, 1) ||
            throw(DimensionMismatch("Σ and f must have the same number of rows."))

        return new{typeof(Λ), typeof(f), typeof(ϕ), typeof(Σ)}(Λ, f, ϕ, Σ)
    end
end

"""
    UnrestrictedUnitRoot <: AbstractUnrestrictedFactorProcess

Unit-root factor process with unrestricted loadings `Λ`, factors `f`, and diagonal covariance matrix `Σ`.
"""
struct UnrestrictedUnitRoot{TΛ <: AbstractMatrix, Tf <: AbstractMatrix, TΣ <: Diagonal} <:
       AbstractUnrestrictedFactorProcess
    Λ::TΛ
    f::Tf
    Σ::TΣ
    function UnrestrictedUnitRoot(Λ::AbstractMatrix, f::AbstractMatrix, Σ::Diagonal)
        size(Λ, 1) >= size(Λ, 2) ||
            throw(ArgumentError("R must be less than or equal to n."))
        size(Λ, 2) == size(f, 1) ||
            throw(DimensionMismatch("multiplication of loadings and factors must be defined."))
        size(Σ, 1) == size(f, 1) ||
            throw(DimensionMismatch("covariance of dist and f must have the same number of rows."))

        return new{typeof(Λ), typeof(f), typeof(Σ)}(Λ, f, Σ)
    end
end

"""
    NelsonSiegelStationary <: AbstractNelsonSiegelFactorProcess

Stationary Nelson-Siegel factor process with decay parameter `λ`, factors `f`, dynamics `ϕ`,
and covariance matrix `Σ` for maturities `τ`.
"""
mutable struct NelsonSiegelStationary{Tλ <: Real, Tτ <: AbstractVector,
                                      Tf <: AbstractMatrix, Tϕ <: AbstractMatrix,
                                      TΣ <: Symmetric} <: AbstractNelsonSiegelFactorProcess
    λ::Tλ
    τ::Tτ
    f::Tf
    ϕ::Tϕ
    Σ::TΣ
    function NelsonSiegelStationary(λ::Real, τ::AbstractVector, f::AbstractMatrix,
                                    ϕ::AbstractMatrix, Σ::Symmetric)
        λ > 0 || throw(DomainError("λ must be positive"))
        minimum(τ) > 0 || throw(DomainError("maturities must be positive"))
        length(τ) >= 3 || throw(ArgumentError("R must be less than or equal to n."))
        size(ϕ, 1) == size(ϕ, 2) || throw(DimensionMismatch("ϕ must be square."))
        size(f, 1) == 3 ||
            throw(DimensionMismatch("multiplication of loadings and factors must be defined."))
        size(Σ, 1) == size(f, 1) ||
            throw(DimensionMismatch("Σ and f must have the same number of rows."))

        return new{typeof(λ), typeof(τ), typeof(f), typeof(ϕ), typeof(Σ)}(λ, τ, f, ϕ, Σ)
    end
end

"""
    NelsonSiegelUnitRoot <: AbstractNelsonSiegelFactorProcess

Unit-root Nelson-Siegel factor process with decay parameter `λ`, factors `f`,
and covariance matrix `Σ` for maturities `τ`.
"""
mutable struct NelsonSiegelUnitRoot{Tλ <: Real, Tτ <: AbstractVector, Tf <: AbstractMatrix,
                                    TΣ <: Diagonal} <: AbstractNelsonSiegelFactorProcess
    λ::Tλ
    τ::Tτ
    f::Tf
    Σ::TΣ
    function NelsonSiegelUnitRoot(λ::Real, τ::AbstractVector, f::AbstractMatrix,
                                  Σ::Diagonal)
        λ > 0 || throw(DomainError("λ must be positive"))
        minimum(τ) > 0 || throw(DomainError("maturities must be positive"))
        length(τ) >= 3 || throw(ArgumentError("R must be less than or equal to n."))
        size(f, 1) == 3 ||
            throw(DimensionMismatch("multiplication of loadings and factors must be defined."))
        size(Σ, 1) == size(f, 1) ||
            throw(DimensionMismatch("Σ and f must have the same number of rows."))

        return new{typeof(λ), typeof(τ), typeof(f), typeof(Σ)}(λ, τ, f, Σ)
    end
end

# methods
decay(F::AbstractNelsonSiegelFactorProcess) = F.λ
maturities(F::AbstractNelsonSiegelFactorProcess) = F.τ
loadings(F::AbstractUnrestrictedFactorProcess) = F.Λ
function loadings(F::AbstractNelsonSiegelFactorProcess)
    n = length(maturities(F))
    Λ = ones(n, 3)
    for (i, τ) in pairs(maturities(F))
        z = exp(-decay(F) * τ)
        Λ[i, 2:end] .= (1 - z) / (decay(F) * τ)
        Λ[i, end] -= z
    end

    return Λ
end
factors(F::AbstractFactorProcess) = F.f
dynamics(F::AbstractFactorProcess) = F.ϕ
dynamics(F::UnrestrictedUnitRoot) = I
dynamics(F::NelsonSiegelUnitRoot) = I
cov(F::AbstractFactorProcess) = F.Σ
cov(F::UnrestrictedStationaryIdentified) = I
nfactors(F::AbstractFactorProcess) = size(factors(F), 1)
function Base.copy(F::UnrestrictedStationaryIdentified)
    Λ = copy(loadings(F))
    f = copy(factors(F))
    ϕ = copy(dynamics(F))

    return UnrestrictedStationaryIdentified(Λ, f, ϕ)
end
function Base.copy(F::UnrestrictedStationaryFull)
    Λ = copy(loadings(F))
    f = copy(factors(F))
    ϕ = copy(dynamics(F))
    Σ = copy(cov(F))

    return UnrestrictedStationaryFull(Λ, f, ϕ, Σ)
end
function Base.copy(F::UnrestrictedUnitRoot)
    Λ = copy(loadings(F))
    f = copy(factors(F))
    Σ = copy(cov(F))

    return UnrestrictedUnitRoot(Λ, f, Σ)
end
function Base.copy(F::NelsonSiegelStationary)
    τ = copy(maturities(F))
    f = copy(factors(F))
    ϕ = copy(dynamics(F))
    Σ = copy(cov(F))

    return NelsonSiegelStationary(decay(F), τ, f, ϕ, Σ)
end
function Base.copy(F::NelsonSiegelUnitRoot)
    τ = copy(maturities(F))
    f = copy(factors(F))
    Σ = copy(cov(F))

    return NelsonSiegelUnitRoot(decay(F), τ, f, Σ)
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
struct DynamicFactorModel{Ty <: AbstractMatrix, Tμ <: AbstractMeanSpecification,
                          Tε <: AbstractErrorModel, TF <: AbstractFactorProcess} <:
       StatisticalModel
    y::Ty
    μ::Tμ
    ε::Tε
    F::TF
    function DynamicFactorModel(y::AbstractMatrix, μ::AbstractMeanSpecification,
                                ε::AbstractErrorModel, F::AbstractFactorProcess)
        size(y, 2) == size(factors(F), 2) ||
            throw(DimensionMismatch("y and factors must have the same number of observations."))

        return new{typeof(y), typeof(μ), typeof(ε), typeof(F)}(y, μ, ε, F)
    end
end

# methods
data(model::DynamicFactorModel) = model.y
mean(model::DynamicFactorModel) = model.μ
errors(model::DynamicFactorModel) = model.ε
cov(model::DynamicFactorModel) = cov(errors(model), full = true)
process(model::DynamicFactorModel) = model.F
loadings(model::DynamicFactorModel) = loadings(process(model))
factors(model::DynamicFactorModel) = factors(process(model))
dynamics(model::DynamicFactorModel) = dynamics(process(model))
nobs(model::DynamicFactorModel) = size(data(model), 2)
nfactors(model::DynamicFactorModel) = nfactors(process(model))
