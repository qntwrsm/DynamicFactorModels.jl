#=
regularization.jl

    Provides a collection of types for reguralization components used in 
    estimation for dynamic factor models.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/12/27
=#

"""
    AbstractReguralization

Abstract type for reguralization components. 
"""
abstract type AbstractReguralization end

"""
    NormL1 <: AbstractReguralization

``ℓ₁``-norm reguralization with strength `λ` and weights `weights`.

```math
f(x) = λ ∑ wᵢ|xᵢ|
```
"""
mutable struct NormL1{Strength<:Real, Weights<:AbstractVector} <: AbstractReguralization
    λ::Strength
    weights::Weights
end

"""
    GeneralizedNormL1 <: AbstractReguralization

Generalized ``ℓ₁``-norm reguralization with strength `λ`, weights `weights`, and
linear mapping `D`.

```math
f(x) = λ ∑ wᵢ|Dᵢx|
```
"""
mutable struct GeneralizedNormL1{
    Strength<:Real, 
    Weights<:AbstractVector, 
    Mapping::AbstractMatrix
} <: AbstractReguralization
    λ::Strength
    weights::Weights
    D::Mapping
end

"""
    NormL1L2 <: AbstractReguralization

Sum of ``ℓ₁``-norm and ``ℓ₂``-norm reguralization with strength `λ`, ratio `α`,
``ℓ₁``-norm weights `weights_l1`, and sum of ``ℓ₂``-norm weights `weights_l2`.

```math
f(x) = αλ ∑ wᵢ|xᵢ| + (1 - α)λ ∑ wⱼ||xⱼ||
```
"""
mutable struct NormL1L2{
    Strength<:Real, 
    Ratio<:Real, 
    WeightsL1<:AbstractVector, 
    WeightsL2<:AbstractVector
} <: AbstractReguralization
    λ::Strength
    α::Ratio
    weights_l1::WeightsL1
    weights_l2::WeightsL2
end