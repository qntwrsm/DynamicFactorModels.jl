#=
regularization.jl

    Provides a collection of types for regularization components used in 
    estimation for dynamic factor models.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/12/27
=#

"""
    NormL1plusL21(λ, γ, dim)

With two nonegative scalars λ and γ, return the function

```math
f(X) = λ ⋅ ∑ |xᵢⱼ| + γ ⋅ ∑ ||xₖ||,
```

where ``xₖ`` is the ``k``-th column of ``X`` if `dim == 1`, and the ``k``-th row
of ``X`` if `dim == 2`. In words, it is the sum of the ``ℓ₁``-norm and sum of
the Euclidean norms of the columns or rows.
"""
struct NormL1plusL21{L1<:NormL1, L21<:NormL21}
    l1::L1
    l21::L21
end

NormL1plusL21(λ::Real=1, γ::Real=1, dim::Int=1) = NormL1plusL21(NormL1(λ), NormL21(γ, dim))

(f::NormL1plusL21)(x) = f.l1(x) + f.l21(x)

function prox!(y, f::NormL1plusL21, x, γ)
    prox!(y, f.l1, x, γ)
    fl21 = prox!(y, f.l21, y, γ)

    return f.l1(y) + fl21
end

"""
    NormL21Weighted(λ, dim=1)

Return the "sum of ``ℓ₂`` norm" function

```math
f(X) = ∑ λᵢ ⋅ ||xᵢ||
```

for a nonnegative `λ` array, where ``xᵢ`` is the ``i``-th column of ``X`` if
`dim == 1`, and the ``i``-th row of ``X`` if `dim == 2`. In words, it is the sum
of the Euclidean norms of the columns or rows.
"""
struct NormL21Weighted{V<:AbstractVector, I}
    λ::V
    dim::I
    function NormL21Weighted{V,I}(λ::V, dim::I) where {V, I}
        if any(λ .< 0)
            error("parameter λ must be nonnegative")
        else
            new(λ, dim)
        end
    end
end

NormL21Weighted(λ::V, dim::I=1) where {V, I} = NormL21Weighted{V, I}(λ, dim)

function (f::NormL21Weighted)(X)
    R = real(eltype(X))
    nslice = R(0)
    n21X = R(0)
    if f.dim == 1
        for j in axes(X, 2)
            nslice = R(0)
            for i in axes(X, 1)
                nslice += abs(X[i, j])^2
            end
            n21X += f.λ[j] * sqrt(nslice)
        end
    elseif f.dim == 2
        for i in axes(X, 1)
            nslice = R(0)
            for j in axes(X, 2)
                nslice += abs(X[i, j])^2
            end
            n21X += f.λ[i] * sqrt(nslice)
        end
    end

    return n21X
end

function prox!(Y, f::NormL21Weighted, X, γ)
    R = real(eltype(X))
    nslice = R(0)
    n21X = R(0)
    if f.dim == 1
        for j in axes(X, 2)
            gl = γ * f.λ[j]
            nslice = R(0)
            for i in axes(X, 1)
                nslice += abs(X[i, j])^2
            end
            nslice = sqrt(nslice)
            scal = 1 - gl / nslice
            scal = scal <= 0 ? R(0) : scal
            for i in axes(X, 1)
                Y[i, j] = scal * X[i, j]
            end
            n21X += f.λ[j] * scal * nslice
        end
    elseif f.dim == 2
        for i in axes(X, 1)
            gl = γ * f.λ[i]
            nslice = R(0)
            for j in axes(X, 2)
                nslice += abs(X[i, j])^2
            end
            nslice = sqrt(nslice)
            scal = 1 - gl / nslice
            scal = scal <= 0 ? R(0) : scal
            for j in axes(X, 2)
                Y[i, j] = scal * X[i, j]
            end
            n21X += f.λ[i] * scal * nslice
        end
    end

    return n21X
end