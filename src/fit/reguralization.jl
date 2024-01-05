#=
regularization.jl

    Provides a collection of types for reguralization components used in 
    estimation for dynamic factor models.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/12/27
=#

"""
    NormL1plusL21(λ, γ, dim)

With two nonegative scalars λ and γ, return the function

```math
f(X) = λ ∑ |xᵢⱼ| + γ ∑ ||xₖ||,
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