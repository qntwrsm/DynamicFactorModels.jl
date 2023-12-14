#=
solver.jl

    Provides expectation-maximization (EM) optimization routines for fitting a 
    dynamic factor model w/ and w/o regularization.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/12/08
=#

"""
    update!(model)

Update parameters of dynamic factor model `model` using an
expectation-maximization (EM) solve w/ regularization term given by
`regularizer`.
"""
function update!(model::DynamicFactorModel, regularizer::NamedTuple)
    # (E)xpectation step
    (α̂, V, Γ) = smoother(model)
    factors(model) .= α̂
    
    # (M)aximization step
    # update factor loadings and dynamics
    update_loadings!(loadings(model), data(model), factors(model), V, regularizer.factors)
    update!(process(model), V, Γ)

    # update mean specification
    update!(mean(model), V, regularizer.mean)

    # update error specification
    update!(errors(model), V, regularizer.error)

    return nothing
end

"""
    update_loadings!(Λ, y, f, V, regularizer)

Update factor loadings `Λ` using the data `y`, smoothed factors `f`, and
smoothed covariance matrix `V` with regularization given by `regularizer`.
"""
function update_loadings!(Λ::AbstractMatrix, y::AbstractMatrix, f::AbstractVector, V::AbstractVector, regularizer::Nothing)
    Eyf = zero(Λ)
    Eff = zero(V[1])
    for (yt, ft, Vt) ∈ zip(eachcol(y), f, V)
        mul!(Eyf, yt, ft', true, true)
        Eff .+= Vt
        mul!(Eff, ft, ft', true, true)
    end
    # update
    Λ .= Eyf / Eff

    return nothing
end

function update!(F::FactorProcess, V::AbstractVector, Γ::AbstractVector)
    Ef1f1 = zero(V[1])
    Eff1 = zero(V[1]) 
    for t ∈ eachindex(Γ)
        Ef1f1 .+= V[t]
        mul!(Ef1f1, factors(F)[t], factors(F)[t]', true, true)
        Eff1 .+= Γ[t]
        mul!(Eff1, factors(F)[t+1], factors(F)[t]', true, true)
    end
    # update
    dynamics(F).diag .= diag(Eff1) ./ diag(Ef1f1)

    return nothing
end

function update!(μ::AbstractMeanSpecification, V::AbstractVector, regularizer)
    #TODO NOT IMPLEMENTED YET

    return nothing
end

function update!(ε::AbstractErrorModel, V::AbstractVector, regularizer)
    #TODO NOT IMPLEMENTED YET

    return nothing
end