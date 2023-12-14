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
    for (t, α̂t) ∈ pairs(α̂)
        factors(model)[:,t] = α̂t
    end
    
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
function update_loadings!(Λ::AbstractMatrix, y::AbstractMatrix, f::AbstractMatrix, V::AbstractVector, regularizer::Nothing)
    Eyf = y * f'
    Eff = f * f'
    for Vt ∈ V
        Eff .+= Vt
    end
    # update
    Λ .= Eyf / Eff

    return nothing
end

function update!(F::FactorProcess, V::AbstractVector, Γ::AbstractVector)
    @views Ef1f1 = factors(F)[:,1:end-1] * factors(F)[:,1:end-1]'
    @views Eff1 = factors(F)[:,2:end] * factors(F)[:,1:end-1]'
    for t ∈ eachindex(Γ)
        Ef1f1 .+= V[t]
        Eff1 .+= Γ[t]
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