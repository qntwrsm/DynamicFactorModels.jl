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
    update_transition!(process(model), V, Γ)

    # update mean specification
    update!(mean(model), regularizer.mean)

    # update error specification
    update!(errors(model), regularizer.error)

    return nothing
end