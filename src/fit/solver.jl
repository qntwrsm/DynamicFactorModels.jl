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
    resid(model) .= data(model)
    mean(model) isa Exogenous && mul!(resid(model), slopes(mean(model)), regressors(mean(model)), -true, true)
    if regularizer.factors isa Nothing
        update_loadings!(loadings(model), resid(model), factors(model), V, regularizer.factors)
    else
        update_loadings!(loadings(model), resid(model), factors(model), V, regularizer.factors, cov(model))
    end
    update!(process(model), V, Γ)

    # update mean specification
    resid(model) .= data(model)
    mul!(resid(model), loadings(model), factors(model), -true, true)
    if regularizer.mean isa Nothing
        update!(mean(model), resid(model), regularizer.mean)
    else
        update!(mean(model), resid(model), regularizer.mean, cov(model))
    end

    # update error specification
    mean(model) isa Exogenous && mul!(resid(model), slopes(mean(model)), regressors(mean(model)), -true, true)
    update!(errors(model), loadings(model), V, regularizer.error)

    return nothing
end

"""
    update_loadings!(Λ, y, f, V, regularizer[, Ω])

Update factor loadings `Λ` using the data `y`, smoothed factors `f`, smoothed
covariance matrix `V`, and covariance matrix `Σ` with regularization given by
`regularizer`.

Update is perfomed using OLS when regularizer is `nothing` and using an
accelerated proximal gradient method when regularizer is `NormL1plusL21`.
"""
function update_loadings!(
    Λ::AbstractMatrix, 
    y::AbstractMatrix, 
    f::AbstractMatrix, 
    V::AbstractVector, 
    regularizer::Nothing
)
    Eyf = y * f'
    Eff = f * f'
    for Vt ∈ V
        Eff .+= Vt
    end
    Λ .= Eyf / Eff

    return nothing
end
function update_loadings!(
    Λ::AbstractMatrix, 
    y::AbstractMatrix, 
    f::AbstractMatrix, 
    V::AbstractVector, 
    regularizer::NormL1plusL21, 
    Σ::AbstractMatrix
)
    Eyf = y * f'
    Eff = f * f'
    for Vt ∈ V
        Eff .+= Vt
    end

    function objective(λ::AbstractVector)
        λmat = reshape(λ, size(Λ))
        Ωλmat = Σ \ λmat
        
        return (0.5 * dot(Ωλmat, λmat * Eff) - dot(Ωλmat, Eyf)) / length(V)
    end
    ffb = FastForwardBackward(
        stop=(iter, state) -> norm(state.res, Inf) < 1e-4
    )
    (solution, _) = ffb(x0=vec(Λ), f=objective, g=regularizer)
    Λ .= reshape(solution, size(Λ))

    return nothing
end

"""
    update!(F, V, Γ)

Update dynamics of factor process `F` using smoothed covariance matrix `V` and
smoothed auto-covariance matrix `Γ` using OLS.
"""
function update!(F::FactorProcess, V::AbstractVector, Γ::AbstractVector)
    @views Ef1f1 = factors(F)[:,1:end-1] * factors(F)[:,1:end-1]'
    @views Eff1 = factors(F)[:,2:end] * factors(F)[:,1:end-1]'
    for t ∈ eachindex(Γ)
        Ef1f1 .+= V[t]
        Eff1 .+= Γ[t]
    end
    dynamics(F).diag .= diag(Eff1) ./ diag(Ef1f1)

    return nothing
end

"""
    update!(μ, y, regularizer[, Σ])

Update mean specification `μ` using the data minus the common component `y` with
regularization given by `regularizer`.

Update is perfomed using OLS when regularizer is `nothing` and using an
accelerated proximal gradient method when regularizer is `NormL1plusL21` for
exogeneous mean specification.
"""
update!(μ::ZeroMean, y::AbstractMatrix, regularizer::Nothing) = nothing
function update!(μ::Exogenous, y::AbstractMatrix, regularizer::Nothing)
    yX = y * regressors(μ)'
    XX = regressors(μ) * regressors(μ)'
    slopes(μ) .= yX / XX

    return nothing
end
function update!(μ::Exogenous, y::AbstractMatrix, regularizer::NormL1plusL21, Σ::AbstractMatrix)
    yX = y * regressors(μ)'
    XX = regressors(μ) * regressors(μ)'
    
    function objective(β::AbstractVector)
        βmat = reshape(β, size(slopes(μ)))
        Ωβmat = Σ \ βmat
        
        return (0.5 * dot(Ωβmat, βmat * XX) - dot(Ωβmat, yX)) / size(regressors(μ), 2)
    end
    ffb = FastForwardBackward(
        stop=(iter, state) -> norm(state.res, Inf) < 1e-4
    )
    (solution, _) = ffb(x0=vec(slopes(μ)), f=objective, g=regularizer)
    slopes(μ) .= reshape(solution, size(slopes(μ)))

    return nothing
end

"""
    update!(ε, Λ, V, regularizer)

Update error model `ε` using factor loadings `Λ`, smoothed covariance matrix
`V`, and regularization given by `regularizer`.

Update is perfomed using MLE when regularizer is `nothing`. This implies for the
covariance matrix the expectation of the (scaled) sum of squared residuals.
"""
function update!(ε::Simple, Λ::AbstractMatrix, V::AbstractVector, regularizer::Nothing)
    Vsum = zero(V[1])
    for Vt ∈ V
        Vsum .+= Vt
    end
    Eee = resid(ε) * resid(ε)' + Λ * Vsum * Λ'
    cov(ε).diag .= diag(Eee) ./ size(resid(ε), 2)

    return nothing
end
function update!(ε::SpatialAutoregression, Λ::AbstractMatrix, V::AbstractVector, regularizer::Nothing)
    Vsum = zero(V[1])
    for Vt ∈ V
        Vsum .+= Vt
    end
    Eee = resid(ε) * resid(ε)' + Λ * Vsum * Λ'

    # update spatial filter
    scale = 2.0 * ε.ρ_max
    offset = ε.ρ_max
    function objective(ρ::AbstractVector)
        spatial(ε) .= scale .* logistic.(ρ) .- offset
        G = poly(ε)
        Ω = G' * (cov(ε) \ G)

        return -logdet(G) + 0.5 * dot(Ω, Eee) / size(resid(ε), 2)
    end
    opt = optimize(
        objective, 
        logit.((spatial(ε) .+ offset) ./ scale), 
        LBFGS(),
        Optim.Options(g_tol=1e-4)
    )
    spatial(ε) .= scale .* logistic.(Optim.minimizer(opt)) .- offset

    # update covariance matrix
    G = poly(ε)
    cov(ε).diag .= diag(G * Eee * G') ./ size(resid(ε), 2)

    return nothing
end
function update!(ε::SpatialAutoregression, Λ::AbstractMatrix, V::AbstractVector, regularizer::TotalVariation1D)
    Vsum = zero(V[1])
    for Vt ∈ V
        Vsum .+= Vt
    end
    Eee = resid(ε) * resid(ε)' + Λ * Vsum * Λ'

    # update spatial filter
    scale = 2.0 * ε.ρ_max
    offset = ε.ρ_max
    function objective(ρ::AbstractVector)
        # Zygote does not support array mutation
        ρ_trans = scale .* logistic.(ρ) .- offset
        if length(ρ_trans) == 1
            G = I - ρ_trans .* weights(ε)
        else
            G = I - Diagonal(ρ_trans) * weights(ε)
        end
        Ω = G' * (cov(ε) \ G)

        return -logdet(G) + 0.5 * dot(Ω, Eee) / size(resid(ε), 2)
    end
    ffb = FastForwardBackward(
        stop=(iter, state) -> norm(state.res, Inf) < 1e-4
    )
    (solution, _) = ffb(x0=logit.((spatial(ε) .+ offset) ./ scale), f=objective, g=regularizer)
    spatial(ε) .= scale .* logistic.(solution) .- offset

    # update covariance matrix
    G = poly(ε)
    cov(ε).diag .= diag(G * Eee * G') ./ size(resid(ε), 2)

    return nothing
end
function update!(ε::SpatialMovingAverage, Λ::AbstractMatrix, V::AbstractVector, regularizer::Nothing)
    Vsum = zero(V[1])
    for Vt ∈ V
        Vsum .+= Vt
    end
    Eee = resid(ε) * resid(ε)' + Λ * Vsum * Λ'

    # update spatial filter
    scale = 2.0 * ε.ρ_max
    offset = ε.ρ_max
    function objective(ρ::AbstractVector)
        spatial(ε) .= scale .* logistic.(ρ) .- offset
        G = poly(ε)
        Σ = G * cov(ε) * G'

        return logdet(G) + 0.5 * tr(Σ \ Eee) / size(resid(ε), 2)
    end
    opt = optimize(
        objective, 
        logit.((spatial(ε) .+ offset) ./ scale), 
        LBFGS(),
        Optim.Options(g_tol=1e-4)
    )
    spatial(ε) .= scale .* logistic.(Optim.minimizer(opt)) .- offset

    # update covariance matrix
    G = poly(ε)
    cov(ε).diag .= diag(G \ (G \ Eee)') ./ size(resid(ε), 2)

    return nothing
end
function update!(ε::SpatialMovingAverage, Λ::AbstractMatrix, V::AbstractVector, regularizer::TotalVariation1D)
    Vsum = zero(V[1])
    for Vt ∈ V
        Vsum .+= Vt
    end
    Eee = resid(ε) * resid(ε)' + Λ * Vsum * Λ'

    # update spatial filter
    scale = 2.0 * ε.ρ_max
    offset = ε.ρ_max
    function objective(ρ::AbstractVector)
        # Zygote does not support array mutation
        ρ_trans = scale .* logistic.(ρ) .- offset
        if length(ρ_trans) == 1
            G = I + ρ_trans .* weights(ε)
        else
            G = I + Diagonal(ρ_trans) * weights(ε)
        end
        Σ = G * cov(ε) * G'

        return logdet(G) + 0.5 * tr(Σ \ Eee) / size(resid(ε), 2)
    end
    ffb = FastForwardBackward(
        stop=(iter, state) -> norm(state.res, Inf) < 1e-4
    )
    (solution, _) = ffb(x0=logit.((spatial(ε) .+ offset) ./ scale), f=objective, g=regularizer)
    spatial(ε) .= scale .* logistic.(solution) .- offset

    # update covariance matrix
    G = poly(ε)
    cov(ε).diag .= diag(G \ (G \ Eee)') ./ size(resid(ε), 2)

    return nothing
end