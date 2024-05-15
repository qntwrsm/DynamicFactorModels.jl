#=
solver.jl

    Provides expectation-maximization (EM) optimization routines for fitting a 
    dynamic factor model w/ and w/o regularization.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/12/08
=#

"""
    update!(model, regularizer)

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
    # update factor process
    resid(model) .= data(model)
    mean(model) isa Exogenous && mul!(resid(model), slopes(mean(model)), regressors(mean(model)), -true, true)
    update_loadings!(process(model), resid(model), cov(model), V, regularizer.factors)
    update_dynamics!(process(model), V, Γ)

    # update mean specification
    resid(model) .= data(model)
    mul!(resid(model), loadings(model), factors(model), -true, true)
    update!(mean(model), resid(model), cov(model), regularizer.mean)

    # update error specification
    mean(model) isa Exogenous && mul!(resid(model), slopes(mean(model)), regressors(mean(model)), -true, true)
    update!(errors(model), loadings(model), V, regularizer.error)

    return nothing
end

"""
    update_loadings!(F, y, Σ, V, regularizer)

Update factor loadings ``Λ`` of the factor process `F` using the data `y`,
smoothed covariance matrix `V`, and covariance matrix `Σ` with regularization
given by `regularizer`.

For an unrestricted loading matrix the update is perfomed using OLS when
regularizer is `nothing` and using an accelerated proximal gradient method when
regularizer is `NormL1plusL21`. When the loading matrix is restricted to a
Nelson-Siegel factor process the decay parameter ``λ`` is updated using a
gradient based minimizer.
"""
function update_loadings!(
    F::AbstractUnrestrictedFactorProcess,
    y::AbstractMatrix, 
    Σ::AbstractMatrix,
    V::AbstractVector, 
    regularizer::Nothing
)
    Efy = factors(F) * y'
    Eff = factors(F) * factors(F)'
    for Vt ∈ V
        Eff .+= Vt
    end
    loadings(F) .= (Eff \ Efy)'

    return nothing
end
function update_loadings!(
    F::AbstractUnrestrictedFactorProcess,
    y::AbstractMatrix, 
    Σ::AbstractMatrix,
    V::AbstractVector, 
    regularizer
)
    Eyf = y * factors(F)'
    Eff = factors(F) * factors(F)'
    for Vt ∈ V
        Eff .+= Vt
    end

    function objective(Λ::AbstractMatrix)
        ΩΛ = Σ \ Λ
        
        return (0.5 * dot(ΩΛ, Λ * Eff) - dot(ΩΛ, Eyf)) / length(V)
    end
    function gradient!(∇::AbstractMatrix, Λ::AbstractMatrix)
        ∇ .= Σ \ (Λ * Eff - Eyf)
        ∇ ./= length(V)
        
        return nothing
    end
    f = ObjectiveGradientWrapper(objective, gradient!)
    ffb = FastForwardBackward(maxit=1_000, tol=1e-4)
    (solution, _) = ffb(x0=loadings(F), f=f, g=regularizer)
    loadings(F) .= solution

    return nothing
end
function update_loadings!(
    F::AbstractNelsonSiegelFactorProcess,
    y::AbstractMatrix, 
    Σ::AbstractMatrix,
    V::AbstractVector, 
    regularizer
)
    Eyf = y * factors(F)'
    Eff = factors(F) * factors(F)'
    for Vt ∈ V
        Eff .+= Vt
    end

    function objective(λ::AbstractVector)
        F.λ = exp(λ[1])
        Λ = loadings(F)
        ΩΛ = Σ \ Λ
        
        return (0.5 * dot(ΩΛ, Λ * Eff) - dot(ΩΛ, Eyf)) / length(V)
    end
    opt = optimize(objective, [log(decay(F))], BFGS(), Optim.Options(g_tol=1e-4))
    F.λ = exp(Optim.minimizer(opt)[1])

    return nothing
end

"""
    update_dynamics!(F, V, Γ)

Update dynamics of factor process `F` using smoothed covariance matrix `V` and
smoothed auto-covariance matrix `Γ` using OLS.
"""
function update_dynamics!(F::UnrestrictedStationaryIdentified, V::AbstractVector, Γ::AbstractVector)
    @views Ef1f1 = factors(F)[:,1:end-1] * factors(F)[:,1:end-1]'
    @views Eff1 = factors(F)[:,2:end] * factors(F)[:,1:end-1]'
    for t ∈ eachindex(Γ)
        Ef1f1 .+= V[t]
        Eff1 .+= Γ[t]
    end
    dynamics(F).diag .= diag(Eff1) ./ diag(Ef1f1)

    return nothing
end
function update_dynamics!(F::UnrestrictedStationaryFull, V::AbstractVector, Γ::AbstractVector)
    @views Ef1f1 = factors(F)[:,1:end-1] * factors(F)[:,1:end-1]'
    @views Eff1 = factors(F)[:,2:end] * factors(F)[:,1:end-1]'
    @views Eff = factors(F)[:,2:end] * factors(F)[:,2:end]'
    for t ∈ eachindex(Γ)
        Ef1f1 .+= V[t]
        Eff1 .+= Γ[t]
        Eff .+= V[t+1]
    end
    dynamics(F) .= (Ef1f1 \ Eff1')'
    cov(F).mat .= (Eff - dynamics(F) * Eff1') ./ length(Γ)
    cov(F).chol.factors .= cholesky(Hermitian(cov(F).mat)).factors

    return nothing
end
function update_dynamics!(F::UnrestrictedUnitRoot, V::AbstractVector, Γ::AbstractVector)
    @views Ef1f1 = factors(F)[:,1:end-1] * factors(F)[:,1:end-1]'
    @views Eff1 = factors(F)[:,2:end] * factors(F)[:,1:end-1]'
    @views Eff = factors(F)[:,2:end] * factors(F)[:,2:end]'
    for t ∈ eachindex(Γ)
        Ef1f1 .+= V[t]
        Eff1 .+= Γ[t]
        Eff .+= V[t+1]
    end
    cov(F).diag .= (diag(Eff) .- 2.0 .* diag(Eff1) .+ diag(Ef1f1)) ./ length(Γ)

    return nothing
end
function update_dynamics!(F::NelsonSiegelStationary, V::AbstractVector, Γ::AbstractVector)
    @views Ef1f1 = factors(F)[:,1:end-1] * factors(F)[:,1:end-1]'
    @views Eff1 = factors(F)[:,2:end] * factors(F)[:,1:end-1]'
    @views Eff = factors(F)[:,2:end] * factors(F)[:,2:end]'
    for t ∈ eachindex(Γ)
        Ef1f1 .+= V[t]
        Eff1 .+= Γ[t]
        Eff .+= V[t+1]
    end
    dynamics(F) .= (Ef1f1 \ Eff1')'
    cov(F).mat .= (Eff - dynamics(F) * Eff1') ./ length(Γ)
    cov(F).chol.factors .= cholesky(Hermitian(cov(F).mat)).factors

    return nothing
end
function update_dynamics!(F::NelsonSiegelUnitRoot, V::AbstractVector, Γ::AbstractVector)
    @views Ef1f1 = factors(F)[:,1:end-1] * factors(F)[:,1:end-1]'
    @views Eff1 = factors(F)[:,2:end] * factors(F)[:,1:end-1]'
    @views Eff = factors(F)[:,2:end] * factors(F)[:,2:end]'
    for t ∈ eachindex(Γ)
        Ef1f1 .+= V[t]
        Eff1 .+= Γ[t]
        Eff .+= V[t+1]
    end
    cov(F).diag .= (diag(Eff) .- 2.0 .* diag(Eff1) .+ diag(Ef1f1)) ./ length(Γ)

    return nothing
end

"""
    update!(μ, y, Σ, regularizer)

Update mean specification `μ` using the data minus the common component `y` and
covariance matrix `Σ` with regularization given by `regularizer`.

Update is perfomed using OLS when regularizer is `nothing` and using an
accelerated proximal gradient method when regularizer is not `nothing` for
exogeneous mean specification.
"""
update!(μ::ZeroMean, y::AbstractMatrix, Σ::AbstractMatrix, regularizer::Nothing) = nothing
function update!(μ::Exogenous, y::AbstractMatrix, Σ::AbstractMatrix, regularizer::Nothing)
    Xy = regressors(μ) * y'
    XX = regressors(μ) * regressors(μ)'
    slopes(μ) .= (XX \ Xy)'

    return nothing
end
function update!(μ::Exogenous, y::AbstractMatrix, Σ::AbstractMatrix, regularizer)
    yX = y * regressors(μ)'
    XX = regressors(μ) * regressors(μ)'
    
    function objective(β::AbstractMatrix)
        Ωβ = Σ \ β
        
        return (0.5 * dot(Ωβ, β * XX) - dot(Ωβ, yX)) / size(regressors(μ), 2)
    end
    function gradient!(∇::AbstractMatrix, β::AbstractMatrix)
        ∇ .= Σ \ (β * XX - yX)
        ∇ ./= size(regressors(μ), 2)
        
        return nothing
    end
    f = ObjectiveGradientWrapper(objective, gradient!)
    ffb = FastForwardBackward(maxit=1_000, tol=1e-4)
    (solution, _) = ffb(x0=slopes(μ), f=f, g=regularizer)
    slopes(μ) .= solution

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
        ConjugateGradient(),
        Optim.Options(g_tol=1e-4)
    )
    spatial(ε) .= scale .* logistic.(Optim.minimizer(opt)) .- offset

    # update covariance matrix
    G = poly(ε)
    cov(ε).diag .= diag(G * Eee * G') ./ size(resid(ε), 2)

    return nothing
end
function update!(ε::SpatialAutoregression, Λ::AbstractMatrix, V::AbstractVector, regularizer)
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
    ffb = FastForwardBackward(maxit=1_000, tol=1e-4)
    (solution, _) = ffb(x0=logit.((spatial(ε) .+ offset) ./ scale), f=ObjectiveWrapper(objective), g=regularizer)
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
        ConjugateGradient(),
        Optim.Options(g_tol=1e-4)
    )
    spatial(ε) .= scale .* logistic.(Optim.minimizer(opt)) .- offset

    # update covariance matrix
    G = poly(ε)
    cov(ε).diag .= diag(G \ (G \ Eee)') ./ size(resid(ε), 2)

    return nothing
end
function update!(ε::SpatialMovingAverage, Λ::AbstractMatrix, V::AbstractVector, regularizer)
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
    ffb = FastForwardBackward(maxit=1_000, tol=1e-4)
    (solution, _) = ffb(x0=logit.((spatial(ε) .+ offset) ./ scale), f=Objective(objective), g=regularizer)
    spatial(ε) .= scale .* logistic.(solution) .- offset

    # update covariance matrix
    G = poly(ε)
    cov(ε).diag .= diag(G \ (G \ Eee)') ./ size(resid(ε), 2)

    return nothing
end