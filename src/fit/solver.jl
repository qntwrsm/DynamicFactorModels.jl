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
    (α, V, Γ) = smoother(model)
    for (t, αt) in pairs(α)
        factors(model)[:, t] = αt
    end

    # (M)aximization step
    # update factor process
    e = copy(data(model))
    mean(model) isa Exogenous &&
        mul!(e, slopes(mean(model)), regressors(mean(model)), -true, true)
    update_loadings!(process(model), e, cov(model), V, regularizer.factors)
    update_dynamics!(process(model), V, Γ)

    # update mean specification
    e .= data(model)
    mul!(e, loadings(model), factors(model), -true, true)
    update!(mean(model), e, cov(model), regularizer.mean)

    # update error specification
    mean(model) isa Exogenous &&
        mul!(e, slopes(mean(model)), regressors(mean(model)), -true, true)
    update!(errors(model), e, loadings(model), V, regularizer.error)

    return nothing
end

"""
    update_loadings!(F, y, Σ, V, regularizer)

Update factor loadings ``Λ`` of the factor process `F` using the data `y`, smoothed
covariance matrix `V`, and covariance matrix `Σ` with regularization given by `regularizer`.

For an unrestricted loading matrix the update is perfomed using OLS when regularizer is
`nothing` and using an accelerated proximal gradient method when regularizer is
`NormL1plusL21`. When the loading matrix is restricted to a Nelson-Siegel factor process the
decay parameter ``λ`` is updated using a gradient based minimizer.
"""
function update_loadings!(F::AbstractUnrestrictedFactorProcess, y::AbstractMatrix,
                          Σ::AbstractMatrix, V::AbstractVector, regularizer::Nothing)
    Efy = factors(F) * y'
    Eff = factors(F) * factors(F)'
    for Vt in V
        Eff .+= Vt
    end
    loadings(F) .= (Eff \ Efy)'

    return nothing
end
function update_loadings!(F::AbstractUnrestrictedFactorProcess, y::AbstractMatrix,
                          Σ::AbstractMatrix, V::AbstractVector, regularizer)
    Eyf = y * factors(F)'
    Eff = factors(F) * factors(F)'
    for Vt in V
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
    ffb = FastForwardBackward(maxit = 100, tol = 1e-2)
    (solution, _) = ffb(x0 = loadings(F), f = f, g = regularizer)
    loadings(F) .= solution

    return nothing
end
function update_loadings!(F::AbstractNelsonSiegelFactorProcess, y::AbstractMatrix,
                          Σ::AbstractMatrix, V::AbstractVector, regularizer)
    Eyf = y * factors(F)'
    Eff = factors(F) * factors(F)'
    for Vt in V
        Eff .+= Vt
    end

    function objective(λ::AbstractVector)
        F.λ = exp(λ[1])
        Λ = loadings(F)
        ΩΛ = Σ \ Λ

        return (0.5 * dot(ΩΛ, Λ * Eff) - dot(ΩΛ, Eyf)) / length(V)
    end
    opt = optimize(objective, [log(decay(F))], BFGS(), Optim.Options(g_tol = 1e-4))
    F.λ = exp(Optim.minimizer(opt)[1])

    return nothing
end

"""
    update_dynamics!(F, V, Γ)

Update dynamics of factor process `F` using smoothed covariance matrix `V` and smoothed
auto-covariance matrix `Γ` using OLS.
"""
function update_dynamics!(F::UnrestrictedStationaryIdentified, V::AbstractVector,
                          Γ::AbstractVector)
    for r in 1:nfactors(F)
        num = denom = zero(eltype(factors(F)))
        for t in eachindex(Γ)
            num += factors(F)[r, t + 1] * factors(F)[r, t] + Γ[t]
            denom += factors(F)[r, t]^2 + V[t]
        end
        dynamics(F).diag[r] = num / denom
    end

    return nothing
end
function update_dynamics!(F::UnrestrictedStationaryFull, V::AbstractVector,
                          Γ::AbstractVector)
    @views flag = factors(F)[:, 1:(end - 1)]
    @views flead = factors(F)[:, 2:end]
    Eflfl = flag * flag'
    Effl = flead * flag'
    Eff = flead * flead'
    for t in eachindex(Γ)
        Eflfl .+= V[t]
        Effl .+= Γ[t]
        Eff .+= V[t + 1]
    end
    dynamics(F) .= (Eflfl \ Effl')'
    cov(F).data .= (Eff - dynamics(F) * Effl') ./ length(Γ)

    return nothing
end
function update_dynamics!(F::UnrestrictedUnitRoot, V::AbstractVector, Γ::AbstractVector)
    for r in 1:nfactors(F)
        Eff = Effl = Eflfl = zero(eltype(factors(F)))
        for t in eachindex(Γ)
            flag = factors(F)[r, t]
            flead = factors(F)[r, t + 1]
            Eflfl += flag^2 + V[t]
            Effl += flead * flag + Γ[t]
            Eff += flead^2 + V[t + 1]
        end
        cov(F).diag[r] = (Eff - 2.0 * Effl + Eflfl) / length(Γ)
    end

    return nothing
end
function update_dynamics!(F::NelsonSiegelStationary, V::AbstractVector, Γ::AbstractVector)
    @views flag = factors(F)[:, 1:(end - 1)]
    @views flead = factors(F)[:, 2:end]
    Eflfl = flag * flag'
    Effl = flead * flead'
    Eff = flead * flead'
    for t in eachindex(Γ)
        Eflfl .+= V[t]
        Effl .+= Γ[t]
        Eff .+= V[t + 1]
    end
    dynamics(F) .= (Eflfl \ Effl')'
    cov(F).data .= (Eff - dynamics(F) * Eff1') ./ length(Γ)

    return nothing
end
function update_dynamics!(F::NelsonSiegelUnitRoot, V::AbstractVector, Γ::AbstractVector)
    for r in 1:nfactors(F)
        Eff = Effl = Eflfl = zero(eltype(factors(F)))
        for t in eachindex(Γ)
            flag = factors(F)[r, t]
            flead = factors(F)[r, t + 1]
            Eflfl += flag^2 + V[t]
            Effl += flead * flag + Γ[t]
            Eff += flead^2 + V[t + 1]
        end
        cov(F).diag[r] = (Eff - 2.0 * Effl + Eflfl) / length(Γ)
    end

    return nothing
end

"""
    update!(μ, y, Σ, regularizer)

Update mean specification `μ` using the data minus the common component `y` and covariance
matrix `Σ` with regularization given by `regularizer`.

Update is perfomed using OLS when regularizer is `nothing` and using an accelerated proximal
gradient method when regularizer is not `nothing` for exogeneous mean specification.
"""
update!(μ::ZeroMean, y::AbstractMatrix, Σ::AbstractMatrix, regularizer::Nothing) = nothing
function update!(μ::Exogenous, y::AbstractMatrix, Σ::AbstractMatrix, regularizer::Nothing)
    slopes(μ) .= (regressors(μ)' \ y')'

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
    ffb = FastForwardBackward(maxit = 100, tol = 1e-2)
    (solution, _) = ffb(x0 = slopes(μ), f = f, g = regularizer)
    slopes(μ) .= solution

    return nothing
end

"""
    update!(ε, e, Λ, V, regularizer)

Update error model `ε` using errors `e`, factor loadings `Λ`, smoothed covariance matrix `V`,
and regularization given by `regularizer`.

Update is perfomed using MLE when regularizer is `nothing`. This implies for the covariance
matrix the expectation of the (scaled) sum of squared residuals.
"""
function update!(ε::Simple, e::AbstractMatrix, Λ::AbstractMatrix, V::AbstractVector,
                 regularizer::Nothing)
    Vsum = zero(V[1])
    for Vt in V
        Vsum .+= Vt
    end
    Eee = e * e' + Λ * Vsum * Λ'
    cov(ε).diag .= diag(Eee) ./ size(e, 2)
    cov(ε).diag .= (dot.(eachrow(e), eachrow(e)) .+ dot.(eachrow(Λ), Ref(Vsum), eachrow(Λ))) ./
                   size(e, 2)

    return nothing
end
function update!(ε::SpatialAutoregression, e::AbstractMatrix, Λ::AbstractMatrix,
                 V::AbstractVector, regularizer::Nothing)
    Vsum = zero(V[1])
    for Vt in V
        Vsum .+= Vt
    end
    Eee = e * e' + Λ * Vsum * Λ'

    # update spatial filter
    scale = 2.0 * ε.ρmax
    offset = ε.ρmax
    function objective(ρ::AbstractVector)
        spatial(ε) .= scale .* logistic.(ρ) .- offset
        G = poly(ε)
        Ω = G' * (cov(ε) \ G)

        return -logdet(G) + 0.5 * dot(Ω, Eee) / size(e, 2)
    end
    opt = optimize(objective, logit.((spatial(ε) .+ offset) ./ scale), ConjugateGradient(),
                   Optim.Options(g_tol = 1e-4))
    spatial(ε) .= scale .* logistic.(Optim.minimizer(opt)) .- offset

    # update covariance matrix
    G = poly(ε)
    cov(ε).diag .= dot.(eachrow(G), Ref(Eee), eachrow(G)) ./ size(e, 2)

    return nothing
end
function update!(ε::SpatialAutoregression, e::AbstractMatrix, Λ::AbstractMatrix,
                 V::AbstractVector, regularizer)
    Vsum = zero(V[1])
    for Vt in V
        Vsum .+= Vt
    end
    Eee = e * e' + Λ * Vsum * Λ'

    # update spatial filter
    scale = 2.0 * ε.ρmax
    offset = ε.ρmax
    function objective(ρ::AbstractVector)
        spatial(ε) .= scale .* logistic.(ρ) .- offset
        G = poly(ε)
        Ω = G' * (cov(ε) \ G)

        return -logdet(G) + 0.5 * dot(Ω, Eee) / size(e, 2)
    end
    cache = FiniteDiff.GradientCache(copy(spatial(ε)), copy(spatial(ε)))
    f = ObjectiveWrapper(objective, cache)
    ffb = FastForwardBackward(maxit = 100, tol = 1e-2)
    (solution, _) = ffb(x0 = logit.((spatial(ε) .+ offset) ./ scale), f = f,
                        g = regularizer)
    spatial(ε) .= scale .* logistic.(solution) .- offset

    # update covariance matrix
    G = poly(ε)
    cov(ε).diag .= dot.(eachrow(G), Ref(Eee), eachrow(G)) ./ size(e, 2)

    return nothing
end
function update!(ε::SpatialMovingAverage, e::AbstractMatrix, Λ::AbstractMatrix,
                 V::AbstractVector, regularizer::Nothing)
    Vsum = zero(V[1])
    for Vt in V
        Vsum .+= Vt
    end
    Eee = e * e' + Λ * Vsum * Λ'

    # update spatial filter
    scale = 2.0 * ε.ρmax
    offset = ε.ρmax
    function objective(ρ::AbstractVector)
        spatial(ε) .= scale .* logistic.(ρ) .- offset
        G = poly(ε)
        Σ = G * cov(ε) * G'

        return logdet(G) + 0.5 * tr(Σ \ Eee) / size(e, 2)
    end
    opt = optimize(objective, logit.((spatial(ε) .+ offset) ./ scale), ConjugateGradient(),
                   Optim.Options(g_tol = 1e-4))
    spatial(ε) .= scale .* logistic.(Optim.minimizer(opt)) .- offset

    # update covariance matrix
    G = poly(ε)
    cov(ε).diag .= diag(G \ (G \ Eee)') ./ size(e, 2)

    return nothing
end
function update!(ε::SpatialMovingAverage, e::AbstractMatrix, Λ::AbstractMatrix,
                 V::AbstractVector, regularizer)
    Vsum = zero(V[1])
    for Vt in V
        Vsum .+= Vt
    end
    Eee = e * e' + Λ * Vsum * Λ'

    # update spatial filter
    scale = 2.0 * ε.ρmax
    offset = ε.ρmax
    function objective(ρ::AbstractVector)
        spatial(ε) .= scale .* logistic.(ρ) .- offset
        G = poly(ε)
        Σ = G * cov(ε) * G'

        return logdet(G) + 0.5 * tr(Σ \ Eee) / size(e, 2)
    end
    cache = FiniteDiff.GradientCache(copy(spatial(ε)), copy(spatial(ε)))
    f = ObjectiveWrapper(objective, cache)
    ffb = FastForwardBackward(maxit = 100, tol = 1e-2)
    (solution, _) = ffb(x0 = logit.((spatial(ε) .+ offset) ./ scale), f = f,
                        g = regularizer)
    spatial(ε) .= scale .* logistic.(solution) .- offset

    # update covariance matrix
    G = poly(ε)
    cov(ε).diag .= diag(G \ (G \ Eee)') ./ size(e, 2)

    return nothing
end
