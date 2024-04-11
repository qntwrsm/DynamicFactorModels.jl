#=
utilities.jl

    Provides a collection of utility tools for fitting dynamic factor models, 
    such as initialization, convergence checks, and log-likelihood evaluation.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/12/08
=#

"""
    init!(model, method)

Initialize the dynamic factor model `model` by `method`.

When `method` is set to `:data`: 
- Initialization of the loading matrix is based on principal component analysis
  (PCA) of the data if the loaings matrix is unrestricted and based on the value
  of the decay parameter found in Diebold and Li (2006) if it obeys a
  Nelson-Siegel structure.
- Initialization of the factor process dynamics is based on the OLS estimates of
  the autoregressions of the PCA factors or Nelson-Siegel factors estimated
  using the two-step approach of Diebold and Li (2006).
- Initialization of the exogeneous mean specification is based on the OLS
  estimates of the slopes based on the residuals, ``yₜ - Λ̂f̂ₜ``.
- Initialization of the simple error model is based on the sample covariance
  matrix of the residuals, ``yₜ - μ̂ₜ - Λ̂f̂ₜ``. For the spatial error models
  the spatial filter is first estimated, after which the sample covariance
  matrix is calculated on the filtered residuals.

When `method` is set to `:none` no initialization is performed and the model is
assumed to have been initialized manually before fitting.
"""
function init!(model::DynamicFactorModel, method::NamedTuple)
    # initialize mean specification
    init!(mean(model), method.mean, data(model))

    # initialize factor component
    if method.factors == :data
        # factors via PCA
        M = fit(PCA, data(model) .- mean(mean(model)), maxoutdim=size(process(model)), pratio=1.0)
        loadings(model) .= projection(M)
        factors(model) .= transform(M, data(model) .- mean(mean(model)))
        
        # factor process
        if process(model) isa Stationary
            for (r, f) = pairs(eachrow(factors(model)))
                ϕi = dot(f[1:end-1], f[2:end]) / sum(abs2, f[1:end-1])
                dynamics(model).diag[r] = max(-0.99, min(0.99, ϕi))
            end
        elseif process(model) isa UnitRoot
            var(process(model)) .= var(factors(model), dims=2)
        end
    end

    # initialize error specification
    resid(model) .= data(model) .- mean(mean(model)) - loadings(model) * factors(model)
    init!(errors(model), method.error)
        
    return nothing
end

# factor process
function init!(F::UnrestrictedStationary, method::Symbol, y::AbstractMatrix)
    if method.factors == :data
        # factors and loadings via PCA
        M = fit(PCA, y, maxoutdim=size(F), pratio=1.0)
        loadings(F) .= projection(M)
        factors(F) .= transform(M, y)
        
        # factor dynamics
        for (r, f) = pairs(eachrow(factors(model)))
            ϕi = dot(f[1:end-1], f[2:end]) / sum(abs2, f[1:end-1])
            dynamics(model).diag[r] = max(-0.99, min(0.99, ϕi))
        end
    end

    return nothing
end
function init!(F::UnrestrictedUnitRoot, method::Symbol, y::AbstractMatrix)
    if method.factors == :data
        # factors and loadings via PCA
        M = fit(PCA, y, maxoutdim=size(F), pratio=1.0)
        loadings(F) .= projection(M)
        factors(F) .= transform(M, y)
        
        # factor variance
        var(process(model)) .= var(factors(model), dims=2)
    end

    return nothing
end
function init!(F::NelsonSiegelStationary, method::Symbol, y::AbstractMatrix)
    if method.factors == :data
        # factors and decay using Diebold and Li (2006)
        decay(F) = 0.0605
        Λ = loadings(F)
        factors(F) .= (Λ' * Λ) \ (Λ' * y)
        
        # factor dynamics
        @views f1f1 = factors(F)[:,1:end-1] * factors(F)[:,1:end-1]'
        @views ff1 = factors(F)[:,2:end] * factors(F)[:,1:end-1]'
        dynamics(F) .= ff1 / f1f1

        # factor variance
        @views η = factors(F)[:,2:end] - dynamics(F) * factors(F)[:,1:end-1]
        cov(F).mat .= cov(η, dims=2)
        cov(F.chol .= cholesky(cov(F).mat))
    end

    return nothing
end
function init!(F::NelsonSiegelUnitRoot, method::Symbol, y::AbstractMatrix)
    if method.factors == :data
        # factors and decay using Diebold and Li (2006)
        decay(F) = 0.0605
        Λ = loadings(F)
        factors(F) .= (Λ' * Λ) \ (Λ' * y)
        
        # factor process
        @views η = factors(F)[:,2:end] - factors(F)[:,1:end-1]
        cov(F).mat .= cov(η, dims=2)
        cov(F.chol .= cholesky(cov(F).mat))
    end

    return nothing
end

# mean specification
init!(μ::ZeroMean, method::Symbol, y::AbstractMatrix) = nothing
init!(μ::Exogenous, method::Symbol, y::AbstractMatrix) = method == :data && (slopes(μ) .= y / regressors(μ))

# error specification
init!(ε::Simple, method::Symbol) = method == :data && (cov(ε).diag .= var(resid(ε), dims=2))
function init!(ε::SpatialAutoregression, method::Symbol)
    if method == :data
        # estimate spatial filter
        if length(spatial(ε)) == 1
            ρ = dot(weights(ε) * resid(ε), resid(ε)) / sum(abs2, weights(ε) * resid(ε))
            spatial(ε) .= max(-0.99 * ε.ρ_max, min(0.99 * ε.ρ_max, ρ))
        else
            for i ∈ eachindex(spatial(ε))
                ρi = dot(weights(ε)[i,:]' * resid(ε), resid(ε)[i,:]) / sum(abs2, weights(ε)[i,:]' * resid(ε))
                spatial(ε)[i] = max(-0.99 * ε.ρ_max, min(0.99 * ε.ρ_max, ρi))
            end
        end

        # estimate covariance matrix
        resid(ε) .= poly(ε) * resid(ε)
        cov(ε).diag .= var(resid(ε), dims=2)
    end

    return nothing
end
function init!(ε::SpatialMovingAverage, method::Symbol)
    if method == :data
        # estimate spatial filter
        if length(spatial(ε)) == 1
            ρ = dot(weights(ε) * resid(ε), resid(ε)) / sum(abs2, weights(ε) * resid(ε))
            spatial(ε) .= max(-0.99 * ε.ρ_max, min(0.99 * ε.ρ_max, ρ))
        else
            for i ∈ eachindex(spatial(ε))
                ρi = dot(weights(ε)[i,:]' * resid(ε), resid(ε)[i,:]) / sum(abs2, weights(ε)[i,:]' * resid(ε))
                spatial(ε)[i] = max(-0.99 * ε.ρ_max, min(0.99 * ε.ρ_max, ρi))
            end
        end

        # estimate covariance matrix
        resid(ε) .= poly(ε) \ resid(ε)
        cov(ε).diag .= var(resid(ε), dims=2)
    end

    return nothing
end

function params(model::DynamicFactorModel)
    # number of parameters
    n_params = length(loadings(model)) + length(cov(errors(model)).diag)
    process(model) isa Stationary && (n_params += length(dynamics(model).diag))
    process(model) isa UnitRoot && (n_params += length(var(process(model))))
    errors(model) isa Union{SpatialAutoregression, SpatialMovingAverage} && (n_params += length(spatial(errors(model))))
    mean(model) isa Exogenous && (n_params += length(slopes(mean(model))))

    # parameters
    θ = zeros(n_params)
    params!(θ, model)

    return θ
end

function params!(θ::AbstractVector, model::DynamicFactorModel)
    idx = 1

    # loadings
    offset = length(loadings(model))
    θ[idx:idx+offset-1] = vec(loadings(model))
    idx += offset

    # factor process
    if process(model) isa Stationary
        offset = length(dynamics(model).diag)
        θ[idx:idx+offset-1] = dynamics(model).diag
        idx += offset
    elseif process(model) isa UnitRoot
        offset = length(var(process(model)))
        θ[idx:idx+offset-1] = var(process(model))
        idx += offset
    end

    # covariance matrix
    offset = length(cov(errors(model)).diag)
    θ[idx:idx+offset-1] = cov(errors(model)).diag
    idx += offset

    # mean
    if mean(model) isa Exogenous
        offset = length(slopes(mean(model)))
        θ[idx:idx+offset-1] = vec(slopes(mean(model)))
        idx += offset
    end
    
    # spatial dependence
    if errors(model) isa Union{SpatialAutoregression, SpatialMovingAverage}
        offset = length(spatial(errors(model)))
        θ[idx:idx+offset-1] = spatial(errors(model))
    end

    return nothing
end
function params!(model::DynamicFactorModel, θ::AbstractVector)
    idx = 1

    # loadings
    offset = length(loadings(model))
    vec(loadings(model)) .= view(θ, idx:idx+offset-1)
    idx += offset

    # factor process
    if process(model) isa Stationary
        offset = length(dynamics(model).diag)
        dynamics(model).diag .= view(θ, idx:idx+offset-1)
        idx += offset
    elseif process(model) isa UnitRoot
        offset = length(var(process(model)))
        var(process(model)) .= view(θ, idx:idx+offset-1)
        idx += offset
    end

    # covariance matrix
    offset = length(cov(errors(model)).diag)
    cov(errors(model)).diag .= view(θ, idx:idx+offset-1)
    idx += offset

    # mean
    if mean(model) isa Exogenous
        offset = length(slopes(mean(model)))
        vec(slopes(mean(model))) .= view(θ, idx:idx+offset-1)
        idx += offset
    end
    
    # spatial dependence
    if errors(model) isa Union{SpatialAutoregression, SpatialMovingAverage}
        offset = length(spatial(errors(model)))
        spatial(errors(model)) .= view(θ, idx:idx+offset-1)
    end

    return nothing
end

function loglikelihood(model::DynamicFactorModel)
    (n, T) = size(model)[1:end-1]
    (_, _, v, F, _) = filter(model)

    # annihilator matrix
    if errors(model) isa SpatialAutoregression
        Hinv = poly(errors(model))' * (cov(errors(model)) \ poly(errors(model)))
        Zt_Hinv = loadings(model)' * Hinv
    else
        Zt_Hinv = (cov(model)' \ loadings(model))'
    end
    Zt_Hinv_Z = Zt_Hinv * loadings(model)
    P = loadings(model) * (Zt_Hinv_Z \ Zt_Hinv)
    M = I - P

    ll = -0.5 * T * (n * log2π + logdet(cov(model)))
    y_demeaned = data(model) .- mean(mean(model))
    e = similar(y_demeaned, n)
    for (t, yt) ∈ pairs(eachcol(y_demeaned))
        ll -= 0.5 * (logdet(F[t]) + dot(v[t], inv(F[t]), v[t]))
        mul!(e, M, yt)
        ll -= 0.5 * dot(e, inv(cov(model)), e)
    end

    return ll
end

function dof(model::DynamicFactorModel)
    # factor component
    k = sum(!iszero, loadings(model))
    process(model) isa Stationary && (k += length(dynamics(model).diag))
    process(model) isa UnitRoot && (k += length(var(process(model))))

    # mean specification
    mean(model) isa Exogenous && (k += sum(!iszero, slopes(mean(model))))

    # error specification
    k += sum(!iszero, cov(errors(model)))
    if errors(model) isa Union{SpatialAutoregression, SpatialMovingAverage}
        k += sum(!iszero, diff(spatial(errors(model))), init=0) + 1
    end

    return k
end