#=
utilities.jl

    Provides a collection of utility tools for fitting dynamic factor models,
    such as initialization, convergence checks, and log-likelihood evaluation.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/12/08
=#

"""
    ObjectiveGradientWrapper

Wrapper for objective and gradient functions to dispatch to custom gradient
function.
"""
struct ObjectiveGradientWrapper{F, G}
    f::F
    g::G
end

(f::ObjectiveGradientWrapper)(x) = f.f(x)

ProximalAlgorithms.value_and_gradient(f::ObjectiveGradientWrapper, x) = (f(x), f.g(x))

"""
    init!(model, method)

Initialize the dynamic factor model `model` by `method`.

When `method` is set to `:data`:
- Initialization of the loading matrix is based on principal component analysis (PCA) of the
  data if the loaings matrix is unrestricted and based on the value of the decay parameter
  found in Diebold and Li (2006) if it obeys a Nelson-Siegel structure.
- Initialization of the factor process dynamics is based on the OLS estimates of the
  autoregressions of the PCA factors or Nelson-Siegel factors estimated using the two-step
  approach of Diebold and Li (2006).
- Initialization of the exogeneous mean specification is based on the OLS estimates of the
  slopes based on the residuals, ``yₜ - Λ̂f̂ₜ``.
- Initialization of the simple error model is based on the sample covariance matrix of the
  residuals, ``yₜ - μ̂ₜ - Λ̂f̂ₜ``. For the spatial error models the spatial filter is first
  estimated, after which the sample covariance matrix is calculated on the filtered
  residuals.

When `method` is set to `:none` no initialization is performed and the model is assumed to
have been initialized manually before fitting.
"""
function init!(model::DynamicFactorModel, method::NamedTuple)
    # initialize mean specification
    init!(mean(model), method.mean, data(model))

    # initialize factor component
    init!(process(model), method.factors, data(model) .- mean(mean(model)))

    # initialize error specification
    e = data(model) .- mean(mean(model)) - loadings(model) * factors(model)
    init!(errors(model), method.error, e)

    return nothing
end

# factor process
function init!(F::UnrestrictedStationaryIdentified, method::Symbol, y::AbstractMatrix)
    if method == :data
        # factors and loadings via PCA
        M = fit(PCA, y, maxoutdim = nfactors(F), pratio = 1.0)
        loadings(F) .= projection(M)
        factors(F) .= predict(M, y)

        # factor dynamics
        for (r, f) in pairs(eachrow(factors(F)))
            # objective function
            function objective(x)
                obj = zero(eltype(x))
                for t in 2:nobs(model)
                    obj += (f[t] - x * f[t - 1])^2
                end

                return obj
            end

            # optimize
            res = optimize(objective, -1.0, 1.0)
            dynamics(F).diag[r] = Optim.minimizer(res)
        end
    end

    return nothing
end
function init!(F::UnrestrictedStationaryFull, method::Symbol, y::AbstractMatrix)
    if method == :data
        # factors and loadings via PCA
        M = fit(PCA, y, maxoutdim = nfactors(F), pratio = 1.0)
        loadings(F) .= projection(M)
        factors(F) .= predict(M, y)

        # lag and lead
        @views flag = factors(F)[:, 1:(end - 1)]
        @views flead = factors(F)[:, 2:end]

        # factor dynamics
        dynamics(F) .= flead / flag

        # factor variance
        η = flead - dynamics(F) * flag
        cov(F).data .= cov(η, dims = 2)
    end

    return nothing
end
function init!(F::UnrestrictedUnitRoot, method::Symbol, y::AbstractMatrix)
    if method == :data
        # factors and loadings via PCA
        M = fit(PCA, y, maxoutdim = nfactors(F), pratio = 1.0)
        loadings(F) .= projection(M)
        factors(F) .= predict(M, y)

        # factor variance
        cov(F).diag .= var(factors(F), dims = 2)
    end

    return nothing
end
function init!(F::NelsonSiegelStationary, method::Symbol, y::AbstractMatrix)
    if method == :data
        # factors and decay using Diebold and Li (2006)
        F.λ = 0.0609
        Λ = loadings(F)
        factors(F) .= Λ \ y

        # lag and lead
        @views flag = factors(F)[:, 1:(end - 1)]
        @views flead = factors(F)[:, 2:end]

        # factor dynamics
        dynamics(F) .= flead / flag

        # factor variance
        η = flead - dynamics(F) * flag
        cov(F).mat .= cov(η, dims = 2)
        cov(F).chol.factors .= cholesky(cov(F).mat).factors
    end

    return nothing
end
function init!(F::NelsonSiegelUnitRoot, method::Symbol, y::AbstractMatrix)
    if method == :data
        # factors and decay using Diebold and Li (2006)
        F.λ = 0.0609
        Λ = loadings(F)
        factors(F) .= Λ \ y

        # factor process
        η = diff(factors(F), dims = 2)
        cov(F).diag .= var(η, dims = 2)
    end

    return nothing
end

# mean specification
init!(μ::ZeroMean, method::Symbol, y::AbstractMatrix) = nothing
function init!(μ::Exogenous, method::Symbol, y::AbstractMatrix)
    method == :data && (slopes(μ) .= y / regressors(μ))
end

# error specification
function init!(ε::Simple, method::Symbol, e::AbstractMatrix)
    method == :data && (cov(ε).diag .= var(e, dims = 2))
end
function init!(ε::SpatialAutoregression, method::Symbol, ξ::AbstractMatrix)
    if method == :data
        # spatial filter
        spatial(ε) .= zero(eltype(spatial(ε)))

        # estimate covariance matrix
        e = poly(ε) * ξ
        cov(ε).diag .= var(e, dims = 2)
    end

    return nothing
end
function init!(ε::SpatialMovingAverage, method::Symbol, ξ::AbstractMatrix)
    if method == :data
        # spatial filter
        spatial(ε) .= zero(eltype(spatial(ε)))

        # estimate covariance matrix
        e = poly(ε) \ ξ
        cov(ε).diag .= var(e, dims = 2)
    end

    return nothing
end

function get_params(model::DynamicFactorModel)
    n = size(data(model), 1)
    R = nfactors(model)

    # number of parameters
    n_params = n
    process(model) isa AbstractUnrestrictedFactorProcess && (n_params += (n + 1) * R)
    process(model) isa UnrestrictedStationaryFull && (n_params += R * (2R - 1))
    process(model) isa AbstractNelsonSiegelFactorProcess && (n_params += 1 + 2R^2)
    process(model) isa NelsonSiegelUnitRoot && (n_params -= 2R^2 - R)
    errors(model) isa Union{SpatialAutoregression, SpatialMovingAverage} &&
        (n_params += length(spatial(errors(model))))
    mean(model) isa Exogenous && (n_params += length(slopes(mean(model))))

    # parameters
    θ = zeros(n_params)
    get_params!(θ, model)

    return θ
end

function get_params!(θ::AbstractVector, model::DynamicFactorModel)
    idx = 1

    # loadings
    if process(model) isa AbstractUnrestrictedFactorProcess
        offset = length(loadings(model))
        θ[idx:(idx + offset - 1)] = vec(loadings(model))
        idx += offset
    elseif process(model) isa AbstractNelsonSiegelFactorProcess
        θ[idx] = decay(process(model))
        idx += 1
    end

    # factor dynamics and variance
    if process(model) isa UnrestrictedStationaryIdentified
        # dynamics
        offset = length(dynamics(model).diag)
        θ[idx:(idx + offset - 1)] = dynamics(model).diag
        idx += offset
    elseif process(model) isa UnrestrictedStationaryFull
        # dynamics
        offset = length(dynamics(model))
        θ[idx:(idx + offset - 1)] = vec(dynamics(model))
        idx += offset
        # variance
        offset = length(cov(process(model)))
        θ[idx:(idx + offset - 1)] = vec(cov(process(model)))
        idx += offset
    elseif process(model) isa UnrestrictedUnitRoot
        # variance
        offset = length(cov(process(model)).diag)
        θ[idx:(idx + offset - 1)] = cov(process(model)).diag
        idx += offset
    elseif process(model) isa NelsonSiegelStationary
        # dynamics
        offset = length(dynamics(model))
        θ[idx:(idx + offset - 1)] = vec(dynamics(model))
        idx += offset
        # variance
        offset = length(cov(process(model)))
        θ[idx:(idx + offset - 1)] = vec(cov(process(model)))
        idx += offset
    elseif process(model) isa NelsonSiegelUnitRoot
        # variance
        offset = length(cov(process(model)).diag)
        θ[idx:(idx + offset - 1)] = cov(process(model)).diag
        idx += offset
    end

    # covariance matrix
    offset = length(cov(errors(model)).diag)
    θ[idx:(idx + offset - 1)] = cov(errors(model)).diag
    idx += offset

    # mean
    if mean(model) isa Exogenous
        offset = length(slopes(mean(model)))
        θ[idx:(idx + offset - 1)] = vec(slopes(mean(model)))
        idx += offset
    end

    # spatial dependence
    if errors(model) isa Union{SpatialAutoregression, SpatialMovingAverage}
        offset = length(spatial(errors(model)))
        θ[idx:(idx + offset - 1)] = spatial(errors(model))
    end

    return nothing
end
function set_params!(model::DynamicFactorModel, θ::AbstractVector)
    idx = 1

    # loadings
    if process(model) isa AbstractUnrestrictedFactorProcess
        offset = length(loadings(model))
        vec(loadings(model)) .= view(θ, idx:(idx + offset - 1))
        idx += offset
    elseif process(model) isa AbstractNelsonSiegelFactorProcess
        process(model).λ = θ[idx]
        idx += 1
    end

    # factor dynamics and variance
    if process(model) isa UnrestrictedStationaryIdentified
        # dynamics
        offset = length(dynamics(model).diag)
        dynamics(model).diag .= view(θ, idx:(idx + offset - 1))
        idx += offset
    elseif process(model) isa UnrestrictedStationaryFull
        # dynamics
        offset = length(dynamics(model))
        vec(dynamics(model)) .= view(θ, idx:(idx + offset - 1))
        idx += offset
        # variance
        offset = length(cov(process(model)))
        vec(cov(process(model)).data) .= view(θ, idx:(idx + offset - 1))
        idx += offset
    elseif process(model) isa UnrestrictedUnitRoot
        # variance
        offset = length(cov(process(model)).diag)
        cov(process(model)).diag .= view(θ, idx:(idx + offset - 1))
        idx += offset
    elseif process(model) isa NelsonSiegelStationary
        # dynamics
        offset = length(dynamics(model))
        vec(dynamics(model)) .= view(θ, idx:(idx + offset - 1))
        idx += offset
        # variance
        offset = length(cov(process(model)))
        vec(cov(process(model)).data) .= view(θ, idx:(idx + offset - 1))
        idx += offset
    elseif process(model) isa NelsonSiegelUnitRoot
        # variance
        offset = length(cov(process(model)).diag)
        cov(process(model)).diag .= view(θ, idx:(idx + offset - 1))
        idx += offset
    end

    # covariance matrix
    offset = length(cov(errors(model)).diag)
    cov(errors(model)).diag .= view(θ, idx:(idx + offset - 1))
    idx += offset

    # mean
    if mean(model) isa Exogenous
        offset = length(slopes(mean(model)))
        vec(slopes(mean(model))) .= view(θ, idx:(idx + offset - 1))
        idx += offset
    end

    # spatial dependence
    if errors(model) isa Union{SpatialAutoregression, SpatialMovingAverage}
        offset = length(spatial(errors(model)))
        spatial(errors(model)) .= view(θ, idx:(idx + offset - 1))
    end

    return nothing
end

function loglikelihood(model::DynamicFactorModel)
    # collapsed state space system
    (y_low, d_low, Z_low, H_low, M) = collapse(model, objective = true)
    T = dynamics(model)
    Q = cov(process(model))
    (a1, P1) = state_space_init(model)

    # filter
    (v, F) = _filter_likelihood(y_low, d_low, Z_low, H_low, T, Q, a1, P1)

    # covariance and concentration matrix
    H = cov(model)
    Hinv = inv(cov(model))

    # demeaned data
    y_demeaned = data(model) .- mean(mean(model))

    # log-likelihood
    # constant
    ll = -0.5 * prod(size(data(model))) * log(2π)
    # projection component
    ll += 0.5 * nobs(model) * (logdet(H_low) - logdet(H))
    # GLS residual
    et = similar(y_demeaned, size(data(model), 1))
    for (t, yt) in pairs(eachcol(y_demeaned))
        # filter component
        ll -= 0.5 * (logdet(F[t]) + dot(v[t], F[t] \ v[t]))
        # collapsed component
        mul!(et, M, yt)
        ll -= 0.5 * dot(et, Hinv, et)
    end

    return ll
end

function objective(model::DynamicFactorModel, regularizer::NamedTuple)
    f = loglikelihood(model)
    isnothing(regularizer.factors) || (f -= regularizer.factors(loadings(model)))
    isnothing(regularizer.mean) || (f -= regularizer.mean(slopes(mean(model))))
    isnothing(regularizer.error) || (f -= regularizer.error(cov(errors(model))))

    return f
end

function dof(model::DynamicFactorModel)
    R = nfactors(model)

    # factor component
    process(model) isa AbstractUnrestrictedFactorProcess &&
        (k = sum(!iszero, loadings(model)) + R)
    process(model) isa UnrestrictedStationaryFull && (k += (R * (3R - 1)) ÷ 2)
    process(model) isa AbstractNelsonSiegelFactorProcess && (k = 1 + (R * (3R + 1)) ÷ 2)
    process(model) isa NelsonSiegelUnitRoot && (k -= (R * (3R - 1)) ÷ 2)

    # mean specification
    mean(model) isa Exogenous && (k += sum(!iszero, slopes(mean(model))))

    # error specification
    k += sum(!iszero, cov(errors(model)))
    if errors(model) isa Union{SpatialAutoregression, SpatialMovingAverage}
        k += sum(!iszero, diff(spatial(errors(model))), init = 0) + 1
    end

    return k
end
