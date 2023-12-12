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
  (PCA) of the data.
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
    # initialize factor process
    if method.factors == :data
        # PCA
        M = fit(PCA, data(model), maxoutdim=size(process(model)), pratio=1.0)
        loadings(model) .= projection(M)
        factors(model) .= transform(M, data(model))
    end

    # initialize mean specification
    init!(mean(model), method.mean, data(model) - loadings(model) * factors(model))

    # initialize error specification
    resid(model) .= data(model) - mean(model) - loadings(model) * factors(model)
    init!(errors(model), method.error)
        
    return nothing
end


"""
    init!(μ, method, y)

Initialize the mean specification `μ` by `method` based on the data `y`.
"""
init!(μ::ZeroMean, method::Symbol, y::AbstractMatrix) = nothing
init!(μ::Exogenous, method::Symbol, y::AbstractMatrix) = method == :data && slopes(μ) .= y / regressors(μ)


"""
    init!(ε, method)

Initialize the error model `ε` by `method`.
"""
init!(ε::Simple, method::Symbol) = method == :data && cov(ε).diag .= var(resid(ε), dims=2)
function init!(ε::SpatialAutoregression, method::Symbol)
    if method == :data
        # estimate spatial filter
        if length(spatial(ε)) == 1
            spatial(ε) .= dot(weights(ε) * resid(ε), resid(ε)) / sum(abs2, weights(ε) * resid(ε))
        else
            for i ∈ eachindex(spatial(ε))
                spatial(ε)[i] = dot(weights(ε)[i,:] * resid(ε), resid(ε)[i,:]) / sum(abs2, weights(ε)[i,:] * resid(ε))
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
            spatial(ε) .= dot(weights(ε) * resid(ε), resid(ε)) / sum(abs2, weights(ε) * resid(ε))
        else
            for i ∈ eachindex(spatial(ε))
                spatial(ε)[i] = dot(weights(ε)[i,:] * resid(ε), resid(ε)[i,:]) / sum(abs2, weights(ε)[i,:] * resid(ε))
            end
        end

        # estimate covariance matrix
        resid(ε) .= poly(ε) \ resid(ε)
        cov(ε).diag .= var(resid(ε), dims=2)
    end

    return nothing
end

"""
    absdiff(x, y) -> δ

Calculate the maximum absolute difference between `x` and `y`.
"""
absdiff(x::AbstractArray, y::AbstractArray) = mapreduce((xi, yi) -> abs(xi - yi), max, x, y)