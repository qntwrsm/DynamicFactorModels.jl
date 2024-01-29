#=
interface.jl

    Provides a collection of interface tools for working with dynamic factor 
    models, such as estimation (w/ and w/o penalization), simulation, 
    forecasting, and variance decomposition. 

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/09/29
=#

"""
    DynamicFactorModel(y, R, μ, ε) -> model

Construct a dynamic factor model with data `y`, mean specification `μ`, error
model `ε`, and `R` dynamic factors.
"""
function DynamicFactorModel(
    y::AbstractMatrix,
    R::Integer,
    μ::AbstractMeanSpecification,
    ε::AbstractErrorModel
)
    (n, T) = size(y)

    # check model specification
    R > n && throw(ArgumentError("R must be less than or equal to n."))

    # instantiate factor process
    Λ = zeros(n, R)
    F = FactorProcess(
        Diagonal{Float64}(undef, R),
        Matrix{Float64}(undef, R, T)
    )

    return DynamicFactorModel(y, μ, ε, Λ, F)
end

"""
    DynamicFactorModel(dims, μ, ε) -> model

Construct a dynamic factor model with dimensions `dims` ``(n, T, R)```, mean
specification `μ`, and error model `ε`.
"""
function DynamicFactorModel(
    dims::Dims,
    μ::AbstractMeanSpecification,
    ε::AbstractErrorModel
)
    (n, T, R) = dims

    # check model specification
    R > n && throw(ArgumentError("R must be less than or equal to n."))

    # instantiate factor process
    Λ = zeros(n, R)
    F = FactorProcess(
        Diagonal{Float64}(undef, R),
        Matrix{Float64}(undef, R, T)
    )

    return DynamicFactorModel(Matrix{Float64}(undef, n, T), μ, ε, Λ, F)
end

"""
    Exogenous(X, n) -> μ

Construct a mean specification with exogenous regressors `X` and `n` time
series.
"""
Exogenous(X::AbstractMatrix, n::Integer) = Exogenous(X, similar(X, (n, size(X, 1))))

"""
    Simple(n, T) -> ε

Construct a simple error model with `n` time series and `T` observations.
"""
Simple(n::Integer, T::Integer) = Simple(Array{Float64}(undef, n, T), MvNormal((1.0I)(n)))

"""
    SpatialAutoregression(n, T, W, spatial=:homo) -> ε

Construct a spatial autoregression error model with `n` time series, `T`
observations, spatial weight matrix `W`, and spatial dependence of type
`spatial`.
"""
function SpatialAutoregression(n::Integer, T::Integer, W::AbstractMatrix, spatial::Symbol=:homo)
    # spatial dependence
    ρ_max = max(inv(opnorm(W, 1)), inv(opnorm(W, Inf)))
    if spatial == :homo
        ρ = zeros(1)
    elseif spatial == :hetero
        ρ = zeros(n)
    else
        throw(ArgumentError("spatial dependence type $spatial not supported."))
    end

    return SpatialAutoregression(Array{Float64}(undef, n, T), MvNormal((1.0I)(n)), ρ, ρ_max, W)
end

"""
    SpatialMovingAverage(n, T, W, spatial=:homo) -> ε

Construct a spatial moving average error model with `n` time series, `T`
observations, spatial weight matrix `W`, and spatial dependence of type
`spatial`.
"""
function SpatialMovingAverage(n::Integer, T::Integer, W::AbstractMatrix, spatial::Symbol=:homo)
    # spatial dependence
    ρ_max = max(inv(opnorm(W, 1)), inv(opnorm(W, Inf)))
    if spatial == :homo
        ρ = zeros(1)
    elseif spatial == :hetero
        ρ = zeros(n)
    else
        throw(ArgumentError("spatial dependence type $spatial not supported."))
    end

    return SpatialMovingAverage(Array{Float64}(undef, n, T), MvNormal((1.0I)(n)), ρ, ρ_max, W)
end

"""
    simulate(model; burn=100, rng=Xoshiro()) -> sim

Simulate data from the dynamic factor model described by `model` and
return a new instance with the simulated data, using random number generator
`rng` and apply a burn-in period of `burn`.
"""
function simulate(model::DynamicFactorModel; burn::Integer=100, rng::AbstractRNG=Xoshiro())
    # factor process
    F_sim = simulate(process(model), burn=burn, rng=rng)

    # error distribution
    ε_sim = simulate(errors(model), rng=rng)

    # simulate data
    y_sim = mean(mean(model)) .+ loadings(model) * factors(F_sim) + resid(ε_sim)

    return DynamicFactorModel(y_sim, copy(mean(model)), ε_sim, copy(loadings(model)), F_sim)
end

"""
    fit!(
        model;
        regularizer=(factors=nothing, mean=nothing, error=nothing),   
        init_method=(factors=:data, mean=:data, error=:data), 
        ϵ=1e-4, 
        max_iter=1000, 
        verbose=false
    ) -> model

Fit the dynamic factor model described by `model` to the data with tolerance `ϵ`
and maximum number of iterations `max_iter`. If `verbose` is true a summary of
the model fitting is printed. `init_method` indicates which method is used for
initialization of the parameters. 'regularizer' indicates the regularization
method used for estimation.

Estimation is done using the Expectation-Maximization algorithm for obtaining
the maximum likelihood estimates of the unregularized model. If a regularizer is
specified, the model is estimated using the Expectation-Maximization algorithm
in combination with a proximal minimization algorithm.
"""
function fit!(
    model::DynamicFactorModel;
    regularizer::NamedTuple=(factors=nothing, mean=nothing, error=nothing),
    init_method::NamedTuple=(factors=:data, mean=:data, error=:data), 
    ϵ::AbstractFloat=1e-4, 
    max_iter::Integer=1000, 
    verbose::Bool=false
)
    keys(regularizer) ⊇ (:factors, :mean, :error) || error("regularizer must be a NamedTuple with keys :factors, :mean, and :error.")
    keys(init_method) ⊇ (:factors, :mean, :error) || error("init_method must be a NamedTuple with keys :factors, :mean, and :error.")

    # model summary
    if verbose
        println("Dynamic factor model")
        println("====================")
        println("Number of series and observations: $(size(model)[1:end-1])")
        println("Number of factors: $(size(model)[end])")
        println("Mean specification: $(Base.typename(typeof(mean(model))).wrapper)")
        println("Error specification: $(Base.typename(typeof(errors(model))).wrapper)")
        println("====================")
        println()
    end
    
    # initialization of model parameters
    init!(model, init_method)

    # instantiate parameter vectors
    θ_prev = params(model)
    θ = similar(θ_prev)

    # optimization
    iter = 0
    converged = false
    dist = Chebyshev()
    while !converged && iter < max_iter
        # update model
        update!(model, regularizer)

        # compute distance metric
        params!(θ, model)
        δ = evaluate(dist, θ, θ_prev)
        copyto!(θ_prev, θ)

        # convergence
        converged = δ < ϵ || δ < ϵ * maximum(abs, θ)

        # update iteration counter
        iter += 1
    end

    # optimization summary
    if verbose
        println("Optimization summary")
        println("====================")
        println("Convergence: ", converged ? "success" : "failed")
        println("Iterations: $iter")
        println("Log-likelihood value: $(loglikelihood(model))")
        println("aic: $(aic(model))")
        println("aicc: $(aicc(model))")
        println("bic: $(bic(model))")
        println("====================")
    end

    return model
end

"""
    forecast(model, periods[, X]) -> forecasts

Forecast `periods` ahead using the dynamic factor model `model`. Future values
of exogeneous regressors `X` should be provided if the mean specification of
`model` is `Exogenous`.
"""
function forecast(model::DynamicFactorModel, periods::Integer)
    mean(model) isa Exogenous && throw(ArgumentError("Exogenous regressors X must be provided if mean specification is Exogenous."))

    # forecast
    (a, _, v, _, K) = filter(model)
    a_next = similar(a[end])
    forecasts = similar(data(model), size(model)[1], periods)
    for h = 1:periods
        if h == 1 
            a_next .= dynamics(model) * a[end] + K[end] * v[end]
        else 
            a_next .= dynamics(model) * a_next
        end
        forecasts[:,h] = mean(mean(model)) .+ loadings(model) * a_next
    end

    return forecasts
end
function forecast(model::DynamicFactorModel, periods::Integer, X::AbstractMatrix)
    !isa(mean(model), Exogenous) && return forecast(model, periods)

    # forecast
    (a, _, v, _, K) = filter(model)
    a_next = similar(a[end])
    forecasts = similar(data(model), size(model)[1], periods)
    for h = 1:periods
        if h == 1 
            a_next .= dynamics(model) * a[end] + K[end] * v[end]
        else 
            a_next .= dynamics(model) * a_next
        end
        forecasts[:,h] = slopes(mean(model)) * X[:,h] + loadings(model) * a_next  
    end

    return forecasts
end

"""
    girf(model, periods) -> irfs

Compute generalized impulse response functions of Koop et al. (1996) `periods`
ahead using the dynamic factor model for a unit shock (i.e. a shock of one
standard deviation in size).
"""
girf(model::DynamicFactorModel, periods::Integer) = stack([loadings(model) * dynamics(model)^h for h = 0:periods])