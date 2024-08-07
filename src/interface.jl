#=
interface.jl

    Provides a collection of interface tools for working with dynamic factor 
    models, such as estimation (w/ and w/o penalization), simulation, 
    forecasting, and variance decomposition. 

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/09/29
=#

"""
    DynamicFactorModel(dims, μ, ε, F; type=Float64) -> model

Construct a dynamic factor model with dimensions `dims` ``(n, T)``, element type
`type`, mean specification `μ`, error model `ε`, and factor process `F`.
"""
function DynamicFactorModel(
    dims::Dims,
    μ::AbstractMeanSpecification,
    ε::AbstractErrorModel,
    F::AbstractFactorProcess;
    type::Type=Float64
)
    return DynamicFactorModel(Matrix{type}(undef, dims), μ, ε, F)
end

"""
    Exogenous(X, n) -> μ

Construct a mean specification with exogenous regressors `X` and `n` time
series.
"""
Exogenous(X::AbstractMatrix, n::Integer) = Exogenous(X, similar(X, (n, size(X, 1))))

"""
    Simple(n, T; type=Float64) -> ε

Construct a simple error model with `n` time series, `T` observations, and
element type `type`.
"""
Simple(n::Integer, T::Integer; type::Type=Float64) = Simple(Array{type}(undef, n, T), MvNormal(one(type)I(n)))

"""
    SpatialAutoregression(n, T, W; spatial=:homo, type=Float64) -> ε

Construct a spatial autoregression error model with `n` time series, `T`
observations, element type `type`, spatial weight matrix `W`, and spatial
dependence of type `spatial`.
"""
function SpatialAutoregression(n::Integer, T::Integer, W::AbstractMatrix; spatial::Symbol=:homo, type::Type=Float64)
    # spatial dependence
    ρ_max = max(inv(opnorm(W, 1)), inv(opnorm(W, Inf)))
    if spatial == :homo
        ρ = zeros(type, 1)
    elseif spatial == :hetero
        ρ = zeros(type, n)
    else
        throw(ArgumentError("spatial dependence type $spatial not supported."))
    end

    return SpatialAutoregression(Array{type}(undef, n, T), MvNormal(one(type)I(n)), ρ, ρ_max, W)
end

"""
    SpatialMovingAverage(n, T, W; spatial=:homo, type=Float64) -> ε

Construct a spatial moving average error model with `n` time series, `T`
observations, element type `type`, spatial weight matrix `W`, and spatial
dependence of type `spatial`.
"""
function SpatialMovingAverage(n::Integer, T::Integer, W::AbstractMatrix; spatial::Symbol=:homo, type::Type=Float64)
    # spatial dependence
    ρ_max = max(inv(opnorm(W, 1)), inv(opnorm(W, Inf)))
    if spatial == :homo
        ρ = zeros(type, 1)
    elseif spatial == :hetero
        ρ = zeros(type, n)
    else
        throw(ArgumentError("spatial dependence type $spatial not supported."))
    end

    return SpatialMovingAverage(Array{type}(undef, n, T), MvNormal(one(type)I(n)), ρ, ρ_max, W)
end

"""
    UnrestrictedStationary(dims; dependence=:identified, type=Float64) -> F

Construct a stationary factor process with unrestricted loadings of dimensions
`dims`, dependence of factors `dependence`, and with element types `type`.
"""
function UnrestrictedStationary(dims::Dims; dependence::Symbol=:identified, type::Type=Float64)
    (n, T, R) = dims
    Λ = Matrix{type}(undef, n, R)
    f = Matrix{type}(undef, R, T)

    if dependence == :identified
        ϕ = Diagonal{type}(undef, R)
        dist = MvNormal(Zeros{type}(R), one(type)I)
        F = UnrestrictedStationaryIdentified
    elseif dependence == :full
        ϕ = Matrix{type}(undef, R, R)
        dist = MvNormal(Matrix(one(type)I(R)))
        F = UnrestrictedStationaryFull
    else
        throw(ArgumentError("dependence of factors $dependence not supported."))
    end
    
    return F(Λ, ϕ, f, dist)
end

"""
    UnrestrictedUnitRoot(dims; type=Float64) -> F

Construct a unit-root factor process with unrestricted loadings of dimensions
`dims` and with element types `type`.
"""
function UnrestrictedUnitRoot(dims::Dims; type::Type=Float64)
    (n, T, R) = dims
    Λ = Matrix{type}(undef, n, R)
    f = Matrix{type}(undef, R, T)
    dist = MvNormal(Diagonal{type}(ones(type, R)))
    
    return UnrestrictedUnitRoot(Λ, f, dist)
end

"""
    NelsonSiegelStationary(T, τ; type=Float64) -> F

Construct a stationary Nelson-Siegel factor process for maturities `τ` with
`T` time series observations and element types `type`.
"""
function NelsonSiegelStationary(T::Integer, τ::AbstractVector; type::Type=Float64)
    λ = 0.0609  
    ϕ = Matrix{type}(undef, 3, 3)
    f = Matrix{type}(undef, 3, T)
    dist = MvNormal(Matrix(one(type)I(3)))
    
    return NelsonSiegelStationary(λ, τ, ϕ, f, dist)
end

"""
    NelsonSiegelUnitRoot(T, τ; type=Float64) -> F

Construct a unit-root Nelson-Siegel factor process for maturities `τ` with
`T` time series observations and element types `type`.
"""
function NelsonSiegelUnitRoot(T::Integer, τ::AbstractVector; type::Type=Float64)
    λ = 0.0609 
    f = Matrix{type}(undef, 3, T)
    dist = MvNormal(Diagonal{type}(ones(type, 3)))
    
    return NelsonSiegelUnitRoot(λ, τ, f, dist)
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

    return DynamicFactorModel(y_sim, copy(mean(model)), ε_sim, F_sim)
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
method used for fitting.

Fitting is done using the Expectation-Maximization algorithm for obtaining
the maximum likelihood estimates of the unregularized model. If a regularizer is
specified, the model is estimated using the Expectation-Maximization algorithm
in combination with a proximal minimization algorithm.
"""
function fit!(
    model::DynamicFactorModel;
    regularizer::NamedTuple=(factors=nothing, mean=nothing, error=nothing),
    init_method::NamedTuple=(factors=:data, mean=:data, error=:data), 
    ϵ::AbstractFloat=1e-4, 
    max_iter::Integer=1_000, 
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
        println("Factor specification: $(Base.typename(typeof(process(model))).wrapper)")
        println("Mean specification: $(Base.typename(typeof(mean(model))).wrapper)")
        println("Error specification: $(Base.typename(typeof(errors(model))).wrapper)")
        println("====================")
        println()
    end
    
    # initialization of model parameters
    init!(model, init_method)

    # optimization
    iter = 0
    obj = -Inf
    converged = false
    violation = false
    while !converged && iter < max_iter
        # update model
        update!(model, regularizer)

        # update objective function
        obj_prev = obj
        obj = objective(model, regularizer)

        # non-decrease violation
        if obj - obj_prev < 0
            violation = true
            if verbose
                println("Objective function value decreased from $iter to $(iter + 1).")
                println()
            end
            break
        end

        # convergence
        δ = 2 * abs(obj - obj_prev) / (abs(obj) + abs(obj_prev))
        converged = δ < ϵ

        # update iteration counter
        iter += 1
    end

    # optimization summary
    if verbose
        println("Optimization summary")
        println("====================")
        println("Convergence: ", converged ? "success" : "failed")
        println("Non-decrease violation: $violation")
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
    model_tuning_ic!(model, regularizers; ic=:bic, parallel=false, verbose=false, kwargs...) -> (model_opt, index_opt)

Search for the optimal regularizer in `regularizers` for the dynamic factor
model `model` using information criterion `ic`. If `parallel` is true, the
search is performed in parallel. If `verbose` is true, a summary of model tuning
and progress of the search is printed. Additional keyword arguments `kwargs` are
passed to the `fit!` function.
"""
function model_tuning_ic!(
    model::DynamicFactorModel,
    regularizers::AbstractArray;
    ic::Symbol=:bic,
    parallel::Bool=false,
    verbose::Bool=false,
    kwargs...
)
    ic ∉ (:aic, :aicc, :bic) && error("Information criterion $ic not supported.")

    if verbose
        println("Model tuning summary")
        println("====================")
        println("Number of regularizers: $(length(regularizers))")
        println("Information criterion: $ic")
        println("Parallel: $(parallel ? "yes" : "no")")
        println("====================")
    end

    # model tuning
    map_func = parallel ? verbose ? progress_pmap : pmap : verbose ? progress_map : map
    θ0 = params(model)
    f0 = copy(factors(model))
    θ = map_func(regularizers) do regularizer
        try
            params!(model, θ0)
            factors(model) .= f0
            fit!(model, regularizer=regularizer; kwargs...)
            params(model)
        catch
            missing
        end
    end
    ic_values = map(θ) do θi
        if all(ismissing.(θi))
            missing
        else
            params!(model, θi)
            eval(ic)(model)
        end
    end
    (ic_opt, index_opt) = findmin(x -> isnan(x) ? Inf : x, skipmissing(ic_values))
    params!(model, θ[index_opt])
    (α̂, _, _) = smoother(model)
    for (t, α̂t) ∈ pairs(α̂)
        factors(model)[:,t] = α̂t
    end

    if verbose
        println("====================")
        println("Optimal regularizer index: $(index_opt)")
        println("Optimal information criterion: $(ic_opt)")
        println("Failed fits: $(sum(ismissing.(ic_values)))")
        println("====================")
    end
    
    return (model, index_opt)
end

"""
    model_tuning_cv!(model, regularizers, blocks, periods; parallel=false, verbose=false, kwargs...) -> (model_opt, index_opt)

Search for the optimal regularizer in `regularizers` for the dynamic factor
model `model` using cross-validation with out-of-sample consisting of `blocks`
blocks with `periods` period ahead forecasts. If `parallel` is true, the search
is performed in parallel. If `verbose` is true, a summary of model tuning and
progress of the search is printed. Additional keyword arguments `kwargs` are
passed to the `fit!` function.
"""
function model_tuning_cv!(
    model::DynamicFactorModel,
    regularizers::AbstractArray,
    blocks::Integer,
    periods::Integer;
    parallel::Bool=false,
    verbose::Bool=false,
    kwargs...
)
    if verbose
        println("Model tuning summary")
        println("====================")
        println("Number of regularizers: $(length(regularizers))")
        println("Number of out-of-sample blocks: $blocks")
        println("Forecast periods per block: $periods")
        println("Parallel: $(parallel ? "yes" : "no")")
        println("====================")
    end

    # train-test split
    (n, T, _) = size(model)
    T_train = T - blocks * periods
    train_model = select_sample(model, 1:T_train)
    test_models = [select_sample(model, 1:t) for t = T_train:T-periods]

    # model tuning
    map_func = parallel ? verbose ? progress_pmap : pmap : verbose ? progress_map : map
    θ0 = params(model)
    f0 = factors(model)[:,1:T_train]
    θ = map_func(regularizers) do regularizer
        try
            params!(train_model, θ0)
            factors(train_model) .= f0
            fit!(train_model, regularizer=regularizer; kwargs...)
            params(train_model)
        catch
            missing
        end
    end
    msfe = map(θ) do θi
        if all(ismissing.(θi))
            missing
        else
            e_sq = zero(eltype(data(model)))
            for (t, test_model) ∈ pairs(test_models)
                params!(test_model, θi)
                oos_range = (T_train + t):(T_train + t + periods - 1)
                e_sq += sum(abs2, view(data(model), :, oos_range) - forecast(test_model, periods))
            end
            e_sq / (n * length(test_models) * periods)
        end
    end
    (msfe_opt, index_opt) = findmin(x -> isnan(x) ? Inf : x, skipmissing(msfe))
    fit!(model, regularizer=regularizers[index_opt]; kwargs...)

    if verbose
        println("====================")
        println("Optimal regularizer index: $(index_opt)")
        println("Optimal forecast accuracy: $(msfe_opt)")
        println("Failed fits: $(sum(ismissing.(msfe)))")
        println("====================")
    end
    
    return (model, index_opt)
end

"""
    forecast(model, periods) -> forecasts

Forecast `periods` ahead using the dynamic factor model `model`.
"""
function forecast(model::DynamicFactorModel, periods::Integer)
    # forecast
    (a, _, v, _, K) = filter(model)
    a_next = similar(a[end])
    forecasts = similar(data(model), size(model)[1], periods)
    μ_hat = forecast(mean(model), periods)
    for h = 1:periods
        if h == 1 
            a_next .= dynamics(model) * a[end] + K[end] * v[end]
        else 
            a_next .= dynamics(model) * a_next
        end
        forecasts[:,h] = μ_hat[:,h] + loadings(model) * a_next
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