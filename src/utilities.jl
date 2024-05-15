#=
utilities.jl

    Provides a collection of utility tools for working with dynamic factor 
    models, such as simulation, filtering, and smoothing. 

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/12/01
=#

"""
    select_sample(model, sample) -> model

Select a sample `sample` of the data from the dynamic factor model `model`.
"""
function select_sample(model::DynamicFactorModel, sample::AbstractUnitRange)
    μ = select_sample(mean(model), sample)
    ε = select_sample(errors(model), sample)
    F = select_sample(process(model), sample)

    return DynamicFactorModel(data(model)[:,sample], μ, ε, F)
end

"""
    select_sample(μ, sample) -> μ

Select a sample `sample` of the data from the mean model `μ`.
"""
select_sample(μ::ZeroMean, sample::AbstractUnitRange) = ZeroMean(μ.type, μ.n)
select_sample(μ::Exogenous, sample::AbstractUnitRange) = Exogenous(regressors(μ)[:,sample], size(slopes(μ), 1))

"""
    select_sample(ε, sample) -> ε

Select a sample `sample` of the data from the error model `ε`.
"""
select_sample(ε::Simple, sample::AbstractUnitRange) = Simple(size(resid(ε), 1), length(sample), type=eltype(resid(ε)))
select_sample(ε::SpatialAutoregression, sample::AbstractUnitRange) = SpatialAutoregression(size(resid(ε), 1), length(sample), copy(weights(ε)), spatial=length(spatial(ε)) == 1 ? :homo : :hetero, type=eltype(resid(ε)))
select_sample(ε::SpatialMovingAverage, sample::AbstractUnitRange) = SpatialMovingAverage(size(resid(ε), 1), length(sample), copy(weights(ε)), spatial=length(spatial(ε)) == 1 ? :homo : :hetero, type=eltype(resid(ε)))

"""
    select_sample(F, sample) -> F

Select a sample `sample` of the data from the dynamic factor process `F`.
"""
select_sample(F::UnrestrictedStationaryIdentified, sample::AbstractUnitRange) = UnrestrictedStationary((size(loadings(F), 1), length(sample), size(F)), dependence=:identified, type=eltype(factors(F)))
select_sample(F::UnrestrictedStationaryFull, sample::AbstractUnitRange) = UnrestrictedStationary((size(loadings(F), 1), length(sample), size(F)), dependence=:full, type=eltype(factors(F)))
select_sample(F::UnrestrictedUnitRoot, sample::AbstractUnitRange) = UnrestrictedUnitRoot((size(loadings(F), 1), length(sample), size(F)), type=eltype(factors(F)))
select_sample(F::NelsonSiegelStationary, sample::AbstractUnitRange) = NelsonSiegelStationary(length(sample), maturities(F), type=eltype(factors(F)))
select_sample(F::NelsonSiegelUnitRoot, sample::AbstractUnitRange) = NelsonSiegelUnitRoot(length(sample), maturities(F), type=eltype(factors(F))) 

"""
    simulate(F; burn=100, rng=Xoshiro()) -> sim

Simulate the dynamic factors from the dynamic factor process `F`
with a burn-in period of `burn` using the random number generator `rng`.
"""
function simulate(F::AbstractFactorProcess; burn::Integer=100, rng::AbstractRNG=Xoshiro())
    # burn-in
    f_prev = rand(rng, dist(F))
    f_next = similar(f_prev)
    for _ = 1:burn
        f_next .= dynamics(F) * f_prev + rand(rng, dist(F))
        f_prev .= f_next
    end

    # simulate data
    F_sim = copy(F)
    for (t, ft) ∈ pairs(eachcol(factors(F_sim)))
        if t == 1
            ft .= f_prev
        else
            ft .= dynamics(F) * factors(F_sim)[:,t-1] + rand(rng, dist(F))
        end
    end

    return F_sim
end

"""
    simulate(ε; rng=Xoshiro()) -> sim

Simulate from the error distribution `ε` using the random number generator 
`rng`.
"""
simulate(ε::Simple; rng::AbstractRNG=Xoshiro()) = Simple(rand(rng, dist(ε), size(resid(ε), 2)), MvNormal(Diagonal(var(ε))))
function simulate(ε::SpatialAutoregression; rng::AbstractRNG=Xoshiro())
    e_sim = similar(resid(ε))
    for et ∈ eachcol(e_sim)
        et .= poly(ε) \ rand(rng, dist(ε))
    end

    return SpatialAutoregression(e_sim, MvNormal(Diagonal(var(ε))), copy(spatial(ε)), ε.ρ_max, copy(weights(ε)))
end
function simulate(ε::SpatialMovingAverage; rng::AbstractRNG=Xoshiro())
    e_sim = similar(resid(ε))
    for et ∈ eachcol(e_sim)
        mul!(et, poly(ε), rand(rng, dist(ε)))
    end

    return SpatialMovingAverage(e_sim, MvNormal(Diagonal(var(ε))), copy(spatial(ε)), ε.ρ_max, copy(weights(ε)))
end

"""
    collapse(model) -> (A_low, Z_basis)

Low-dimensional collapsing matrices for the dynamic factor model `model`
following the approach of Jungbacker and Koopman (2015).
"""
function collapse(model::DynamicFactorModel)
    # active factors
    active = [!all(iszero, λ) for λ ∈ eachcol(loadings(model))]

    # collapsing matrix
    H = cov(model)
    if all(active)
        C = (H \ loadings(model))' * loadings(model)
        Z_basis = (C \ loadings(model)')'
    else
        Z_basis = loadings(model)[:,active]
    end
    A_low = (H \ Z_basis)'

    return (A_low, Z_basis)
end

"""
    state_space(model) -> (y_low, Z_low, d_low, H_low, a1, P1)

State space form of the collapsed dynamic factor model `model` following the
approach of Jungbacker and Koopman (2015).
"""
function state_space(model::DynamicFactorModel)
    R = size(process(model))
    Ty = eltype(data(model))

    # active factors
    active = [!all(iszero, λ) for λ ∈ eachcol(loadings(model))]

    # collapsing matrix
    if any(active)
        (A_low, _) = collapse(model)
    else
        A_low = I
    end

    # collapsing
    y_low = A_low * data(model)
    Z_low = A_low * loadings(model)
    if mean(model) isa ZeroMean
        d_low = Zeros(mean(model).type, size(y_low))
    elseif mean(model) isa Exogenous
        d_low = A_low * mean(mean(model))
    end
    H_low = A_low * cov(model) * A_low'

    # initial conditions
    a1 = zeros(Ty, R)
    P1 = Matrix{Ty}(I, R, R)

    return (y_low, Z_low, d_low, H_low, a1, P1)
end

"""
    _filter(y, Z, d, H, T, Q, a1, P1) -> (a, P, v, F, K)

Kalman filter for the state space system `y`, `Z`, `d`, `H`, `T`, `Q`, `a1`,
and `P1`.
"""
function _filter(y, Z, d, H, T, Q, a1, P1)
    # initialize filter output
    n = size(y, 2)
    a = Vector{typeof(a1)}(undef, n)
    P = Vector{typeof(P1)}(undef, n)
    v = Vector{typeof(a1)}(undef, n)
    F = Vector{typeof(P1)}(undef, n)
    K = Vector{typeof(P1)}(undef, n)

    # initialize filter
    a[1] = a1
    P[1] = P1

    # filter
    for (t, yt) ∈ pairs(eachcol(y))
        # forecast error
        v[t] = yt - Z * a[t] - view(d, :, t)
        F[t] = Z * P[t] * Z' + H

        # Kalman gain
        K[t] = T * P[t] * (F[t] \ Z)'

        # prediction
        if t < n
            a[t+1] = T * a[t] + K[t] * v[t]
            P[t+1] = T * P[t] * (T - K[t] * Z)' + Q
        end
    end

    return (a, P, v, F, K)
end

"""
    filter(model) -> (a, P, v, F, K)

Collapsed Kalman filter for the dynamic factor model `model`. Returns the
filtered state `a`, covariance `P`, forecast error `v`, forecast error variance
`F`, and Kalman gain `K`.
"""
function filter(model::DynamicFactorModel)
    # collapsed state space system
    (y, Z, d, H, a1, P1) = state_space(model)
    T = dynamics(model)
    Q = cov(process(model))

    return _filter(y, Z, d, H, T, Q, a1, P1)
end

"""
    smoother(model) -> (α̂, V, Γ)

Collapsed Kalman smoother for the dynamic factor model `model`. Returns the
smoothed state `α̂`, covariance `V`, and autocovariance `Γ`.
"""
function smoother(model::DynamicFactorModel)
    # collapsed state space system
    (y, Z, d, H, a1, P1) = state_space(model)
    T = dynamics(model)
    Q = cov(process(model))

    # filter
    (a, P, v, F, K) = _filter(y, Z, d, H, T, Q, a1, P1)

    # initialize smoother output
    α̂ = similar(a)
    V = similar(P)
    Γ = similar(P, length(a) - 1)

    # initialize smoother
    r = zero(a1)
    N = zero(P1)
    L = similar(P1)

    # smoother
    for t ∈ reverse(eachindex(a))
        L .= T - K[t] * Z

        # backward recursion
        r .= (F[t] \ Z)' * v[t] + L' * r
        N .= (F[t] \ Z)' * Z + L' * N * L

        # smoothing
        α̂[t] = a[t] + P[t] * r
        V[t] = P[t] - P[t] * N * P[t]
        t > 1 && (Γ[t-1] = I - P[t] * N)
        t < length(a) && (Γ[t] *= L * P[t])
    end

    return (α̂, V, Γ)
end

"""
    forecasts(μ, periods) -> forecasts

Forecast the mean model `μ` `periods` ahead.
"""
forecast(μ::AbstractMeanSpecification, periods::Integer) = error("forecast for $(Base.typename(typeof(μ)).wrapper) not implemented.")
forecast(μ::ZeroMean, periods::Integer) = Zeros(μ.type, μ.n, periods)