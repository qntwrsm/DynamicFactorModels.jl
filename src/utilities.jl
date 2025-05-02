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
    ε = copy(errors(model))
    F = select_sample(process(model), sample)

    return DynamicFactorModel(data(model)[:, sample], μ, ε, F)
end

"""
    select_sample(μ, sample) -> μ

Select a sample `sample` of the data from the mean model `μ`.
"""
select_sample(μ::ZeroMean, sample::AbstractUnitRange) = ZeroMean(μ.type, μ.n)
function select_sample(μ::Exogenous, sample::AbstractUnitRange)
    Exogenous(regressors(μ)[:, sample], size(slopes(μ), 1))
end

"""
    select_sample(F, sample) -> F

Select a sample `sample` of the data from the dynamic factor process `F`.
"""
function select_sample(F::UnrestrictedStationaryIdentified, sample::AbstractUnitRange)
    UnrestrictedStationary((size(loadings(F), 1), length(sample), size(F)),
                           dependence = :identified, type = eltype(factors(F)))
end
function select_sample(F::UnrestrictedStationaryFull, sample::AbstractUnitRange)
    UnrestrictedStationary((size(loadings(F), 1), length(sample), size(F)),
                           dependence = :full, type = eltype(factors(F)))
end
function select_sample(F::UnrestrictedUnitRoot, sample::AbstractUnitRange)
    UnrestrictedUnitRoot((size(loadings(F), 1), length(sample), size(F)),
                         type = eltype(factors(F)))
end
function select_sample(F::NelsonSiegelStationary, sample::AbstractUnitRange)
    NelsonSiegelStationary(length(sample), maturities(F), type = eltype(factors(F)))
end
function select_sample(F::NelsonSiegelUnitRoot, sample::AbstractUnitRange)
    NelsonSiegelUnitRoot(length(sample), maturities(F), type = eltype(factors(F)))
end

"""
    simulate(F, S; rng = Xoshiro()) -> f

Simulate the dynamic factors from the dynamic factor process `F` `S` times using the random
number generator `rng`.
"""
function simulate(F::AbstractFactorProcess, S::Integer; rng::AbstractRNG = Xoshiro())
    R = size(loadings(F), 2)
    f = similar(factors(F), R, S)
    dist = MvNormal(cov(F))
    for (s, fs) in pairs(eachcol(f))
        if s == 1
            # initial condition
            fs .= rand(rng, dist)
        else
            fs .= dynamics(F) * f[:, s - 1] + rand(rng, dist)
        end

    end

    return f
end

"""
    simulate(ε, S; rng = Xoshiro()) -> e

Simulate from the error distribution `ε` `S` times using the random number generator `rng`.
"""
function simulate(ε::Simple, S::Integer; rng::AbstractRNG = Xoshiro())
    rand(rng, MvNormal(cov(ε)), S)
end
function simulate(ε::SpatialAutoregression, S::Integer; rng::AbstractRNG = Xoshiro())
    e = rand(rng, MvNormal(cov(ε)), S)
    G = poly(ε)
    for es in eachcol(e)
        es .= G \ es
    end

    return e
end
function simulate(ε::SpatialMovingAverage, S::Integer; rng::AbstractRNG = Xoshiro())
    e = rand(rng, MvNormal(cov(ε)), S)
    G = poly(ε)
    for es in eachcol(e)
        es .= G * es
    end

    return e
end

"""
    collapse(model) -> (A_low, Z_basis)

Low-dimensional collapsing matrices for the dynamic factor model `model`
following the approach of Jungbacker and Koopman (2015).
"""
function collapse(model::DynamicFactorModel)
    # active factors
    active = [!all(iszero, λ) for λ in eachcol(loadings(model))]

    # collapsing matrix
    H = cov(model)
    if all(active)
        C = (H \ loadings(model))' * loadings(model)
        Z_basis = (C \ loadings(model)')'
    else
        Z_basis = loadings(model)[:, active]
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
    active = [!all(iszero, λ) for λ in eachcol(loadings(model))]

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
    for (t, yt) in pairs(eachcol(y))
        # forecast error
        v[t] = yt - Z * a[t] - view(d, :, t)
        F[t] = Z * P[t] * Z' + H

        # Kalman gain
        K[t] = T * P[t] * (F[t] \ Z)'

        # prediction
        if t < n
            a[t + 1] = T * a[t] + K[t] * v[t]
            P[t + 1] = T * P[t] * (T - K[t] * Z)' + Q
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
    for t in reverse(eachindex(a))
        L .= T - K[t] * Z

        # backward recursion
        r .= (F[t] \ Z)' * v[t] + L' * r
        N .= (F[t] \ Z)' * Z + L' * N * L

        # smoothing
        α̂[t] = a[t] + P[t] * r
        V[t] = P[t] - P[t] * N * P[t]
        t > 1 && (Γ[t - 1] = I - P[t] * N)
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
