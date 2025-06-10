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
    state_space_init(model) -> (a1, P1)

State space initial conditions of the state space form of the dynamic factor model `model`.
"""
function state_space_init(model::DynamicFactorModel)
    T = eltype(data(model))
    R = nfactors(model)
    a1 = zeros(T, R)
    P1 = Matrix{T}(I, R, R)

    return (a1, P1)
end

"""
    collapse(model; objective = false) -> (y_low, d_low, Z_low, H_low[, M])

Collapsed system components for the collapsed state space form of the dynamic factor model
`model` following the approach of Jungbacker and Koopman (2015). Optional `objective`
boolean indicating whether the collapsed system components are used for objective function
(log-likelihood) computation, in which case additionally the annihilator matrix is returned.
"""
function collapse(model::DynamicFactorModel; objective::Bool = false)
    # linearly independent columns
    F = qr(loadings(model), ColumnNorm())
    ic = isapprox.(diag(F.R), 0.0, atol = 1e-8)

    # collapsing matrices
    Z_basis = loadings(model)[:, .!ic]
    if all(ic)
        A_low = I
    else
        A_low = (cov(model) \ Z_basis)'
    end

    # collapsed system
    y_low = [A_low * yt for yt in eachcol(data(model))]
    Z_low = A_low * loadings(model)
    if mean(model) isa ZeroMean
        d_low = [Zeros(mean(model).type, size(Z_low, 1)) for _ in 1:nobs(model)]
    elseif mean(model) isa Exogenous
        d_low = [A_low * μt for μt in eachcol(mean(mean(model)))]
    end
    H_low = A_low * cov(model) * A_low'

    # annihilator matrix for log-likelihood
    if objective
        if all(ic)
            M = Zeros(eltype(data(model)), size(data(model), 1))
        else
            M = I - Z_basis * (H_low \ A_low)
        end

        return (y_low, d_low, Z_low, H_low, M)
    else
        return (y_low, d_low, Z_low, H_low)
    end
end

"""
    filter(model; predict = false) -> (a, P, v, F)

Collapsed Kalman filter for the dynamic factor model `model`. Returns the filtered state `a`
, covariance `P`, forecast error `v`, and forecast error variance `F`. If `predict` is
`true` the filter reports the one-step ahead out-of-sample prediction.
"""
function filter(model::DynamicFactorModel; predict::Bool = false)
    # collapsed state space system
    (y, d, Z, H) = collapse(model)
    T = dynamics(model)
    Q = cov(process(model))
    (a1, P1) = state_space_init(model)

    # initialize filter output
    n = length(y) + (predict ? 1 : 0)
    a = similar(y, typeof(a1), n)
    P = similar(y, typeof(P1), n)
    v = similar(y)
    F = similar(y, typeof(P1))

    # initialize storage
    ZtFinv = similar(P1)
    att = similar(a1)
    Ptt = similar(P1)

    # initialize filter
    a[1] = a1
    P[1] = P1

    # filter
    for t in eachindex(y)
        # forecast error
        v[t] = y[t] - d[t] - Z * a[t]
        F[t] = Z * P[t] * Z' + H

        # update
        ZtFinv .= (F[t] \ Z)'
        att .= a[t] + P[t] * ZtFinv * v[t]
        Ptt .= P[t] - P[t] * ZtFinv * Z * P[t]

        # prediction
        if predict || t < length(y)
            a[t + 1] = T * att
            P[t + 1] = T * Ptt * T' + Q
            # enforce symmetry for numerical stability
            P[t + 1] = 0.5 * (P[t + 1] + P[t + 1]')
        end
    end

    return (a, P, v, F)
end

"""
    _filter_smoother(y, d, Z, H, T, Q, a1, P1) -> (a, P, v, ZtFinv)

Collapsed Kalman filter for the dynamic factor model used internally by the `smoother`
routine to avoid duplicate expensive computation of state space system matrices.
"""
function _filter_smoother(y, d, Z, H, T, Q, a1, P1)
    # initialize filter output
    a = similar(y, typeof(a1))
    P = similar(y, typeof(P1))
    v = similar(y)
    ZtFinv = similar(y, typeof(P1))

    # initialize storage
    F = similar(H)
    att = similar(a1)
    Ptt = similar(P1)

    # initialize filter
    a[1] = a1
    P[1] = P1

    # filter
    for t in eachindex(y)
        # forecast error
        v[t] = y[t] - d[t] - Z * a[t]
        F .= Z * P[t] * Z' + H

        # update
        ZtFinv[t] = (F \ Z)'
        att .= a[t] + P[t] * ZtFinv[t] * v[t]
        Ptt .= P[t] - P[t] * ZtFinv[t] * Z * P[t]

        # prediction
        if t < length(y)
            a[t + 1] = T * att
            P[t + 1] = T * Ptt * T' + Q
            # enforce symmetry for numerical stability
            P[t + 1] = 0.5 * (P[t + 1] + P[t + 1]')
        end
    end

    return (a, P, v, ZtFinv)
end

"""
    _filter_likelihood(y, d, Z, H, T, Q, a1, P1) -> (v, F)

Collapsed Kalman filter for the dynamic factor model used internally by the `loglikelihood`
routine to avoid duplicate expensive computation of collapsing components and state space
system matrices.
"""
function _filter_likelihood(y, d, Z, H, T, Q, a1, P1)
    # initialize filter output
    v = similar(y)
    F = similar(y, typeof(H))

    # initialize storage
    ZtFinv = similar(Z')
    att = similar(a1)
    Ptt = similar(P1)

    # initialize filter
    a = copy(a1)
    P = copy(P1)

    # filter
    for t in eachindex(y)
        # forecast error
        v[t] = y[t] - d[t] - Z * a
        F[t] = Z * P * Z' + H

        # update
        ZtFinv .= (F[t] \ Z)'
        att .= a + P * ZtFinv * v[t]
        Ptt .= P - P * ZtFinv * Z * P

        # prediction
        if t < length(y)
            a .= T * att
            P = T * Ptt * T' + Q
            # enforce symmetry for numerical stability
            P = 0.5 * (P + P')
        end
    end

    return (v, F)
end

"""
    smoother(model) -> (α, V, Γ)

Collapsed Kalman smoother for the dynamic factor model `model`. Returns the smoothed state
`α`, covariance `V`, and autocovariance `Γ`.
"""
function smoother(model::DynamicFactorModel)
    # collapsed state space system
    (y, d, Z, H) = collapse(model)
    T = dynamics(model)
    Q = cov(process(model))
    (a1, P1) = state_space_init(model)

    # filter
    (a, P, v, ZtFinv) = _filter_smoother(y, d, Z, H, T, Q, a1, P1)

    # initialize smoother output
    α = similar(a)
    V = similar(P)
    Γ = similar(P, length(y) - 1)

    # initialize smoother
    r = zero(a[1])
    N = zero(P[1])
    L = similar(P[1])

    # smoother
    for t in reverse(eachindex(y))
        L .= T - T * P[t] * ZtFinv[t] * Z

        # backward recursion
        r .= ZtFinv[t] * v[t] + L' * r
        N .= ZtFinv[t] * Z + L' * N * L

        # smoothing
        α[t] = a[t] + P[t] * r
        V[t] = P[t] - P[t] * N * P[t]
        t > 1 && (Γ[t - 1] = I - P[t] * N)
        t < length(y) && (Γ[t] *= L * P[t])
    end

    return (α, V, Γ)
end

"""
    forecasts(μ, periods) -> forecasts

Forecast the mean model `μ` `periods` ahead.
"""
function forecast(μ::AbstractMeanSpecification, periods::Integer)
    error("forecast for $(Base.typename(typeof(μ)).wrapper) not implemented.")
end
forecast(μ::ZeroMean, periods::Integer) = Zeros(μ.type, μ.n, periods)
