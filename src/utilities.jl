#=
utilities.jl

    Provides a collection of utility tools for working with dynamic factor 
    models, such as simulation, filtering, and smoothing. 

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/12/01
=#

"""
    simulate(F; burn=100, rng=Xoshiro()) -> sim

Simulate the dynamic factors from the dynamic factor process `F`
with a burn-in period of `burn` using the random number generator `rng`.
"""
function simulate(F::FactorProcess; burn::Integer=100, rng::AbstractRNG=Xoshiro())
    # burn-in
    f_prev = randn(rng, size(F))
    f_next = similar(f_prev)
    for _ = 1:burn
        f_next .= dynamics(F) * f_prev + randn(rng, size(F))
        f_prev .= f_next
    end

    # simulate data
    f_sim = similar(factors(F))
    for (t, ft) ∈ pairs(eachcol(f_sim))
        if t == 1
            ft .= f_prev
        else
            ft .= dynamics(F) * f_sim[:,t-1] + randn(rng, size(F))
        end
    end

    return FactorProcess(copy(dynamics(F)), f_sim)
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

    return SpatialAutoregression(e_sim, MvNormal(Diagonal(var(ε))), copy(spatial(ε)), ε.ρ_max, weights(ε))
end
function simulate(ε::SpatialMovingAverage; rng::AbstractRNG=Xoshiro())
    e_sim = similar(resid(ε))
    for et ∈ eachcol(e_sim)
        mul!(et, poly(ε), rand(rng, dist(ε)))
    end

    return SpatialMovingAverage(e_sim, MvNormal(Diagonal(var(ε))), copy(spatial(ε)), ε.ρ_max, weights(ε))
end

"""
    state_space(model) -> (y_star, Z_star, d_star, a1, P1)

State space form of the collapsed dynamic factor model `model`.
"""
function state_space(model::DynamicFactorModel)
    R = size(process(model))
    Ty = eltype(data(model))

    # projection components
    if errors(model) isa SpatialAutoregression
        Hinv = poly(errors(model))' * (cov(errors(model)) \ poly(errors(model)))
        Zt_Hinv = loadings(model)' * Hinv
    else
        Zt_Hinv = (cov(model)' \ loadings(model))'
    end
    Zt_Hinv_Z = Zt_Hinv * loadings(model)
    
    # Cholesky decomposition
    C = cholesky(Hermitian(inv(Zt_Hinv_Z)))

    # collapsing projection matrix
    A_star = C.U * Zt_Hinv

    # collapsing
    y_star = [A_star * yt for yt ∈ eachcol(data(model))]
    Z_star = inv(C.L)
    if mean(model) isa ZeroMean
        d_star = [Zeros(Ty, R) for _ ∈ axes(data(model), 2)]
    elseif mean(model) isa Exogenous
        d_star = [A_star * μt for μt ∈ eachcol(mean(mean(model)))]
    end

    # initial conditions
    a1 = zeros(Ty, R)
    P1 = reshape((I - kron(dynamics(model), dynamics(model))) \ vec(I(R)), R, R)

    return (y_star, Z_star, d_star, a1, P1)
end

"""
    filter(model) -> (a, P, v, F, K)

Collapsed Kalman filter for the dynamic factor model `model`. Returns the
filtered state `a`, covariance `P`, forecast error `v`, forecast error variance
`F`, and Kalman gain `K`.
"""
function filter(model::DynamicFactorModel)
    # collapsed state space system
    (y, Z, d, a1, P1) = state_space(model)
    T = dynamics(model)

    # initialize filter output
    a = similar(y, typeof(a1))
    P = similar(y, typeof(P1))
    v = similar(y)
    F = similar(y, typeof(P1))
    K = similar(y, typeof(P1))

    # initialize filter
    a[1] = a1
    P[1] = P1

    # filter
    for t ∈ eachindex(y)
        # forecast error
        v[t] = y[t] - Z * a[t] - d[t]
        F[t] = Z * P[t] * Z' + I

        # Kalman gain
        K[t] = T * P[t] * Z' / F[t]

        # prediction
        if t < length(y)
            a[t+1] = T * a[t] + K[t] * v[t]
            P[t+1] = T * P[t] * (T - K[t] * Z)' + I
        end
    end

    return (a, P, v, F, K)
end

"""
    smoother(model) -> (α̂, V, Γ)

Collapsed Kalman smoother for the dynamic factor model `model`. Returns the
smoothed state `α̂`, covariance `V`, and autocovariance `Γ`.
"""
function smoother(model::DynamicFactorModel)
    # collapsed state space system
    (y, Z, _, a1, P1) = state_space(model)
    T = dynamics(model)

    # filter
    (a, P, v, F, K) = filter(model)

    # initialize smoother output
    α̂ = similar(a)
    V = similar(P)
    Γ = similar(P, length(y) - 1)

    # initialize smoother
    r = zero(a1)
    N = zero(P1)
    L = similar(P1)

    # smoother
    for t ∈ reverse(eachindex(y))
        L .= T - K[t] * Z

        # backward recursion
        r .= Z' / F[t] * v[t] + L' * r
        N .= Z' / F[t] * Z + L' * N * L

        # smoothing
        α̂[t] = a[t] + P[t] * r
        V[t] = P[t] - P[t] * N * P[t]
        t > 1 && (Γ[t-1] = I - P[t] * N)
        t < length(y) && (Γ[t] *= L * P[t])
    end

    return (α̂, V, Γ)
end