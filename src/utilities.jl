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
            ft .= dynamics(F) * f_sim[:,t-1] + rand(rng, dist(F))
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

    return SpatialMovingAverage(e_sim, MvNormal(Diagonal(var(ε))), copy(spatial(ε)), weights(ε))
end