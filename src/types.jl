mutable struct SimulationParametersSDE
    l::Float64
    alpha::Float64
    beta::Float64
    sigma1::Float64
    sigma2::Float64
    m::Float64
    T::Float64
    Δt::Float64
    resolution::Int
    x0::Float64
    y0::Float64
    v0x::Float64
    v0y::Float64
    minima::Vector{Float64}
    A::Float64
    ω::Float64
    ϕ::Float64
end

mutable struct SimulationParametersStroboscopic
    l::Float64
    alpha::Vector{Float64}
    k₀::Float64
    τ::Float64
    T::Float64
    Δt::Float64
    x0::Float64
    y0::Float64
    v0x::Float64
    v0y::Float64
    memory_cutoff::Float64
    minima::Vector{Float64}
    A::Float64
    ω::Float64
    ϕ::Float64
end