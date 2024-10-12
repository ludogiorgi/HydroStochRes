using Random
using SpecialFunctions

# Euler-Maruyama-Stratonovich integration for the stochastic process
function simulate_sde(params::SimulationParametersSDE)

    l, alpha, beta, sigma1, sigma2, m, T, Δt = params.l, params.alpha, params.beta, params.sigma1, params.sigma2, params.m, params.T, params.Δt
    x0, y0, v0x, v0y, minima, A, ω, ϕ = params.x0, params.y0, params.v0x, params.v0y, params.minima, params.A, params.ω, params.ϕ

    n_steps = Int(T / Δt)
    x, y, vx, vy, t = x0, y0, v0x, v0y, 0.0
    x_traj, y_traj = [x], [y]

    for i in 1:n_steps
        dU_dx, dU_dy = gradient_double_well(x, y, l, alpha)
        grad_bessel_x, grad_bessel_y = gradient_bessel(x, y, minima, beta)

        ξ1x, ξ1y = sqrt(Δt) * randn(), sqrt(Δt) * randn()
        ξ2x, ξ2y = sqrt(Δt) * randn(), sqrt(Δt) * randn()

        force_x = A * cos(2π * ω * t + ϕ)
        vx_mid = vx + 0.5 * Δt * (-dU_dx / m - vx / m + force_x / m) + 0.5 * (sigma1 * ξ1x / m + sigma2 * grad_bessel_x * ξ2x / m)
        vy_mid = vy + 0.5 * Δt * (-dU_dy / m - vy / m) + 0.5 * (sigma1 * ξ1y / m + sigma2 * grad_bessel_y * ξ2y / m)

        x_mid, y_mid = x + 0.5 * Δt * vx_mid, y + 0.5 * Δt * vy_mid
        grad_bessel_x_mid, grad_bessel_y_mid = gradient_bessel(x_mid, y_mid, minima, beta)

        vx = vx + Δt * (-dU_dx / m - vx_mid / m + force_x / m) + sigma1 * ξ1x / m + sigma2 * grad_bessel_x_mid * ξ2x / m
        vy = vy + Δt * (-dU_dy / m - vy_mid / m) + sigma1 * ξ1y / m + sigma2 * grad_bessel_y_mid * ξ2y / m

        x, y = x + Δt * vx_mid, y + Δt * vy_mid
        t += Δt
        if i % params.resolution == 0
            push!(x_traj, x)
            push!(y_traj, y)
        end
    end

    return x_traj, y_traj
end

# Function to compute the time derivative of the state vector
function f!(du, u, grad_h_term, l, alpha, k₀, t, A, ω, ϕ)
    x_p = u[1:2]
    v_p = u[3:4]
    dU = gradient_double_well(x_p[1], x_p[2], l, alpha)
    du[1:2] .= v_p                      
    du[3:4] .= - 2 * grad_h_term .- v_p .- dU
    du[3] += A * cos(2π * ω * t + ϕ)
    du[3:4] ./= k₀
end

# RK4 method to solve the system of ODEs with in-place updates
function rk4_step!(u, du, t, Δt, grad_h_term, memory_cutoff, past_positions, l, alpha, k₀, τ, A, ω, ϕ)
    k1 = similar(du)
    k2 = similar(du)
    k3 = similar(du)
    k4 = similar(du)
    
    compute_grad_h!(grad_h_term, u[1:2], t, memory_cutoff, past_positions, Δt, τ)
    f!(k1, u, grad_h_term, l, alpha, k₀, t, A, ω, ϕ)
    
    compute_grad_h!(grad_h_term, u[1:2] .+ Δt / 2 * k1[1:2], t + Δt / 2, memory_cutoff, past_positions, Δt, τ)
    f!(k2, u .+ Δt / 2 * k1, grad_h_term, l, alpha, k₀, t, A, ω, ϕ)
    
    compute_grad_h!(grad_h_term, u[1:2] .+ Δt / 2 * k2[1:2], t + Δt / 2, memory_cutoff, past_positions, Δt, τ)
    f!(k3, u .+ Δt / 2 * k2, grad_h_term, l, alpha, k₀, t, A, ω, ϕ)
    
    compute_grad_h!(grad_h_term, u[1:2] .+ Δt * k3[1:2], t + Δt, memory_cutoff, past_positions, Δt, τ)
    f!(k4, u .+ Δt * k3, grad_h_term, l, alpha, k₀, t, A, ω, ϕ)
    
    u .+= Δt / 6 * (k1 .+ 2 * k2 .+ 2 * k3 .+ k4) #+ 0.1 * sqrt(Δt) * randn(4)
end

# Time integration using RK4 with a sliding window for memory
function simulate_stroboscopic(params::SimulationParametersStroboscopic)
    l, alpha, k₀, τ, T, Δt, x0, y0, v0x, v0y, memory_cutoff, A, ω, ϕ = params.l, params.alpha, params.k₀, params.τ, params.T, params.Δt, params.x0, params.y0, params.v0x, params.v0y, params.memory_cutoff, params.A, params.ω, params.ϕ 
    t0 = 0.0
    tf = T
    u = [x0, y0, v0x, v0y]
    du = zeros(size(u))  
    grad_h_term = zeros(2) 
    num_steps = Int(memory_cutoff / Δt) + 1
    past_positions = []
    times = collect(t0:Δt:tf)  
    positions = zeros(length(times), 2)  
    positions[1, :] .= u[1:2]
    push!(past_positions, copy(u[1:2]))
    
    for i in eachindex(times[2:end])
        t = times[i]
        push!(past_positions, copy(u[1:2]))
        if length(past_positions) > num_steps
            popfirst!(past_positions)
        end
        compute_grad_h!(grad_h_term, u[1:2], t, memory_cutoff, past_positions, Δt, τ)
        rk4_step!(u, du, t, Δt, grad_h_term, memory_cutoff, past_positions, l, alpha, k₀, τ, A, ω, ϕ)
        positions[i, :] .= u[1:2]
    end

    return times, positions
end
