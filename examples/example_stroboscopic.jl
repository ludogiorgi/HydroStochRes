using SpecialFunctions   # For Bessel functions J_1
using Plots              # For plotting
using LinearAlgebra      # For norm
using ProgressBars       # For progress bars
using KernelDensity

# Define parameters
k₀ = 2.0    # mass
τ = 8    # memory decay constant
Δt = 0.1  # time step

# Memory cutoff for integration, to avoid negligible contributions from -∞
memory_cutoff = 5.0 * τ  # After this time, the exponential decay becomes negligible

# Define the gradient of the pilot wave using J_1
function grad_pilot_wave!(grad, x, xp, t, s)
    r = norm(x - xp)
    
    if r == 0.0
        grad .= 0.0  # Avoid division by zero, set gradient to zero
        return
    end
    grad .= - (x - xp) / r * besselj(1, r) * exp(-(t - s) / τ)
end

function gradient_double_well(x::Float64, y::Float64; l = 2.0, alpha = 1.0)
    dU_dx = 4 * x * (x^2 / l^2 - 1) / l^2  
    dU_dy = 2 * alpha * y                   
    return (dU_dx, dU_dy)
end

# Function to compute the full integral for the gradient of h(x, t) over the window [t - cutoff, t]
function compute_grad_h!(grad_h_term, x_p, t, memory_cutoff, past_positions)
    grad_h_term .= 0.0  # Reset the gradient term

    # Numerical integration over the range [t - memory_cutoff, t]
    for (i, s) in enumerate(max(0.0, t - memory_cutoff):Δt:t)
        temp_grad = zeros(2)  # Temporary gradient value
        
        # x_p corresponds to the current position at time t
        # past_positions[i] corresponds to the past position at time s
        grad_pilot_wave!(temp_grad, x_p, past_positions[i], t, s)
        grad_h_term .+= temp_grad  # Sum contributions from each time step
    end
    grad_h_term *= Δt  # Scale by the time step
end

# Function to compute the time derivative of the state vector
function f!(du, u, grad_h_term, t)
    x_p = u[1:2]
    v_p = u[3:4]
    dU = gradient_double_well(x_p[1], x_p[2])
    
    # Compute the derivatives: velocity (dx_p) and acceleration (dv_p)
    du[1:2] .= v_p                      
    du[3:4] .= (- 2 * grad_h_term .- v_p .- 30 .*dU) / k₀  
end

# RK4 method to solve the system of ODEs with in-place updates
function rk4_step!(u, du, t, Δt, grad_h_term, memory_cutoff, past_positions)
    k1 = similar(du)
    k2 = similar(du)
    k3 = similar(du)
    k4 = similar(du)
    
    # Recompute the gradient of h at each RK4 substep
    compute_grad_h!(grad_h_term, u[1:2], t, memory_cutoff, past_positions)
    f!(k1, u, grad_h_term, t)
    
    compute_grad_h!(grad_h_term, u[1:2] .+ Δt / 2 * k1[1:2], t + Δt / 2, memory_cutoff, past_positions)
    f!(k2, u .+ Δt / 2 * k1, grad_h_term, t + Δt / 2)
    
    compute_grad_h!(grad_h_term, u[1:2] .+ Δt / 2 * k2[1:2], t + Δt / 2, memory_cutoff, past_positions)
    f!(k3, u .+ Δt / 2 * k2, grad_h_term, t + Δt / 2)
    
    compute_grad_h!(grad_h_term, u[1:2] .+ Δt * k3[1:2], t + Δt, memory_cutoff, past_positions)
    f!(k4, u .+ Δt * k3, grad_h_term, t + Δt)
    
    u .+= Δt / 6 * (k1 .+ 2 * k2 .+ 2 * k3 .+ k4)
end

# Time integration using RK4 with a sliding window for memory
function integrate_rk4(u0, tspan, Δt, memory_cutoff)
    t0, tf = tspan
    t = t0
    u = copy(u0)
    du = zeros(size(u0))  # Preallocate array for derivatives
    grad_h_term = zeros(2)  # Preallocate gradient term
    
    # Preallocate for past positions, store positions in a sliding window
    num_steps = Int(memory_cutoff / Δt) + 1
    past_positions = [copy(u0[1:2]) for i in 1:num_steps]  # Store past positions
    
    times = collect(t0:Δt:tf)  # Precompute time steps
    positions = zeros(length(times), 2)  # Preallocate positions array
    positions[1, :] .= u0[1:2]  # Store initial position
    
    for i in ProgressBar(2:length(times))
        t = times[i]
        
        # Update the past positions for use in the integral
        push!(past_positions, copy(u[1:2]))  # Add the current position to past_positions
        if length(past_positions) > num_steps
            popfirst!(past_positions)  # Remove the oldest position
        end
        
        # Recompute the full integral of the gradient at each time step
        compute_grad_h!(grad_h_term, u[1:2], t, memory_cutoff, past_positions)
        
        # Perform one RK4 step
        rk4_step!(u, du, t, Δt, grad_h_term, memory_cutoff, past_positions)
        
        # Store current position
        positions[i, :] .= u[1:2]
    end
    
    return times, positions
end

# Initial conditions: [x_p1, x_p2, v_p1, v_p2]
u0 = [randn(), randn(), randn(), randn()]  # Initial position and velocity

# Time span
tspan = (0.0, 1000.0)

# Perform RK4 integration
times, positions = integrate_rk4(u0, tspan, Δt, memory_cutoff)

# Plot the results
plt1 = plot(positions[:, 1], positions[:, 2], label="Particle Trajectory", xlabel="x", ylabel="y", legend=:top)
plt2 = plot(times, positions[:, 1], label="x vs. t", xlabel="t", ylabel="x", legend=:top)
plot(plt1, plt2, layout=(2, 1), size=(800, 800))

##

kde_xy = kde(positions[1:end,1:2])
plt_kde = heatmap(kde_xy.x, kde_xy.y, kde_xy.density', xlabel="x", ylabel="y", title="PDF", color=:inferno)

##
kde_x = kde(positions[5:end, 1])
plt_kde_x = plot(kde_x.x, kde_x.density, xlabel="x", ylabel="PDF", title="PDF of x", color=:inferno)