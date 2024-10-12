# Double-well potential with minima along y=0 and located at x = ±l
function double_well_potential(x::Float64, y::Float64, l::Float64, alpha::Float64, B::Float64)
    U_x = alpha * ((x^2 / l^2) - 1)^2
    U_y = alpha * y^2
    return B * (U_x + U_y)
end

# Gradient of the potential at (x, y)
function gradient_double_well(x::Float64, y::Float64, l::Float64, alpha::Vector{Float64})
    dU_dx = 4 * alpha[1] * x * (x^2 / l^2 - 1) / l^2  
    dU_dy = 2 * alpha[2] * y                   
    return (dU_dx, dU_dy)
end

# Gradient of the Bessel function J_0 with respect to x and y
function gradient_bessel(x::Float64, y::Float64, minima::Vector{Float64}, beta::Float64)
    r = sqrt(minimum(abs.(x .- minima))^2 + y^2)  
    if r == 0
        return (0.0, 0.0)  
    end
    factor = -beta*besselj(1, beta*r) / r  
    dJ0_dx = factor * x  
    dJ0_dy = factor * y  
    return (dJ0_dx, dJ0_dy)
end

# Gradient of the pilot wave function at (x, y) and time t
function grad_pilot_wave!(grad, x, xp, t, s, τ)
    r = norm(x - xp)
    if r == 0.0
        grad .= 0.0  
        return
    end
    grad .= - (x - xp) / r * besselj(1, r) * exp(-(t - s) / τ)
end

# Function to compute the full integral for the gradient of h(x, t) over the window [t - cutoff, t]
function compute_grad_h!(grad_h_term, x_p, t, memory_cutoff, past_positions, Δt, τ)
    grad_h_term .= 0.0
    for (i, s) in enumerate(max(0.0, t - memory_cutoff):Δt:t)
        temp_grad = zeros(2)
        grad_pilot_wave!(temp_grad, x_p, past_positions[i], t, s, τ)
        grad_h_term .+= temp_grad
    end
    grad_h_term *= Δt
end