# Double-well potential with minima along y=0 and located at x = ±l
function double_well_potential(x::Float64, y::Float64, l::Float64, alpha::Float64)
    U_x = ((x^2 / l^2) - 1)^2
    U_y = alpha * y^2
    return U_x + U_y
end

# Gradient of the potential at (x, y)
function gradient_double_well(x::Float64, y::Float64, l::Float64, alpha::Float64)
    dU_dx = 4 * x * (x^2 / l^2 - 1) / l^2  
    dU_dy = 2 * alpha * y                   
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