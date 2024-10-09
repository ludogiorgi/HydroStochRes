using Random
using SpecialFunctions

# Euler-Maruyama-Stratonovich integration for the stochastic process
function simulate_sde(params::SimulationParameters)

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
