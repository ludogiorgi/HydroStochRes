using Plots, KernelDensity, LinearAlgebra

function plot_potential_and_gradient_norm(params::SimulationParameters)
    l, alpha, beta, minima = params.l, params.alpha, params.beta, params.minima
    x = range(-2*l, 2*l, length=100)
    y = range(-1*l, 1*l, length=100)
    U = [double_well_potential(xi, yi, l, alpha) for yi in y, xi in x]
    gradient_norm = [norm(gradient_double_well(xi, yi, l, alpha)) for yi in y, xi in x]
    bessel_U_norm = [params.sigma2*norm(gradient_bessel(xi, yi, minima, beta))^2/2 for yi in y, xi in x]

    plt = heatmap(x, y, U + bessel_U_norm, xlabel="x", ylabel="y", title="Double-Well Effective Potential", color=:inferno, clims=(0, 10))
    # p2 = heatmap(x, y, gradient_norm, xlabel="x", ylabel="y", title="Gradient Norm of Double-Well Potential", color=:plasma)
    # p3 = heatmap(x, y, bessel_U_norm, xlabel="x", ylabel="y", title="Bessel Potential", color=:viridis)
    return plt
end

function plot_stochastic_trajectory(x_traj, y_traj, t_traj, params::SimulationParameters; res=1, n_max=100000)
    A, ω, ϕ = params.A, params.ω, params.ϕ
    plt_xy = plot(x_traj[1:res:n_max*res], y_traj[1:res:n_max*res], legend=false, xlabel="x", ylabel="y", title="y vs x (noise=0.022)", lw=0.2)
    plt_xt = plot(t_traj[1:res:n_max*res], x_traj[1:res:n_max*res], label="trajectory", xlabel="t", ylabel="x", title="x vs t (noise=0.022)", lw=0.5)
    plt_xt = plot!(t_traj[1:res:n_max*res], A .* cos.(2π * ω .* t_traj[1:res:n_max*res] .+ ϕ), lw=3, label="forcing")
    return plt_xy, plt_xt
end

function plot_snr(noise, snrs)
    plt_snr = plot(noise, snrs, marker=:o, xlabel="Noise Intensity", ylabel="SNR",
    title="Signal-to-Noise Ratio vs. Noise Intensity", legend=false)
    return plt_snr
end

function plot_kde(x_traj, y_traj)
    kde_xy = kde([x_traj y_traj])
    plt_kde = heatmap(kde_xy.x, kde_xy.y, kde_xy.density', xlabel="x", ylabel="y", title="PDF", color=:inferno)
    return plt_kde
end

function plot_psd(x_traj, params::SimulationParameters)
    ps_f, ps_p = compute_power_spectrum(x_traj, params)
    plot(ps_f[2:1:100000], ps_p[2:1:100000], xscale=:log10, yscale=:log10, xlabel="Frequency", ylabel="Power Spectral Density", title="Power Spectrum", legend=false)
end
