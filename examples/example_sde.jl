using Revise
using HydroStochRes
using Plots, ProgressBars
using StatsBase: autocor

params = SimulationParametersSDE(
    2.0,      # l
    1.0,      # alpha
    10.0,     # beta
    0.022,     # sigma1
    0.022,     # sigma2
    1.0,      # m
    100000.0, # T
    0.01,     # Δt
    1,        # resolution
    1.0,      # x0
    0.0,      # y0
    0.0,      # v0x
    0.0,      # v0y
    [-2.0, 2.0], # minima
    0.3,      # A
    1/1000,   # ω
    0.0       # ϕ
)

x_traj, y_traj = simulate_sde(params)
t_traj = [i * params.Δt for i in 1:length(x_traj)]
plt_xy, plt_xt = plot_stochastic_trajectory(x_traj, y_traj, t_traj, params; res=10, n_max=100000)
plot(plt_xy, plt_xt, layout=(2, 1), size=(800, 800))
##


# plot(x_traj[1:100:3000])
# acf = autocor(x_traj, [0:1000:300000...]) 
# plot(acf)
# ##
mfpt = calculate_mfpt(x_traj, params)
println("Mean First Passage Time (MFPT): ", mfpt)

snrs = []
noise = [0.012:0.001:0.05...]

for sigma in ProgressBar(noise)
    params.sigma1 = sigma
    params.sigma2 = sigma
    xt, _ = simulate_sde(params)
    freqs, psd = compute_power_spectrum(xt, params)
    snr = compute_snr(freqs, psd, params)
    push!(snrs, snr)
end

##

plt_xy, plt_xt = plot_stochastic_trajectory(x_traj, y_traj, t_traj, params; res=10, n_max=100000)
plt_snr = plot_snr(noise, snrs)
plt_kde = plot_kde(x_traj, y_traj)
plt_psd = plot_psd(x_traj, params)
plt_Ueff = plot_potential_and_gradient_norm(params)

plot(plt_xy, plt_xt, plt_snr, plt_kde, plt_psd, plt_Ueff, layout=(3, 2), size=(800, 800))
savefig("example_simulation.png")