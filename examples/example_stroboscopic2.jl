using Revise
using HydroStochRes
using Plots, ProgressBars
using StatsBase: autocor
using KernelDensity
plotly()

params = SimulationParametersStroboscopic(
    2.0,       # l
    [100.0, 50.0],      # alpha
    2.5,       # k₀
    20.0,      # τ
    5000.0,    # T
    0.1,       # Δt
    2.1,       # x0
    0.1,       # y0
    1/3,       # v0x
    1/7,       # v0y
    50.0,      # memory_cutoff
    [-2.0, 2.0], # minima
    30.0,      # A
    1/100,   # ω
    0.0       # ϕ
)
params.memory_cutoff = params.τ * 5

times, positions = simulate_stroboscopic(params)

mfpt = calculate_mfpt(positions[:,1], params)
println("Mean First Passage Time (MFPT): ", mfpt)

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

##
ps_f_35, ps_p_35 = compute_power_spectrum(positions[:,1], params)
snr = compute_snr(ps_f_35, ps_p_35, params)
println(snr)
plot(ps_f_05[2:1:end], ps_p_05[2:1:end], xscale=:log10, yscale=:log10, xlabel="Frequency", ylabel="Power Spectral Density", title="Power Spectrum", legend=false)
plot!(ps_f_35[2:1:end], ps_p_35[2:1:end], xscale=:log10, yscale=:log10, xlabel="Frequency", ylabel="Power Spectral Density", title="Power Spectrum", legend=false)
##
bandwidth = 0.001
f_signal = params.ω
freqs = ps_f_05
psd = ps_p_05
signal_band_1 = (freqs .>= f_signal - bandwidth/2) .& (freqs .<= f_signal + bandwidth/2)
odd_multiples = [f_signal * (2n + 1) for n in 0:floor(Int, maximum(freqs) / (2 * f_signal))]
signal_band_odds = reduce((a, b) -> a .| b, [(freqs .>= f - bandwidth/2) .& (freqs .<= f + bandwidth/2) for f in odd_multiples])
signal_power = sum(psd[signal_band_1]) / sum(signal_band_1)
noise_power = sum(psd[.!signal_band_odds]) / (length(psd) - sum(signal_band_odds))

snr = signal_power / noise_power


##

snrs = []
taus = [5.0:1.0:60.0...]

for tau in ProgressBar(taus)
    params.τ = tau
    params.memory_cutoff = tau * 5
    times, positions = simulate_stroboscopic(params)
    xt = positions[:,1]
    freqs, psd = compute_power_spectrum(xt, params)
    snr = compute_snr(freqs, psd, params)
    push!(snrs, snr)
end

##

plot(taus, snrs, marker=:o, xlabel="Noise Intensity", ylabel="SNR",
    title="Signal-to-Noise Ratio vs. Noise Intensity", legend=false)

##
Threads.nthreads()