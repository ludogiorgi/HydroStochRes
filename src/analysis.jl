using DSP: periodogram
using Statistics: mean

function calculate_mfpt(x_traj::Vector{Float64}, params::SimulationParameters)
    Δt = params.Δt
    crossing_times = []
    in_negative_well = x_traj[1] < 0

    for i in 2:length(x_traj)
        if in_negative_well && x_traj[i] > 0
            push!(crossing_times, i * Δt)
            in_negative_well = false
        elseif !in_negative_well && x_traj[i] < 0
            push!(crossing_times, i * Δt)
            in_negative_well = true
        end
    end

    crossing_intervals = diff(crossing_times)
    return isempty(crossing_intervals) ? 0.0 : mean(crossing_intervals)
end

function compute_power_spectrum(u, params::SimulationParameters)
    dt = params.Δt * params.resolution
    fs = 1 / dt
    u_zero_mean = u .- mean(u)
    p = periodogram(u_zero_mean, fs=fs)
    return p.freq, p.power
end

function compute_snr(freqs, psd, params::SimulationParameters; bandwidth=0.001)
    f_signal = params.ω
    signal_band = (freqs .>= f_signal - bandwidth/2) .& (freqs .<= f_signal + bandwidth/2)
    signal_power = sum(psd[signal_band]) / sum(signal_band)
    noise_power = sum(psd[.!signal_band]) / (length(psd) - sum(signal_band))
    return signal_power / noise_power
end
