module HydroStochRes

export SimulationParameters, 
       double_well_potential, 
       gradient_double_well, 
       gradient_bessel, 
       simulate_sde, 
       calculate_mfpt, 
       plot_potential_and_gradient_norm, 
       plot_stochastic_trajectory, 
       compute_power_spectrum, 
       compute_snr, 
       plot_snr,
       plot_kde,
       plot_psd

include("types.jl")
include("potential.jl")
include("simulation.jl")
include("analysis.jl")
include("plotting.jl")

end
