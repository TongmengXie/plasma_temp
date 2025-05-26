import numpy as np

def realistic_temperature(r, t):
    """Generate realistic tokamak plasma temperature profile.
    
    Args:
        r (array): Radial positions normalized to [0,1]
        t (array): Time points normalized to [0,1]
    
    Returns:
        Temperature profile that includes:
        - Core peaking
        - Edge pedestal 
        - Time-dependent heating and diffusion
    """
    # Core temperature parameters
    T0_core = 5.0  # Core temperature
    T0_edge = 0.1  # Edge temperature 
    
    # Pedestal parameters
    ped_loc = 0.8  # Pedestal location
    ped_width = 0.1  # Pedestal width
    
    # Time-dependent heating
    heat_rate = 3.0  # Heating rate
    diff_rate = 0.5  # Diffusion rate
    
    # Core profile (gaussian-like)
    core_prof = T0_core * np.exp(-4 * (r/ped_loc)**2)
    
    # Edge pedestal (tanh profile)
    ped_prof = (T0_core - T0_edge) * (1 - np.tanh((r - ped_loc)/ped_width))/2 + T0_edge
    
    # Combine core and edge
    space_prof = np.minimum(core_prof, ped_prof)
    
    # Add time dependence
    # - Heating phase (0 to 0.3)
    # - Steady state (0.3 to 0.7)
    # - Cooling phase (0.7 to 1.0)
    time_factor = np.ones_like(t)
    heat_mask = t < 0.3
    cool_mask = t > 0.7
    
    time_factor[heat_mask] = 0.2 + heat_rate * t[heat_mask]
    time_factor[cool_mask] = 1.0 - diff_rate * (t[cool_mask] - 0.7)
    
    return space_prof * time_factor