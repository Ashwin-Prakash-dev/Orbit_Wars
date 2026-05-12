
# Function to build the world state matrix - includes the position for every planet for every step 
" first check if planets rotate or not"
" then "

import math

def build_world_state_matrix(obs):
    """
    Builds a complete lookup table of all planetary coordinates across 500 steps.
    Returns: dict mapping planet_id -> list of (x, y) tuples for steps 0 to 500.
    """
    # 1. Extract root variables 
    initial_planets = obs.get("initial_planets", [])
    omega = obs.get("angular_velocity", 0.0)
    max_steps = 500
    sun_x, sun_y = 50.0, 50.0
    
    matrix = {}

    # 2. Outer loop: Process each base planet
    for p in initial_planets:
        # Unpack raw list representation: [id, owner, x, y, radius, ships, production]
        p_id = p[0]
        x0 = p[2]
        y0 = p[3]
        p_radius = p[4]
        
        # Calculate distance from central Sun
        dx = x0 - sun_x
        dy = y0 - sun_y
        orbital_radius = math.hypot(dx, dy)
        
        # Pre-allocate array for 501 steps (0 through 500)
        trajectory = [None] * (max_steps + 1)
        
        # Check if Static Planet
        if orbital_radius + p_radius >= 50.0:
            # Since static planets never move - duplicate initial coordinates
            matrix[p_id] = [(x0, y0)] * (max_steps + 1)
            continue
            
        # 3. Inner loop: Orbiting Planets 
        theta_0 = math.atan2(dy, dx)
        
        for step in range(max_steps + 1):
            theta_step = theta_0 + (omega * step)
            x_step = sun_x + orbital_radius * math.cos(theta_step)
            y_step = sun_y + orbital_radius * math.sin(theta_step)
            trajectory[step] = (x_step, y_step)
            
        matrix[p_id] = trajectory
        
    return matrix