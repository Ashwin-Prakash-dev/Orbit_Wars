
import math

_planet_matrix = None
_fleet_tracker = None

# Function to build the world state matrix - includes the position for every planet for every step 
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

class FleetTracker:
    def __init__(self, sun_x=50.0, sun_y=50.0):
        self.sun_x = sun_x
        self.sun_y = sun_y
        
        # O(1) Active Registry: mapping fleet_id -> FleetProfile dict
        self.active_fleets = {}

    def calculate_speed(self, ships):
        """Calculates exact scalar velocity using the engine's logarithmic curve."""
        # Speed scales logarithmically from 1.0 to 6.0 based on ship count
        return 1.0 + (6.0 - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5

    def update(self, current_step, fleets_obs, global_planet_matrix):
        """
        Synchronizes the registry with the current environment observation.
        Must be called at the start of every turn.
        """
        # 1. Track currently visible IDs for fast set-based pruning
        current_visible_ids = set()

        # 2. Process all airborne fleets
        for raw_fleet in fleets_obs:
            # Unpack observation: [id, owner, x, y, angle, from_planet_id, ships]
            f_id = raw_fleet[0]
            owner = raw_fleet[1]
            fx = raw_fleet[2]
            fy = raw_fleet[3]
            angle = raw_fleet[4]
            ships = raw_fleet[6]
            
            current_visible_ids.add(f_id)

            # If fleet is already registered, skip heavy math
            if f_id in self.active_fleets:
                continue

            # --- NEW FLEET DETECTED: Execute one-time math ---
            speed = self.calculate_speed(ships)
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)

            # Resolve target by ray-casting against the global planetary matrix
            target_planet_id, impact_step = self._resolve_target(
                current_step, fx, fy, vx, vy, speed, global_planet_matrix
            )

            # Bundle into a persistent cached profile
            self.active_fleets[f_id] = {
                "owner": owner,
                "start_x": fx,
                "start_y": fy,
                "vx": vx,
                "vy": vy,
                "speed": speed,
                "ships": ships,
                "spawn_step": current_step,
                "target_planet_id": target_planet_id,  # None if decoy/out-of-bounds
                "impact_step": impact_step             # Precise step of expected arrival
            }

        # 3. Prune dead fleets (Struck planets, hit sun, or left map)
        # Any ID in our registry that is missing from the current observation is dead.
        dead_ids = self.active_fleets.keys() - current_visible_ids
        for d_id in list(dead_ids):
            del self.active_fleets[d_id]

    def _resolve_target(self, start_step, fx, fy, vx, vy, speed, planet_matrix):
        """
        Projects the fleet ray forward to find spatio-temporal intersections 
        with pre-computed planetary positions.
        """
        # Scan forward steps up to the absolute simulation limit
        max_lookahead = 500 - start_step
        
        for dt in range(1, max_lookahead + 1):
            eval_step = start_step + dt
            
            # Project linear fleet position at step dt
            ray_x = fx + (vx * dt)
            ray_y = fy + (vy * dt)
            
            # Fast out-of-bounds check
            if not (0 <= ray_x <= 100 and 0 <= ray_y <= 100):
                return None, None
                
            # Fast Sun collision check (Radius 10 at 50,50)
            if math.hypot(ray_x - self.sun_x, ray_y - self.sun_y) <= 10.0:
                return None, None

            # Check if this ray coordinate intersects any planet's pre-computed position
            for p_id, trajectory in planet_matrix.items():
                if eval_step < len(trajectory):
                    px, py = trajectory[eval_step]
                    
                    # Assume average radius check (or pass actual planet radii mapping)
                    # If distance between ray and planet center is <= radius, we have a hit
                    if math.hypot(ray_x - px, ray_y - py) <= 3.0: # Rough bounding box
                        return p_id, eval_step
                        
        return None, None

    def get_incoming_threats(self, planet_id):
        """Returns all hostile fleets projected to strike a specific planet."""
        threats = []
        for f in self.active_fleets.values():
            if f["target_planet_id"] == planet_id:
                threats.append(f)
        return sorted(threats, key=lambda x: x["impact_step"])
    

def calculate_fleet_arrival(launch_planet_id, target_planet_id, fleet_size, current_step, obs, planet_matrix):
    
    # Look up launch and target planets from obs
    launch_x, launch_y, launch_radius = None, None, None
    target_radius = 2.0  # fallback if target not found

    for p in obs.get("planets", []):
        if p[0] == launch_planet_id:
            launch_x, launch_y, launch_radius = p[2], p[3], p[4]
        if p[0] == target_planet_id:
            target_radius = p[4]  # FIX 1: actual radius, not hardcoded 2.0

    if launch_x is None:
        return None, None

    # Logarithmic fleet speed
    speed = 1.0 + (6.0 - 1.0) * (math.log(fleet_size) / math.log(1000.0)) ** 1.5

    target_trajectory = planet_matrix.get(target_planet_id)
    if not target_trajectory:
        return None, None

    sun_x, sun_y, sun_radius = 50.0, 50.0, 10.0

    for future_step in range(current_step + 1, len(target_trajectory)):
        tx, ty = target_trajectory[future_step]
        dx, dy = tx - launch_x, ty - launch_y
        total_distance = math.hypot(dx, dy)

        if total_distance == 0:
            continue

        # FIX 3: subtract both radii — fleet spawns outside launch radius,
        # collision triggers at target boundary
        travel_distance = total_distance - launch_radius - target_radius

        if travel_distance <= 0:
            continue  # already overlapping, shouldn't happen but guard it

        # FIX 4: check distance covered, not turns required
        # fleet arrives when available_turns * speed >= travel_distance
        available_turns = future_step - current_step
        if available_turns * speed < travel_distance:
            continue

        launch_angle = math.atan2(dy, dx)

        # Sun collision check
        line_dist = abs(dx * (launch_y - sun_y) - dy * (launch_x - sun_x)) / total_distance
        dot = ((sun_x - launch_x) * dx + (sun_y - launch_y) * dy) / (total_distance ** 2)

        if line_dist <= sun_radius and 0 <= dot <= 1:
            continue  # FIX 2: continue not return — orbit may clear the sun later

        return future_step, launch_angle

    return None, None


def evaluate_parasite_attack(target_planet_id, current_step, obs, planet_matrix, fleet_tracker, my_player_id):
    """
    Evaluates if a Parasite Strike is possible against a specific target planet.
    
    Returns:
        dict: Best launch parameters containing:
              {'from_planet_id', 'target_planet_id', 'ships', 'angle', 'impact_step', 'net_gain'}
        None: If no viable parasite window exists.
    """
    # 1. Fetch target planet profile safely
    target_planet = None
    for p in obs.get("planets", []):
        if p[0] == target_planet_id:
            target_planet = p
            break
            
    if not target_planet:
        return None
        
    p_production = target_planet[6]
    p_radius = target_planet[4]
    
    # 2. Get incoming hostile threats sorted chronologically
    threats = fleet_tracker.get_incoming_threats(target_planet_id)
    if not threats:
        return None
        
    # Find the primary structural capture event
    # (Simulate standard garrison subtraction logic to see when/if the node flips)
    current_garrison = target_planet[5]
    current_owner = target_planet[1]
    
    flip_event = None
    
    for threat in threats:
        # Ignore our own support fleets or third-party non-hostiles if logic dictates
        if threat["owner"] == my_player_id:
            continue
            
        t_impact = threat["impact_step"]
        enemy_ships = threat["ships"]
        
        # Calculate natural production growth before impact if target is owned
        growth_turns = t_impact - current_step
        if current_owner != -1: # Base or enemy owned
            projected_garrison = current_garrison + (p_production * growth_turns)
        else: # Neutral nodes do not regenerate base garrisons
            projected_garrison = current_garrison
            
        if enemy_ships > projected_garrison:
            # The enemy successfully captures the node on this step
            surviving_enemy = enemy_ships - projected_garrison
            flip_event = {
                "step": t_impact,
                "surviving_garrison": surviving_enemy,
                "new_owner": threat["owner"]
            }
            break
        else:
            # Enemy fleet breaks against the defenses; update remaining structural mass
            current_garrison = projected_garrison - enemy_ships
            # Reset current step baseline for subsequent simulation loops
            current_step = t_impact 

    if not flip_event:
        return None  # No structural flip detected; standard parasite conditions fail

    # --- WE HAVE A VIABLE FLIP EVENT ---
    # Optimal strike window is exactly 1 frame post-impact
    target_impact_step = flip_event["step"] + 1
    required_transit_turns = target_impact_step - current_step
    
    if required_transit_turns <= 0:
        return None # Too late to intercept
        
    # Projected garrison we must defeat includes 1 turn of natural production growth
    garrison_to_defeat = flip_event["surviving_garrison"] + p_production
    
    # Retrieve target's future coordinates at our precise impact frame
    tx, ty = planet_matrix[target_planet_id][target_impact_step]
    
    best_strike = None
    sun_x, sun_y, sun_radius = 50.0, 50.0, 10.0
    ln_1000 = math.log(1000.0)

    # 3. Iterate over all candidate friendly bases to find the cheapest valid vector
    for p in obs.get("planets", []):
        if p[1] != my_player_id: # Only launch from owned nodes
            continue
            
        from_id = p[0]
        if from_id == target_planet_id:
            continue
            
        lx, ly = p[2], p[3]
        l_radius = p[4]
        l_ships = p[5]
        
        # Calculate spatial distance (border-to-border)
        dx, dy = tx - lx, ty - ly
        center_dist = math.hypot(dx, dy)
        travel_distance = center_dist - l_radius - p_radius
        
        if travel_distance <= 0:
            continue
            
        # Calculate required continuous velocity
        v_req = travel_distance / required_transit_turns
        
        # Speed bounds check (Engine clamps speed between 1.0 and 6.0)
        if v_req < 1.0 or v_req > 6.0:
            continue
            
        # --- Analytic Speed-to-Mass Inversion ---
        # k = exp( ln(1000) * ((v - 1) / 5)^(2/3) )
        if v_req == 1.0:
            k_speed = 1
        else:
            exponent = ((v_req - 1.0) / 5.0) ** (2.0 / 3.0)
            k_speed = math.ceil(math.exp(ln_1000 * exponent))
            
        # Total payload must satisfy BOTH speed requirements and combat requirements
        required_ships = max(k_speed, garrison_to_defeat + 1)
        
        # Verify launch base has sufficient mass reserves
        if required_ships > l_ships:
            continue
            
        # Recalculate actual speed using finalized payload to ensure integer scaling buffers hold
        actual_speed = 1.0 + 5.0 * ((math.log(required_ships) / ln_1000) ** 1.5)
        actual_turns = travel_distance / actual_speed
        
        # Ensure floating-point precision adjustments don't push arrival into the wrong integer bin
        if math.floor(current_step + actual_turns) != (target_impact_step - 1): 
            # Note: Engine collision triggers mid-step traversal; indexing checks match boundary conditions
            continue

        # --- Sun Collision Safety Check ---
        line_dist = abs(dx * (ly - sun_y) - dy * (lx - sun_x)) / center_dist
        dot_product = ((sun_x - lx) * dx + (sun_y - ly) * dy) / (center_dist ** 2)
        
        if line_dist <= sun_radius and 0 <= dot_product <= 1:
            continue # Lethal solar vector; skip base
            
        # Score solution efficiency (Cheapest structural mass investment wins)
        net_gain = p_production * (500 - target_impact_step) - required_ships
        
        if not best_strike or required_ships < best_strike["ships"]:
            best_strike = {
                "from_planet_id": from_id,
                "target_planet_id": target_planet_id,
                "ships": required_ships,
                "angle": math.atan2(dy, dx),
                "impact_step": target_impact_step,
                "net_gain": net_gain
            }

    return best_strike

def dispatch_network_reinforcements(current_step, obs, planet_matrix, fleet_tracker, my_player_id):
    """
    Scans owned planets for projected deficits and dispatches synchronized 
    reinforcements from safe surplus nodes.
    """
    actions = []
    owned_planets = {p[0]: p for p in obs.get("planets", []) if p[1] == my_player_id}
    
    # Keep track of committed donor mass during this turn's loop
    committed_mass = {p_id: 0 for p_id in owned_planets}

    # 1. Evaluate all owned nodes for critical defensive deficits
    for target_id, target_p in owned_planets.items():
        production = target_p[6]
        current_garrison = target_p[5]
        
        threats = fleet_tracker.get_incoming_threats(target_id)
        if not threats:
            continue
            
        # Simulate chronological survival to find the first fatal impact
        simulated_garrison = current_garrison
        last_step = current_step
        fatal_threat = None
        
        for threat in threats:
            # Ignore our own incoming support fleets
            if threat["owner"] == my_player_id:
                simulated_garrison += threat["ships"]
                continue
                
            dt = threat["impact_step"] - last_step
            simulated_garrison += (production * dt)
            simulated_garrison -= threat["ships"]
            last_step = threat["impact_step"]
            
            if simulated_garrison < 0:
                fatal_threat = threat
                break
                
        if not fatal_threat:
            continue # Planet holds naturally; no reinforcements needed

        # --- NODE DEFINITELY FALLS: Calculate absolute deficit ---
        deficit = abs(simulated_garrison)
        target_impact_step = fatal_threat["impact_step"]
        
        # 2. Find the optimal donor planet (y) to cover this deficit
        best_donor_id = None
        best_launch_angle = None
        best_payload = None
        min_mass_cost = float('inf')
        
        for donor_id, donor_p in owned_planets.items():
            if donor_id == target_id:
                continue
                
            # Verify donor is completely safe from incoming vectors
            if fleet_tracker.get_incoming_threats(donor_id):
                continue
                
            available_ships = donor_p[5] - committed_mass[donor_id]
            if available_ships <= deficit:
                continue
                
            # Test if donor can achieve Pre-Arrival Fortification (Window 1)
            # Try sending minimal required mass to see its arrival step
            test_payload = deficit + 1
            arr_step, l_angle = calculate_fleet_arrival(
                donor_id, target_id, test_payload, current_step, obs, planet_matrix
            )
            
            if not arr_step:
                continue
                
            # If standard minimal speed lands too late, test if launching a HEAVIER 
            # fleet (logarithmic acceleration) pushes arrival frame before impact
            if arr_step >= target_impact_step:
                # Analytic inversion check: calculate required speed to land at impact_step - 1
                req_turns = (target_impact_step - 1) - current_step
                if req_turns <= 0:
                    continue
                    
                # Fetch target positions to get exact dynamic distance
                tx, ty = planet_matrix[target_id][target_impact_step - 1]
                lx, ly = donor_p[2], donor_p[3]
                dist = math.hypot(tx - lx, ty - ly) - donor_p[4] - target_p[4]
                
                req_speed = dist / req_turns
                if req_speed > 6.0: 
                    continue # Structurally impossible to reach in time
                    
                # Invert speed formula to find necessary heavy payload
                if req_speed > 1.0:
                    exponent = ((req_speed - 1.0) / 5.0) ** (2.0 / 3.0)
                    heavy_payload = math.ceil(math.exp(math.log(1000.0) * exponent))
                else:
                    heavy_payload = test_payload
                    
                test_payload = max(test_payload, heavy_payload)
                
                # Re-verify donor has sufficient reserves for this heavy acceleration burn
                if test_payload > available_ships:
                    continue
                    
                arr_step, l_angle = calculate_fleet_arrival(
                    donor_id, target_id, test_payload, current_step, obs, planet_matrix
                )
                
            # Final Safety Gate: Ensure we land strictly before the strike
            if arr_step and arr_step < target_impact_step:
                if test_payload < min_mass_cost:
                    min_mass_cost = test_payload
                    best_donor_id = donor_id
                    best_launch_angle = l_angle
                    best_payload = test_payload
                    
        # 3. Commit the optimal action if a valid solution was found
        if best_donor_id is not None:
            actions.append([best_donor_id, best_launch_angle, best_payload])
            committed_mass[best_donor_id] += best_payload
            
    return actions

def _project_garrison(planet, current_step, future_step):
    """
    Projects a planet's garrison at a future step assuming no combat.
    Owned planets grow; neutral planets do not.
    """
    owner = planet[1]
    ships = planet[5]
    production = planet[6]
    dt = future_step - current_step
    if owner != -1:
        return ships + production * dt
    return ships  # neutral planets do not regenerate
 
 
def _opponent_reach_probability(target_id, target_impact_step, current_step,
                                obs, planet_matrix, my_player_id):
    """
    Estimates P(opponent contests target before or at target_impact_step).
 
    Strategy:
      - Find all enemy planets
      - For each, compute whether they can physically reach the target in time
        using maximum fleet speed (6.0)
      - Weight by how many ships they have available (richer enemies = higher threat)
      - Return a probability in [0, 1] via a sigmoid-style normalization
    """
    sun_x, sun_y, sun_radius = 50.0, 50.0, 10.0
 
    if target_id not in planet_matrix:
        return 0.0
 
    trajectory = planet_matrix[target_id]
    if target_impact_step >= len(trajectory):
        return 0.0
 
    tx, ty = trajectory[target_impact_step]
 
    threat_score = 0.0
    total_enemy_ships = 0.0
 
    for p in obs.get("planets", []):
        owner = p[1]
        if owner == my_player_id or owner == -1:
            continue  # only actual opponents
 
        ex, ey, e_radius = p[2], p[3], p[4]
        e_ships = p[5]
 
        dx, dy = tx - ex, ty - ey
        center_dist = math.hypot(dx, dy)
        travel_dist = center_dist - e_radius - 2.0  # rough target radius
        if travel_dist <= 0:
            travel_dist = 0.1
 
        # Can opponent reach at max speed (6.0) in time?
        available_turns = target_impact_step - current_step
        if available_turns <= 0:
            continue
 
        max_coverable = 6.0 * available_turns
        if travel_dist > max_coverable:
            continue  # physically impossible for this enemy
 
        # Sun block check
        if center_dist > 0:
            line_dist = abs(dx * (ey - sun_y) - dy * (ex - sun_x)) / center_dist
            dot = ((sun_x - ex) * dx + (sun_y - ey) * dy) / (center_dist ** 2)
            if line_dist <= sun_radius and 0 <= dot <= 1:
                continue  # sun blocks this enemy's path
 
        # Proximity ratio: closer enemies with more ships = higher threat
        proximity_ratio = 1.0 - (travel_dist / max_coverable)  # 0 to 1
        threat_score += proximity_ratio * e_ships
        total_enemy_ships += e_ships
 
    if total_enemy_ships == 0:
        return 0.0
 
    # Normalize to [0, 1] — not a true probability but a calibrated risk score
    raw = threat_score / (total_enemy_ships + 1e-6)
    # Sigmoid compression so extreme values don't dominate
    probability = 1.0 / (1.0 + math.exp(-10.0 * (raw - 0.3)))
    return min(probability, 0.95)  # cap at 0.95, never certain
 
 
def _compute_delta_v(target_planet, impact_step, ships_sent,
                     current_step, total_my_ships):
    """
    Computes delta_v for a candidate launch per the NPV formula:
 
        delta_v = prod_p * (T_remaining - ETA)   [future gain]
           - k * (T_remaining / total_ships) [opportunity cost]
 
    Returns float. Positive = worth launching.
    """
    prod = target_planet[6]
    t_remaining = 500 - current_step
    eta = impact_step - current_step
 
    future_gain = prod * (t_remaining - eta)
    opportunity_cost = ships_sent * (t_remaining / max(total_my_ships, 1))
 
    return future_gain - opportunity_cost
 
 
def _expected_delta_v(delta_v, p_opponent_contests, garrison_to_beat, ships_sent):
    """
    Discounts raw delta_v by the probability the opponent gets there first.
 
    If opponent contests:
      - If we still win the race (ships_sent > garrison_to_beat after opponent
        adds their fleet), we gain delta_v but at higher cost — approximated
        as a 40% value reduction (contested capture is messy and costly).
      - If opponent gets there first with overwhelming force, expected value = 0.
 
    E[ΔV] = (1 - p_contest) * delta_v
           + p_contest * 0.4 * delta_v  [win contested, reduced value]
           simplified conservatively as:
           = delta_v * (1 - 0.6 * p_contest)
    """
    return delta_v * (1.0 - 0.6 * p_opponent_contests)
 
 
def dispatch_regret_minimized_attacks(current_step, obs, planet_matrix,
                                      fleet_tracker, my_player_id,
                                      already_committed_mass=None):
    """
    Main entry point for the regret minimization attack layer.
 
    Parameters:
        current_step          : int, current game step
        obs                   : dict, raw observation from environment
        planet_matrix         : dict, precomputed trajectories from build_world_state_matrix
        fleet_tracker         : FleetTracker instance, already updated this turn
        my_player_id          : int, your player ID
        already_committed_mass: dict {planet_id: ships_committed} from earlier passes
                                (parasite + reinforcement). Prevents double-spending.
 
    Returns:
        list of [from_planet_id, angle, num_ships] actions
    """
    actions = []
 
    all_planets = obs.get("planets", [])
    owned = [p for p in all_planets if p[1] == my_player_id]
 
    if not owned:
        return actions
 
    # ── 1. Compute total ship economy for opportunity cost term ───────────────
    total_my_ships = sum(p[5] for p in owned)
    for f in fleet_tracker.active_fleets.values():
        if f["owner"] == my_player_id:
            total_my_ships += f["ships"]
 
    # ── 2. Select top 50% of owned planets by current garrison ───────────────
    owned_sorted = sorted(owned, key=lambda p: p[5], reverse=True)
    top_half_count = max(1, math.ceil(len(owned_sorted) / 2))
    launchers = owned_sorted[:top_half_count]
 
    # ── 3. Build committed mass tracker (merge with earlier passes) ───────────
    committed = {p[0]: 0 for p in owned}
    if already_committed_mass:
        for pid, mass in already_committed_mass.items():
            if pid in committed:
                committed[pid] = mass
 
    # ── 4. Identify candidate targets: neutral + enemy planets ───────────────
    targets = [p for p in all_planets if p[1] != my_player_id]
 
    if not targets:
        return actions
 
    # ── 5. For each launcher, score all targets and pick best expected deltaV ─────
    # Collect (launcher, target, action_dict) scored globally,
    # then greedily assign to avoid double-committing mass.
 
    scored_actions = []
 
    for launcher in launchers:
        l_id = launcher[0]
        l_ships = launcher[5]
        l_radius = launcher[4]
        available = l_ships - committed[l_id]
 
        if available <= 1:
            continue  # nothing meaningful to send
 
        for target in targets:
            t_id = target[0]
            t_owner = target[1]
 
            # Skip targets we already have a fleet heading toward
            already_targeting = any(
                f["target_planet_id"] == t_id and f["owner"] == my_player_id
                for f in fleet_tracker.active_fleets.values()
            )
            if already_targeting:
                continue
 
            # Get arrival step and angle
            impact_step, launch_angle = calculate_fleet_arrival(
                l_id, t_id, available, current_step, obs, planet_matrix
            )
            if impact_step is None:
                continue
 
            # Project garrison at arrival
            projected_garrison = _project_garrison(target, current_step, impact_step)
 
            # Minimum ships needed to capture
            ships_to_capture = projected_garrison + 1
 
            if ships_to_capture > available:
                continue  # can't afford it
 
            # Use exactly enough to capture (preserve reserves)
            ships_sent = ships_to_capture
 
            # Recompute arrival for exact payload (speed changes with fleet size)
            if ships_sent != available:
                impact_step, launch_angle = calculate_fleet_arrival(
                    l_id, t_id, ships_sent, current_step, obs, planet_matrix
                )
                if impact_step is None:
                    continue
 
            # Raw ΔV
            dv = _compute_delta_v(target, impact_step, ships_sent,
                                   current_step, total_my_ships)
 
            if dv <= 0:
                continue  # not worth it even without opposition
 
            # Opponent contest probability
            p_contest = _opponent_reach_probability(
                t_id, impact_step, current_step, obs, planet_matrix, my_player_id
            )
 
            # Expected ΔV after discounting for opponent interference
            e_dv = _expected_delta_v(dv, p_contest, projected_garrison, ships_sent)
 
            if e_dv <= 0:
                continue  # opponent makes this launch not worth it
 
            scored_actions.append({
                "launcher_id": l_id,
                "target_id": t_id,
                "ships": ships_sent,
                "angle": launch_angle,
                "impact_step": impact_step,
                "e_dv": e_dv,
                "p_contest": p_contest,
                "raw_dv": dv
            })
 
    # ── 6. Greedy assignment: highest expected deltaV first, no double-spending ───
    scored_actions.sort(key=lambda x: x["e_dv"], reverse=True)
 
    used_targets = set()
 
    for act in scored_actions:
        l_id = act["launcher_id"]
        t_id = act["target_id"]
 
        # Don't send two fleets to the same target this turn
        if t_id in used_targets:
            continue
 
        # Re-check available ships (may have been committed by a higher-ranked action)
        launcher_planet = next((p for p in owned if p[0] == l_id), None)
        if not launcher_planet:
            continue
 
        available_now = launcher_planet[5] - committed[l_id]
        if act["ships"] > available_now:
            continue
 
        # Commit and record
        actions.append([l_id, act["angle"], act["ships"]])
        committed[l_id] += act["ships"]
        used_targets.add(t_id)
 
    return actions
 
class OutgoingLedger:
    def __init__(self):
        # O(1) Active Registry: mapping synthetic_id -> OutgoingProfile dict
        self.active_dispatches = {}

    def register_launch(self, source_id, target_id, ships, current_step, arrival_step):
        """
        Logs a newly committed friendly launch vector into the active matrix.
        """
        # Generate a unique synthetic key for this specific dispatch event
        # (Appending ship count resolves edge cases where multiple distinct logic blocks 
        # launch independent micro-fleets along identical vectors in the same turn)
        synthetic_id = f"{source_id}_{target_id}_{current_step}_{ships}"
        
        self.active_dispatches[synthetic_id] = {
            "source_id": source_id,
            "target_id": target_id,
            "ships": ships,
            "launch_step": current_step,
            "arrival_step": arrival_step
        }

    def update(self, current_step):
        """
        Prunes expired dispatches natively based on deterministic arrival steps.
        Must be called at the absolute start of every turn loop.
        """
        # Find all keys where the current simulation step meets or exceeds the projected impact frame
        expired_keys = [
            k for k, dispatch in self.active_dispatches.items() 
            if current_step >= dispatch["arrival_step"]
        ]
        
        # Purge impacts from memory in O(1) time
        for k in expired_keys:
            del self.active_dispatches[k]

    def get_friendly_reinforcements(self, target_planet_id):
        """
        Retrieves all airborne friendly fleets projected to land on a specific node.
        Returns a list sorted chronologically by arrival step.
        """
        reinforcements = [
            d for d in self.active_dispatches.values() 
            if d["target_id"] == target_planet_id
        ]
        return sorted(reinforcements, key=lambda x: x["arrival_step"])
    
import math

def launch_neutral_expansion(self, source_planet_id, obs):
    """
    Scans all neutral planets to find the fastest valid geometric interception.
    Calculates exact deterministic speed from the required capture payload
    and cross-references the pre-computed matrix to guarantee a hit.
    
    Returns:
        list: [source_planet_id, launch_angle_radians, required_payload] or None
    """
    current_step = obs.get("step", 0)
    
    # 1. Retrieve source base parameters safely
    source_p = None
    for p in obs.get("planets", []):
        if p[0] == source_planet_id:
            source_p = p
            break
            
    if not source_p or source_p[5] <= 1:
        return None
        
    sx, sy = source_p[2], source_p[3]
    s_radius = source_p[4]
    available_ships = source_p[5]
    
    best_target_id = None
    best_launch_angle = None
    best_payload = None
    min_arrival_step = float('inf')
    
    sun_x, sun_y, sun_radius = 50.0, 50.0, 10.0
    ln_1000 = math.log(1000.0)
    
    # 2. Evaluate candidate neutral targets
    for target_p in obs.get("planets", []):
        t_id = target_p[0]
        owner = target_p[1]
        
        # Filter strictly for unowned neutral space
        if owner != -1:
            continue
            
        target_garrison = target_p[5]
        t_radius = target_p[4]
        
        # Absolute minimal payload to trigger an ownership flip
        payload = target_garrison + 1
        
        # Verify source base has adequate physical reserves
        if payload > available_ships:
            continue
            
        # Optional Check: Skip if our FleetTracker flags an incoming hostile vector
        # targeting this neutral node to avoid walking into a multi-agent trap.
        if hasattr(self, 'fleet_tracker') and self.fleet_tracker.get_incoming_threats(t_id):
            continue
            
        # Derive exact deterministic scalar speed for this specific mass
        speed = 1.0 + 5.0 * ((math.log(payload) / ln_1000) ** 1.5)
        
        # Access target's pre-computed 500-step position matrix
        trajectory = self.planet_matrix.get(t_id)
        if not trajectory:
            continue
            
        # 3. Chronological Scan: Find the first continuous collision frame
        max_search_limit = min(501, len(trajectory))
        for future_step in range(current_step + 1, max_search_limit):
            # If this candidate arrival frame is already slower than a verified hit, abort loop
            if future_step >= min_arrival_step:
                break
                
            tx, ty = trajectory[future_step]
            
            # Calculate center-to-center distance vector at the target frame
            dx, dy = tx - sx, ty - sy
            c2c_dist = math.hypot(dx, dy)
            
            # Subtract source spawn offset and target outer boundary collision trigger
            travel_dist = c2c_dist - s_radius - t_radius
            
            # Protect against immediate proximity bounds triggers
            if travel_dist <= 0:
                min_arrival_step = future_step
                best_target_id = t_id
                best_launch_angle = math.atan2(dy, dx)
                best_payload = payload
                break
                
            required_turns = travel_dist / speed
            available_turns = future_step - current_step
            
            # Continuous collision logic validates if the required transit clears during this step
            if available_turns >= required_turns:
                launch_angle = math.atan2(dy, dx)
                
                # --- Sun Collision Safety Check ---
                # Calculate perpendicular distance from Sun core to the trajectory line segment
                line_dist = abs(dx * (sy - sun_y) - dy * (sx - sun_x)) / c2c_dist
                dot_product = ((sun_x - sx) * dx + (sun_y - sy) * dy) / (c2c_dist ** 2)
                
                # Only prune if the trajectory segment passes directly through the solar radius
                if line_dist <= sun_radius and 0 <= dot_product <= 1:
                    continue 
                    
                # Optimal interception state confirmed
                min_arrival_step = future_step
                best_target_id = t_id
                best_launch_angle = launch_angle
                best_payload = payload
                break

    if best_target_id is not None:
        return [source_planet_id, best_launch_angle, best_payload]
        
    return None

import math

class NetworkTopologyCache:
    def __init__(self, obs):
        self.sun_x, self.sun_y = 50.0, 50.0
        self.omega = obs.get("angular_velocity", 0.0)
        self.period = math.ceil(2 * math.pi / self.omega) if self.omega > 0 else 500
        
        # O(1) Lookup Tables
        # static_distances[id_A][id_B] -> float
        self.static_distances = {}
        # dynamic_distances[orb_id][stat_id] -> list of floats (length == period)
        self.dynamic_distances = {}
        
        self._build_topology(obs.get("initial_planets", []))

    def _build_topology(self, initial_planets):
        # Parse profiles and separate by dynamic classification
        static_nodes = {}
        orbiting_nodes = {}
        
        for p in initial_planets:
            p_id, _, x0, y0, radius = p[:5]
            dx, dy = x0 - self.sun_x, y0 - self.sun_y
            dist_sun = math.hypot(dx, dy)
            
            if dist_sun + radius >= 50.0:
                static_nodes[p_id] = (x0, y0)
                self.static_distances[p_id] = {}
            else:
                orbiting_nodes[p_id] = {
                    "R": dist_sun,
                    "theta": math.atan2(dy, dx)
                }
                self.static_distances[p_id] = {}
                self.dynamic_distances[p_id] = {}

        # --- REGIME A: Static to Static ---
        for id_a, pos_a in static_nodes.items():
            for id_b, pos_b in static_nodes.items():
                d = math.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1])
                self.static_distances[id_a][id_b] = d

        # --- REGIME B: Orbiting to Orbiting ---
        for id_a, polar_a in orbiting_nodes.items():
            for id_b, polar_b in orbiting_nodes.items():
                if id_a == id_b:
                    self.static_distances[id_a][id_b] = 0.0
                    continue
                # Law of Cosines (Time independent!)
                r_a, r_b = polar_a["R"], polar_b["R"]
                delta_theta = polar_a["theta"] - polar_b["theta"]
                d = math.sqrt(r_a**2 + r_b**2 - 2 * r_a * r_b * math.cos(delta_theta))
                self.static_distances[id_a][id_b] = d

        # --- REGIME C: Orbiting to Static (Periodic Pre-computation) ---
        for orb_id, polar_orb in orbiting_nodes.items():
            r_orb, theta_orb_0 = polar_orb["R"], polar_orb["theta"]
            
            for stat_id, pos_stat in static_nodes.items():
                dx_stat, dy_stat = pos_stat[0] - self.sun_x, pos_stat[1] - self.sun_y
                r_stat = math.hypot(dx_stat, dy_stat)
                theta_stat = math.atan2(dy_stat, dx_stat)
                
                # Pre-compute exactly one rotational cycle
                periodic_curve = [0.0] * self.period
                for t in range(self.period):
                    theta_orb_t = theta_orb_0 + (self.omega * t)
                    delta_theta = theta_stat - theta_orb_t
                    periodic_curve[t] = math.sqrt(
                        r_orb**2 + r_stat**2 - 2 * r_orb * r_stat * math.cos(delta_theta)
                    )
                self.dynamic_distances[orb_id][stat_id] = periodic_curve

    def get_distance(self, id_a, id_b, step):
        """Instantaneous network distance resolution."""
        # Check static tables first (Regimes A & B)
        if id_b in self.static_distances.get(id_a, {}):
            return self.static_distances[id_a][id_b]
            
        # Check periodic dynamic curves (Regime C)
        if id_a in self.dynamic_distances and id_b in self.dynamic_distances[id_a]:
            return self.dynamic_distances[id_a][id_b][step % self.period]
        elif id_b in self.dynamic_distances and id_a in self.dynamic_distances[id_b]:
            return self.dynamic_distances[id_b][id_a][step % self.period]
            
        return 999.0 # Fallback safety for unmapped extra-solar entities (Comets)


# =====================================================================
# --- VALUATION LAYER EVALUATOR ---
# =====================================================================
def compute_planet_defensibility(target_p, current_step, obs, topology_cache):
    """
    Computes an integrated Defensibility Score for an enemy planet based on 
    native garrison size and the spatial density of supporting enemy clusters.
    """
    target_id = target_p[0]
    target_owner = target_p[1]
    base_garrison = target_p[5]
    
    # Critical support decay variables
    d_critical = 35.0
    gamma = 2.0
    
    network_support = 0.0
    
    # Scan all active nodes to identify supporting network capacity
    for p in obs.get("planets", []):
        p_id = p[0]
        owner = p[1]
        
        # Only evaluate supporting nodes sharing identical hostile ownership
        if owner != target_owner or p_id == target_id:
            continue
            
        reserves = p[5]
        if reserves <= 1:
            continue
            
        # Fetch pre-computed O(1) distance at the evaluation step
        dist = topology_cache.get_distance(target_id, p_id, current_step)
        
        # Apply spatial decay curve
        if dist < d_critical:
            support_factor = 1.0 - ((dist / d_critical) ** gamma)
            network_support += reserves * support_factor
            
    # Final unified evaluation score
    return base_garrison + network_support


# ─────────────────────────────────────────────────────────────────────────────
# MASTER AGENT ORCHESTRATOR
# Integrates all three passes in priority order:
#   1. Reinforcement (defensive, highest priority)
#   2. Parasite strikes (opportunistic captures on flip events)
#   3. Regret-minimized attacks (offensive expansion)
# ─────────────────────────────────────────────────────────────────────────────

def agent(obs):
    global _planet_matrix, _fleet_tracker
    player_id = obs.get("player", 0)
    current_step = obs.get("step", 0)

    if _planet_matrix is None:
        _planet_matrix = build_world_state_matrix(obs)

    if _fleet_tracker is None:
        _fleet_tracker = FleetTracker()
    _fleet_tracker.update(current_step, obs.get("fleets", []), _planet_matrix)

    all_actions = []
    committed_mass = {}

    reinforcement_actions = dispatch_network_reinforcements(
        current_step, obs, _planet_matrix, _fleet_tracker, player_id
    )
    all_actions.extend(reinforcement_actions)

    for action in reinforcement_actions:
        pid = action[0]
        committed_mass[pid] = committed_mass.get(pid, 0) + action[2]

    all_planets = obs.get("planets", [])
    enemy_and_neutral = [p for p in all_planets if p[1] != player_id]

    for target in enemy_and_neutral:
        strike = evaluate_parasite_attack(
            target[0], current_step, obs, _planet_matrix, _fleet_tracker, player_id
        )
        if strike and strike.get("net_gain", 0) > 0:
            from_id = strike["from_planet_id"]
            already = committed_mass.get(from_id, 0)
            launcher = next((p for p in all_planets if p[0] == from_id), None)
            if launcher and (already + strike["ships"]) <= launcher[5]:
                all_actions.append([from_id, strike["angle"], strike["ships"]])
                committed_mass[from_id] = already + strike["ships"]

    expansion_actions = dispatch_regret_minimized_attacks(
        current_step, obs, _planet_matrix, _fleet_tracker, player_id,
        already_committed_mass=committed_mass
    )
    all_actions.extend(expansion_actions)


    return all_actions