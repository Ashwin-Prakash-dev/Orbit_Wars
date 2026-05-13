import math

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────
_planet_matrix   = None
_fleet_tracker   = None
_planet_radii    = None
_outgoing_ledger = None
_topology_cache  = None   # FIX 5: wired in for defensibility-based attack scoring
_game_id         = None   # episode-reset detection


# ─────────────────────────────────────────────────────────────────────────────
# WORLD STATE MATRIX
# ─────────────────────────────────────────────────────────────────────────────
def build_world_state_matrix(obs):
    """
    Builds a complete lookup table of all planetary coordinates across 500 steps.
    Returns: dict mapping planet_id -> list of (x, y) tuples for steps 0..500.
    angular_velocity is a single global constant for all orbiting planets.
    """
    initial_planets = obs.get("initial_planets", [])
    omega           = obs.get("angular_velocity", 0.0)
    max_steps       = 500
    sun_x, sun_y    = 50.0, 50.0

    matrix = {}
    for p in initial_planets:
        p_id     = p[0]
        x0, y0   = p[2], p[3]
        p_radius = p[4]

        dx, dy         = x0 - sun_x, y0 - sun_y
        orbital_radius = math.hypot(dx, dy)

        if orbital_radius + p_radius >= 50.0:
            matrix[p_id] = [(x0, y0)] * (max_steps + 1)
            continue

        theta_0    = math.atan2(dy, dx)
        trajectory = [None] * (max_steps + 1)
        for step in range(max_steps + 1):
            theta = theta_0 + omega * step
            trajectory[step] = (
                sun_x + orbital_radius * math.cos(theta),
                sun_y + orbital_radius * math.sin(theta),
            )
        matrix[p_id] = trajectory

    return matrix


def build_planet_radii(obs):
    """Returns {planet_id: radius} from initial_planets."""
    return {p[0]: p[4] for p in obs.get("initial_planets", [])}


# ─────────────────────────────────────────────────────────────────────────────
# OUTGOING LEDGER
# Tracks all fleets we have launched (this turn and prior turns still airborne)
# so that every dispatcher can avoid double-spending ships from the same planet
# and avoid redundantly targeting the same destination.
# ─────────────────────────────────────────────────────────────────────────────
class OutgoingLedger:
    def __init__(self):
        self.active_dispatches = {}   # synthetic_id -> dispatch profile

    def register_launch(self, source_id, target_id, ships, current_step, arrival_step):
        """Log a newly committed friendly launch into the ledger."""
        synthetic_id = f"{source_id}_{target_id}_{current_step}_{ships}"
        self.active_dispatches[synthetic_id] = {
            "source_id":    source_id,
            "target_id":    target_id,
            "ships":        ships,
            "launch_step":  current_step,
            "arrival_step": arrival_step,
        }

    def prune(self, current_step):
        """
        Remove dispatches whose arrival step has already passed.
        Call once at the start of every turn, BEFORE any dispatcher runs.

        FIX 10: was `>= arrival_step`, changed to `> arrival_step` so fleets
        are still counted as inbound on their exact arrival step, giving a full
        turn of correct inbound accounting before combat resolves.
        """
        expired = [
            k for k, d in self.active_dispatches.items()
            if current_step > d["arrival_step"]
        ]
        for k in expired:
            del self.active_dispatches[k]

    def ships_committed_from(self, source_id):
        """
        Total ships from source_id that are currently airborne.
        Deducted from available garrison so we never over-draw across turns.
        """
        return sum(
            d["ships"] for d in self.active_dispatches.values()
            if d["source_id"] == source_id
        )

    def inbound_friendly_ships(self, target_id):
        """
        Total friendly ships already registered en route to target_id.
        Used to decide whether a target still needs more coverage.
        """
        return sum(
            d["ships"] for d in self.active_dispatches.values()
            if d["target_id"] == target_id
        )

    def get_friendly_reinforcements(self, target_id):
        """Sorted list of all airborne friendly dispatches heading to target_id."""
        return sorted(
            [d for d in self.active_dispatches.values() if d["target_id"] == target_id],
            key=lambda x: x["arrival_step"]
        )


# ─────────────────────────────────────────────────────────────────────────────
# FLEET TRACKER  (enemy + all airborne fleet intelligence)
# ─────────────────────────────────────────────────────────────────────────────
class FleetTracker:
    def __init__(self, planet_radii, sun_x=50.0, sun_y=50.0):
        self.sun_x         = sun_x
        self.sun_y         = sun_y
        self.planet_radii  = planet_radii
        self.active_fleets = {}   # fleet_id -> profile dict

    def calculate_speed(self, ships):
        if ships <= 1:
            return 1.0
        return 1.0 + 5.0 * (math.log(ships) / math.log(1000.0)) ** 1.5

    def update(self, current_step, fleets_obs, global_planet_matrix,
               my_player_id=None, outgoing_ledger=None):
        """
        Sync registry with the current environment observation.

        FIX 7: accepts my_player_id + outgoing_ledger.  For our own fleets
        we look up the target from the ledger by (source_planet, ships), which
        skips the expensive O(steps × planets) _resolve_target ray-march
        entirely for already-registered friendly launches.
        The `from_planet_id` field (raw_fleet[5]) was previously ignored.
        """
        current_visible_ids = set()

        for raw_fleet in fleets_obs:
            f_id           = raw_fleet[0]
            owner          = raw_fleet[1]
            fx             = raw_fleet[2]
            fy             = raw_fleet[3]
            angle          = raw_fleet[4]
            from_planet_id = raw_fleet[5]   # FIX 7: was silently ignored
            ships          = raw_fleet[6]

            current_visible_ids.add(f_id)
            if f_id in self.active_fleets:
                continue

            speed = self.calculate_speed(ships)
            vx    = speed * math.cos(angle)
            vy    = speed * math.sin(angle)

            # FIX 7: for our own fleets, resolve target from the ledger first
            # to avoid the full ray-march.
            target_planet_id = None
            impact_step      = None
            if owner == my_player_id and outgoing_ledger is not None:
                for dispatch in outgoing_ledger.active_dispatches.values():
                    if (dispatch["source_id"] == from_planet_id
                            and dispatch["ships"] == ships):
                        target_planet_id = dispatch["target_id"]
                        impact_step      = dispatch["arrival_step"]
                        break

            if target_planet_id is None:
                target_planet_id, impact_step = self._resolve_target(
                    current_step, fx, fy, vx, vy, speed, global_planet_matrix
                )

            self.active_fleets[f_id] = {
                "owner":            owner,
                "start_x":          fx,
                "start_y":          fy,
                "vx":               vx,
                "vy":               vy,
                "speed":            speed,
                "ships":            ships,
                "spawn_step":       current_step,
                "target_planet_id": target_planet_id,
                "impact_step":      impact_step,
            }

        dead_ids = self.active_fleets.keys() - current_visible_ids
        for d_id in list(dead_ids):
            del self.active_fleets[d_id]

    def _resolve_target(self, start_step, fx, fy, vx, vy, speed, planet_matrix):
        """
        Projects the fleet ray forward step-by-step with a segment-circle
        secondary check so fast fleets cannot tunnel through small planets.
        Uses real planet radii from self.planet_radii.
        """
        max_lookahead = 500 - start_step

        for dt in range(1, max_lookahead + 1):
            eval_step = start_step + dt
            ray_x     = fx + vx * dt
            ray_y     = fy + vy * dt

            if not (0 <= ray_x <= 100 and 0 <= ray_y <= 100):
                return None, None
            if math.hypot(ray_x - self.sun_x, ray_y - self.sun_y) <= 10.0:
                return None, None

            prev_x     = fx + vx * (dt - 1)
            prev_y     = fy + vy * (dt - 1)
            seg_dx     = ray_x - prev_x
            seg_dy     = ray_y - prev_y
            seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy

            for p_id, trajectory in planet_matrix.items():
                if eval_step >= len(trajectory):
                    continue
                px, py   = trajectory[eval_step]
                p_radius = self.planet_radii.get(p_id, 1.5)

                if math.hypot(ray_x - px, ray_y - py) <= p_radius:
                    return p_id, eval_step

                if seg_len_sq > 0:
                    t_proj    = max(0.0, min(1.0, (
                        (px - prev_x) * seg_dx + (py - prev_y) * seg_dy
                    ) / seg_len_sq))
                    closest_x = prev_x + t_proj * seg_dx
                    closest_y = prev_y + t_proj * seg_dy
                    if math.hypot(closest_x - px, closest_y - py) <= p_radius:
                        return p_id, eval_step

        return None, None

    def get_incoming_threats(self, planet_id):
        """All fleets projected to strike planet_id, sorted by ETA."""
        return sorted(
            [f for f in self.active_fleets.values()
             if f["target_planet_id"] == planet_id],
            key=lambda x: x["impact_step"] if x["impact_step"] is not None else float("inf")
        )


# ─────────────────────────────────────────────────────────────────────────────
# FLEET ARRIVAL GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────
def calculate_fleet_arrival(launch_planet_id, target_planet_id, fleet_size,
                             current_step, obs, planet_matrix, planet_radii):
    """
    Returns (arrival_step, launch_angle) for the earliest valid intercept,
    or (None, None). Uses border-to-border distance and real planet radii.
    """
    launch_x = launch_y = launch_radius = None
    target_radius = 1.5

    for p in obs.get("planets", []):
        if p[0] == launch_planet_id:
            launch_x, launch_y, launch_radius = p[2], p[3], p[4]
        if p[0] == target_planet_id:
            target_radius = p[4]

    if launch_x is None:
        return None, None
    if launch_radius is None:
        launch_radius = planet_radii.get(launch_planet_id, 1.5)
    if fleet_size <= 0:
        return None, None

    speed             = 1.0 + 5.0 * (math.log(max(fleet_size, 1)) / math.log(1000.0)) ** 1.5
    target_trajectory = planet_matrix.get(target_planet_id)
    if not target_trajectory:
        return None, None

    sun_x, sun_y, sun_radius = 50.0, 50.0, 10.0

    for future_step in range(current_step + 1, min(len(target_trajectory), 501)):
        tx, ty         = target_trajectory[future_step]
        dx, dy         = tx - launch_x, ty - launch_y
        total_distance = math.hypot(dx, dy)
        if total_distance == 0:
            continue

        travel_distance = total_distance - launch_radius - target_radius
        if travel_distance <= 0:
            return future_step, math.atan2(dy, dx)

        if (future_step - current_step) * speed < travel_distance:
            continue

        launch_angle = math.atan2(dy, dx)
        line_dist    = abs(dx * (launch_y - sun_y) - dy * (launch_x - sun_x)) / total_distance
        if line_dist <= sun_radius:
            dot = ((sun_x - launch_x) * dx + (sun_y - launch_y) * dy) / (total_distance ** 2)
            if 0 <= dot <= 1:
                continue

        return future_step, launch_angle

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# GARRISON PROJECTION
# ─────────────────────────────────────────────────────────────────────────────
def _project_garrison(planet, current_step, future_step,
                       fleet_tracker=None, my_player_id=None):
    """
    Projects the planet's garrison at future_step, accounting for production.

    FIX 8: optionally subtracts third-party hostile fleets (fleets from neither
    us nor the garrison owner) that arrive before future_step.  This gives a
    more accurate capture cost for 4-player scenarios where a rival is already
    weakening an enemy planet for us to grab.  In a 2-player game the filter
    simply has no effect since only our fleets and the enemy's own fleets can
    be in transit.
    """
    owner, ships, production = planet[1], planet[5], planet[6]
    dt   = future_step - current_step
    base = ships + production * dt if owner != -1 else ships

    if fleet_tracker is not None and my_player_id is not None:
        for f in fleet_tracker.get_incoming_threats(planet[0]):
            if (f["owner"] != my_player_id
                    and f["owner"] != owner          # not garrison owner's own reinforcements
                    and f["impact_step"] is not None
                    and f["impact_step"] <= future_step):
                base = max(0, base - f["ships"])

    return base


# ─────────────────────────────────────────────────────────────────────────────
# NEUTRAL EXPANSION
# Dedicated fast-capture pass for neutral planets.
# Uses exact speed-inversion to send the minimum viable payload.
# Respects OutgoingLedger for across-turn airborne budget accounting.
# ─────────────────────────────────────────────────────────────────────────────
def dispatch_neutral_expansion(current_step, obs, planet_matrix, planet_radii,
                                fleet_tracker, outgoing_ledger, my_player_id,
                                already_committed_mass=None):
    """
    For each uncovered neutral planet, find the owned planet that can deliver
    the minimum viable capture payload with the earliest arrival, and queue it.

    Returns (actions, committed_delta) where committed_delta is
    {planet_id: ships_committed} for the orchestrator to merge.
    """
    actions          = []
    committed_delta  = {}
    comet_ids        = set(obs.get("comet_planet_ids", []))
    sun_x, sun_y, sun_radius = 50.0, 50.0, 10.0
    ln_1000          = math.log(1000.0)

    # Effective available ships per owned planet:
    # garrison - previously airborne (ledger) - committed this turn (earlier passes)
    owned_available = {}
    for p in obs.get("planets", []):
        if p[1] != my_player_id:
            continue
        airborne  = outgoing_ledger.ships_committed_from(p[0])
        this_turn = (already_committed_mass or {}).get(p[0], 0)
        owned_available[p[0]] = p[5] - airborne - this_turn

    for target_p in obs.get("planets", []):
        t_id    = target_p[0]
        t_owner = target_p[1]

        if t_owner != -1:
            continue
        if t_id in comet_ids:
            continue   # comets are too transient for reliable expansion

        # Skip if already-registered inbound friendly ships are enough
        inbound  = outgoing_ledger.inbound_friendly_ships(t_id)
        garrison = target_p[5]   # neutral: no production growth
        if inbound > garrison:
            continue

        t_radius       = target_p[4]
        payload_needed = garrison - inbound + 1   # only cover the remaining gap

        best_source_id = None
        best_angle     = None
        best_arrival   = float("inf")
        best_payload   = None

        for src_p in obs.get("planets", []):
            if src_p[1] != my_player_id:
                continue
            s_id      = src_p[0]
            s_avail   = owned_available.get(s_id, 0)
            if s_avail <= 0:
                continue

            actual_payload = min(payload_needed, s_avail)
            if actual_payload <= 0:
                continue

            speed      = 1.0 + 5.0 * ((math.log(max(actual_payload, 1)) / ln_1000) ** 1.5)
            trajectory = planet_matrix.get(t_id)
            if not trajectory:
                continue

            sx, sy   = src_p[2], src_p[3]
            s_radius = src_p[4]

            for future_step in range(current_step + 1, min(len(trajectory), 501)):
                if future_step >= best_arrival:
                    break   # can't improve on existing best from this source

                tx, ty      = trajectory[future_step]
                dx, dy      = tx - sx, ty - sy
                c2c         = math.hypot(dx, dy)
                travel_dist = max(c2c - s_radius - t_radius, 0.0)
                avail_turns = future_step - current_step

                if avail_turns * speed < travel_dist:
                    continue

                if c2c > 0:
                    ld  = abs(dx * (sy - sun_y) - dy * (sx - sun_x)) / c2c
                    dot = ((sun_x - sx) * dx + (sun_y - sy) * dy) / (c2c ** 2)
                    if ld <= sun_radius and 0 <= dot <= 1:
                        continue

                best_source_id = s_id
                best_angle     = math.atan2(dy, dx)
                best_arrival   = future_step
                best_payload   = actual_payload
                break

        if best_source_id is not None:
            actions.append([best_source_id, best_angle, best_payload])
            committed_delta[best_source_id] = (
                committed_delta.get(best_source_id, 0) + best_payload
            )
            # Reduce available so a second neutral target can't double-draw
            owned_available[best_source_id] = (
                owned_available.get(best_source_id, 0) - best_payload
            )
            outgoing_ledger.register_launch(
                best_source_id, t_id, best_payload, current_step, best_arrival
            )

    return actions, committed_delta


# ─────────────────────────────────────────────────────────────────────────────
# PARASITE ATTACK EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_parasite_attack(target_planet_id, current_step, obs, planet_matrix,
                              fleet_tracker, my_player_id, planet_radii):
    """
    Arrives 1 step after an enemy capture event when their garrison is weakest.
    sim_step / sim_garrison are local cursors — current_step is never mutated.
    """
    target_planet = None
    for p in obs.get("planets", []):
        if p[0] == target_planet_id:
            target_planet = p
            break
    if not target_planet:
        return None

    p_production = target_planet[6]
    p_radius     = target_planet[4]

    threats = fleet_tracker.get_incoming_threats(target_planet_id)
    if not threats:
        return None

    sim_garrison = target_planet[5]
    sim_owner    = target_planet[1]
    sim_step     = current_step
    flip_event   = None

    for threat in threats:
        if threat["owner"] == my_player_id:
            continue

        t_impact    = threat["impact_step"]
        enemy_ships = threat["ships"]

        growth_turns = t_impact - sim_step
        if sim_owner != -1:
            sim_garrison += p_production * growth_turns

        if enemy_ships > sim_garrison:
            flip_event = {
                "step":               t_impact,
                "surviving_garrison": enemy_ships - sim_garrison,
                "new_owner":          threat["owner"],
            }
            break
        else:
            sim_garrison -= enemy_ships
            sim_step      = t_impact

    if not flip_event:
        return None

    target_impact_step     = flip_event["step"] + 1
    required_transit_turns = target_impact_step - current_step

    if required_transit_turns <= 0:
        return None

    garrison_to_defeat = flip_event["surviving_garrison"] + p_production

    traj = planet_matrix.get(target_planet_id, [])
    if target_impact_step >= len(traj):
        return None

    best_strike = None
    sun_x, sun_y, sun_radius = 50.0, 50.0, 10.0
    ln_1000 = math.log(1000.0)

    for p in obs.get("planets", []):
        if p[1] != my_player_id:
            continue
        from_id  = p[0]
        if from_id == target_planet_id:
            continue

        lx, ly   = p[2], p[3]
        l_radius = p[4]
        l_ships  = p[5]

        tx, ty          = traj[target_impact_step]
        dx, dy          = tx - lx, ty - ly
        center_dist     = math.hypot(dx, dy)
        travel_distance = max(center_dist - l_radius - p_radius, 0.0)

        v_req = travel_distance / required_transit_turns if required_transit_turns > 0 else 999
        if v_req > 6.0:
            continue
        v_req = max(v_req, 1.0)

        if v_req <= 1.0:
            k_speed = 1
        else:
            exponent = ((v_req - 1.0) / 5.0) ** (2.0 / 3.0)
            k_speed  = math.ceil(math.exp(ln_1000 * exponent))

        required_ships = max(k_speed, garrison_to_defeat + 1)
        if required_ships > l_ships:
            continue

        # FIX 4: use calculate_fleet_arrival for integer-step arrival check
        # instead of float arithmetic with an ad-hoc asymmetric tolerance window.
        # The window allows ±1 step to absorb minor speed-rounding differences.
        arr_step, launch_angle = calculate_fleet_arrival(
            from_id, target_planet_id, required_ships, current_step,
            obs, planet_matrix, planet_radii
        )
        if arr_step is None or not (target_impact_step - 1 <= arr_step <= target_impact_step + 1):
            continue

        if center_dist > 0:
            ld  = abs(dx * (ly - sun_y) - dy * (lx - sun_x)) / center_dist
            dot = ((sun_x - lx) * dx + (sun_y - ly) * dy) / (center_dist ** 2)
            if ld <= sun_radius and 0 <= dot <= 1:
                continue

        net_gain = net_gain = p_production * (500 - target_impact_step) * 2 - required_ships

        if best_strike is None or required_ships < best_strike["ships"]:
            best_strike = {
                "from_planet_id":   from_id,
                "target_planet_id": target_planet_id,
                "ships":            required_ships,
                "angle":            launch_angle,   # FIX 4: from calculate_fleet_arrival, not raw atan2
                "impact_step":      target_impact_step,
                "net_gain":         net_gain,
            }

    return best_strike


# ─────────────────────────────────────────────────────────────────────────────
# REINFORCEMENT DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────
def dispatch_network_reinforcements(current_step, obs, planet_matrix,
                                    fleet_tracker, outgoing_ledger,
                                    my_player_id, planet_radii):
    """
    Defends owned planets projected to fall. Donor availability accounts for
    ships already airborne (outgoing_ledger) so we never over-draw across turns.
    Already-registered friendly inbound ships are included in the survival sim.
    Returns (actions, committed_mass) for the orchestrator to merge.
    """
    actions        = []
    owned_planets  = {p[0]: p for p in obs.get("planets", []) if p[1] == my_player_id}
    committed_mass = {p_id: 0 for p_id in owned_planets}

    for target_id, target_p in owned_planets.items():
        production       = target_p[6]
        current_garrison = target_p[5]

        threats = fleet_tracker.get_incoming_threats(target_id)
        if not threats:
            continue

        # Merge hostile threats and already-registered friendly inbound events
        friendly_inbound = outgoing_ledger.get_friendly_reinforcements(target_id)
        all_events = []
        for t in threats:
            all_events.append(("hostile", t["impact_step"], t["ships"]))
        for r in friendly_inbound:
            all_events.append(("friendly", r["arrival_step"], r["ships"]))
        all_events.sort(key=lambda e: e[1])

        simulated_garrison = current_garrison
        last_step          = current_step
        fatal_threat       = None

        for kind, step, ships in all_events:
            dt = step - last_step
            simulated_garrison += production * dt
            last_step           = step

            if kind == "friendly":
                simulated_garrison += ships
            else:
                simulated_garrison -= ships
                if simulated_garrison < 0:
                    fatal_threat = {"impact_step": step}
                    break

        if not fatal_threat:
            continue

        deficit            = abs(simulated_garrison)
        target_impact_step = fatal_threat["impact_step"]

        best_donor_id     = None
        best_launch_angle = None
        best_payload      = None
        best_arr_step     = None   # FIX 6: track so we avoid a final re-call
        min_mass_cost     = float("inf")

        for donor_id, donor_p in owned_planets.items():
            if donor_id == target_id:
                continue
            # Sum all hostile incoming ships to this donor
            hostile_incoming = sum(
                f["ships"] for f in fleet_tracker.get_incoming_threats(donor_id)
                if f["owner"] != my_player_id
            )

            # Project garrison to the earliest hostile arrival so we know what survives
            earliest_threat_step = min(
                (f["impact_step"] for f in fleet_tracker.get_incoming_threats(donor_id)
                if f["owner"] != my_player_id and f["impact_step"] is not None),
                default=current_step
            )
            dt = max(0, earliest_threat_step - current_step)
            projected_donor_garrison = donor_p[5] + donor_p[6] * dt

            # Only ships above the survival floor are safe to donate
            safe_garrison_floor = hostile_incoming + 1
            airborne  = outgoing_ledger.ships_committed_from(donor_id)
            available = projected_donor_garrison - safe_garrison_floor - airborne - committed_mass[donor_id]
            if available <= deficit:
                continue

            test_payload = deficit + 1

            arr_step, l_angle = calculate_fleet_arrival(
                donor_id, target_id, test_payload, current_step,
                obs, planet_matrix, planet_radii
            )
            # FIX 3: `if not arr_step` would silently drop step 0; use explicit None check
            if arr_step is None:
                continue

            if arr_step >= target_impact_step:
                req_turns = (target_impact_step - 1) - current_step
                if req_turns <= 0:
                    continue
                
                tx, ty = planet_matrix[target_id][target_impact_step - 1]
                lx, ly = donor_p[2], donor_p[3]
                dist   = max(math.hypot(tx - lx, ty - ly) - donor_p[4] - target_p[4], 0.0)
                
                center_dist = math.hypot(tx - lx, ty - ly)
                if center_dist > 0:
                    dx_line = tx - lx
                    dy_line = ty - ly
                    sun_x, sun_y, sun_radius = 50.0, 50.0, 10.0
                    ld  = abs(dx_line * (ly - sun_y) - dy_line * (lx - sun_x)) / center_dist
                    dot = ((sun_x - lx) * dx_line + (sun_y - ly) * dy_line) / (center_dist ** 2)
                    if ld <= sun_radius and 0 <= dot <= 1:
                        continue   # this donor's straight-line path passes through the sun

                req_speed = dist / req_turns if req_turns > 0 else 999
                if req_speed > 6.0:
                    continue

                if req_speed > 1.0:
                    exponent      = ((req_speed - 1.0) / 5.0) ** (2.0 / 3.0)
                    heavy_payload = math.ceil(math.exp(math.log(1000.0) * exponent))
                else:
                    heavy_payload = test_payload

                test_payload = max(test_payload, heavy_payload)
                if test_payload > available:
                    continue

                # FIX 6: verify the heavy payload can make it in time using
                # direct geometry, avoiding a second calculate_fleet_arrival call
                # in the hot path.
                actual_speed = 1.0 + 5.0 * (
                    math.log(max(test_payload, 2)) / math.log(1000.0)
                ) ** 1.5
                if dist > actual_speed * req_turns:
                    continue
                arr_step = target_impact_step - 1
                l_angle  = math.atan2(ty - donor_p[3], tx - donor_p[2])

            # FIX 3: explicit None check throughout
            if arr_step is not None and arr_step < target_impact_step:
                if test_payload < min_mass_cost:
                    min_mass_cost     = test_payload
                    best_donor_id     = donor_id
                    best_launch_angle = l_angle
                    best_payload      = test_payload
                    best_arr_step     = arr_step   # FIX 6: cache so no re-call needed

        if best_donor_id is not None:
            actions.append([best_donor_id, best_launch_angle, best_payload])
            committed_mass[best_donor_id] += best_payload
            # FIX 6: use cached arrival step instead of calling calculate_fleet_arrival again
            if best_arr_step is not None:
                outgoing_ledger.register_launch(
                    best_donor_id, target_id, best_payload, current_step, best_arr_step
                )

    return actions, committed_mass


# ─────────────────────────────────────────────────────────────────────────────
# REGRET-MINIMIZED ATTACK DISPATCHER — helpers
# ─────────────────────────────────────────────────────────────────────────────

def _opponent_reach_probability(target_id, target_impact_step, current_step,
                                obs, planet_matrix, my_player_id,
                                fleet_tracker=None):   # add fleet_tracker param
    # If a confirmed enemy fleet is already heading here, contest is certain
    if fleet_tracker is not None:
        for f in fleet_tracker.active_fleets.values():
            if (f["owner"] != my_player_id
                    and f["target_planet_id"] == target_id
                    and f["impact_step"] is not None
                    and f["impact_step"] <= target_impact_step + 5):  # ±5 step window
                return 1.0

    # Fall back to the heuristic for unconfirmed threats
    sun_x, sun_y, sun_radius = 50.0, 50.0, 10.0
    if target_id not in planet_matrix:
        return 0.0
    trajectory = planet_matrix[target_id]
    if target_impact_step >= len(trajectory):
        return 0.0

    tx, ty            = trajectory[target_impact_step]
    threat_score      = 0.0
    total_enemy_ships = 0.0

    for p in obs.get("planets", []):
        owner = p[1]
        if owner == my_player_id or owner == -1:
            continue
        ex, ey, e_radius = p[2], p[3], p[4]
        e_ships          = p[5]
        dx, dy           = tx - ex, ty - ey
        center_dist      = math.hypot(dx, dy)
        travel_dist      = max(center_dist - e_radius - 2.0, 0.1)

        available_turns = target_impact_step - current_step
        if available_turns <= 0:
            continue
        if travel_dist > 6.0 * available_turns:
            continue

        if center_dist > 0:
            ld  = abs(dx * (ey - sun_y) - dy * (ex - sun_x)) / center_dist
            dot = ((sun_x - ex) * dx + (sun_y - ey) * dy) / (center_dist ** 2)
            if ld <= sun_radius and 0 <= dot <= 1:
                continue

        threat_score      += (1.0 - travel_dist / (6.0 * available_turns)) * e_ships
        total_enemy_ships += e_ships

    if total_enemy_ships == 0:
        return 0.0
    raw = threat_score / (total_enemy_ships + 1e-6)
    return min(1.0 / (1.0 + math.exp(-10.0 * (raw - 0.3))), 0.95)


def _compute_delta_v(target_planet, impact_step, ships_sent, current_step, total_my_ships):
    prod        = target_planet[6]
    t_remaining = 500 - current_step
    eta         = impact_step - current_step
    return prod * (t_remaining - eta) - ships_sent * (t_remaining / max(total_my_ships, 1))


def _expected_delta_v(delta_v, p_opponent_contests):
    return delta_v * (1.0 - 0.6 * p_opponent_contests)


# ─────────────────────────────────────────────────────────────────────────────
# REGRET-MINIMIZED ATTACK DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────
def dispatch_regret_minimized_attacks(current_step, obs, planet_matrix,
                                      fleet_tracker, outgoing_ledger,
                                      my_player_id, planet_radii,
                                      already_committed_mass=None,
                                      topology_cache=None):
    """
    Scores all (launcher, enemy-target) pairs by expected delta-V and greedily
    assigns non-conflicting launches.

    FIX 5: accepts topology_cache and uses compute_planet_defensibility to
    produce an accurate ships_to_capture estimate that accounts for network
    support from neighbouring same-owner planets, not just the raw garrison.
    FIX 8: passes fleet_tracker to _project_garrison so third-party incoming
    fleets are deducted when estimating target garrison at impact time.
    FIX 9: uses all owned planets as launchers (was top 50%, inconsistent with
    neutral expansion which already used all planets).
    FIX 3: all arr_step checks use explicit `is not None`.
    """
    actions     = []
    all_planets = obs.get("planets", [])
    owned       = [p for p in all_planets if p[1] == my_player_id]
    if not owned:
        return actions

    total_my_ships = sum(p[5] for p in owned)
    for f in fleet_tracker.active_fleets.values():
        if f["owner"] == my_player_id:
            total_my_ships += f["ships"]

    # FIX 9: all owned planets are eligible launchers, not just the top 50%.
    # Small, well-positioned planets are often the cheapest vector to nearby
    # targets; the garrison-based sort still prioritises larger planets first.
    owned_sorted = sorted(owned, key=lambda p: p[5], reverse=True)

    committed = {p[0]: 0 for p in owned}
    if already_committed_mass:
        for pid, mass in already_committed_mass.items():
            if pid in committed:
                committed[pid] = mass

    targets = [p for p in all_planets if p[1] != my_player_id]
    if not targets:
        return actions

    scored_actions = []

    for launcher in owned_sorted:
        l_id      = launcher[0]
        airborne  = outgoing_ledger.ships_committed_from(l_id)
        available = launcher[5] - airborne - committed[l_id]

        if available <= 1:
            continue

        for target in targets:
            t_id = target[0]

            # Skip only when registered inbound is already sufficient to capture
            inbound = outgoing_ledger.inbound_friendly_ships(t_id)
            t_garr  = _project_garrison(target, current_step, current_step)
            if inbound > t_garr:
                continue

            impact_step, launch_angle = calculate_fleet_arrival(
                l_id, t_id, available, current_step, obs, planet_matrix, planet_radii
            )
            # FIX 3: explicit None check
            if impact_step is None:
                continue

            # FIX 5: use defensibility for accurate capture threshold on enemy planets.
            # FIX 8: pass fleet_tracker to subtract third-party hostile fleets.
            projected_garrison = _project_garrison(
                target, current_step, impact_step, fleet_tracker, my_player_id
            )
            if topology_cache is not None and target[1] != -1:
                # Defensibility includes network support from neighbouring same-owner
                # planets; use it as the floor for ships_to_capture so well-defended
                # enemy planets aren't attacked with too few ships.
                defensibility    = compute_planet_defensibility(
                    target, current_step, obs, topology_cache
                )
                ships_to_capture = max(projected_garrison, math.ceil(defensibility)) + 1
            else:
                ships_to_capture = projected_garrison + 1

            if ships_to_capture > available:
                continue

            ships_sent = ships_to_capture
            if ships_sent != available:
                impact_step, launch_angle = calculate_fleet_arrival(
                    l_id, t_id, ships_sent, current_step, obs, planet_matrix, planet_radii
                )
                # FIX 3: explicit None check
                if impact_step is None:
                    continue

            dv = _compute_delta_v(target, impact_step, ships_sent,
                                   current_step, total_my_ships)
            if dv <= 0:
                continue

            p_contest = _opponent_reach_probability(
                t_id, impact_step, current_step, obs, planet_matrix, my_player_id,
                fleet_tracker=fleet_tracker
            )
            e_dv = _expected_delta_v(dv, p_contest)
            if e_dv <= 0:
                continue

            scored_actions.append({
                "launcher_id": l_id,
                "target_id":   t_id,
                "ships":       ships_sent,
                "angle":       launch_angle,
                "impact_step": impact_step,
                "e_dv":        e_dv,
            })

    scored_actions.sort(key=lambda x: x["e_dv"], reverse=True)
    used_targets = set()

    for act in scored_actions:
        l_id = act["launcher_id"]
        t_id = act["target_id"]

        if t_id in used_targets:
            continue

        launcher_planet = next((p for p in owned if p[0] == l_id), None)
        if not launcher_planet:
            continue

        airborne      = outgoing_ledger.ships_committed_from(l_id)
        available_now = launcher_planet[5] - airborne - committed[l_id]
        if act["ships"] > available_now:
            continue

        actions.append([l_id, act["angle"], act["ships"]])
        committed[l_id] += act["ships"]
        used_targets.add(t_id)
        outgoing_ledger.register_launch(
            l_id, t_id, act["ships"], current_step, act["impact_step"]
        )

    return actions


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK TOPOLOGY CACHE
# ─────────────────────────────────────────────────────────────────────────────
class NetworkTopologyCache:
    """
    Pre-computes pairwise planet distances for O(1) lookup.
    - Static / Static: constant.
    - Orbiting / Orbiting: constant (law of cosines on fixed relative angle).
    - Orbiting / Static: periodic, one full cycle pre-computed.
    """
    def __init__(self, obs):
        self.sun_x, self.sun_y = 50.0, 50.0
        self.omega  = obs.get("angular_velocity", 0.0)
        self.period = max(1, math.ceil(2 * math.pi / self.omega)) if self.omega > 0 else 500
        self.static_distances  = {}
        self.dynamic_distances = {}
        self._build_topology(obs.get("initial_planets", []))

    def _build_topology(self, initial_planets):
        static_nodes   = {}
        orbiting_nodes = {}

        for p in initial_planets:
            p_id, _, x0, y0, radius = p[0], p[1], p[2], p[3], p[4]
            dx, dy   = x0 - self.sun_x, y0 - self.sun_y
            dist_sun = math.hypot(dx, dy)
            self.static_distances[p_id] = {}
            if dist_sun + radius >= 50.0:
                static_nodes[p_id] = (x0, y0)
            else:
                orbiting_nodes[p_id] = {"R": dist_sun, "theta": math.atan2(dy, dx)}
                self.dynamic_distances[p_id] = {}

        for id_a, pos_a in static_nodes.items():
            for id_b, pos_b in static_nodes.items():
                self.static_distances[id_a][id_b] = math.hypot(
                    pos_a[0] - pos_b[0], pos_a[1] - pos_b[1]
                )

        for id_a, pa in orbiting_nodes.items():
            for id_b, pb in orbiting_nodes.items():
                if id_a == id_b:
                    self.static_distances[id_a][id_b] = 0.0
                    continue
                r_a, r_b    = pa["R"], pb["R"]
                delta_theta = pa["theta"] - pb["theta"]
                self.static_distances[id_a][id_b] = math.sqrt(
                    r_a**2 + r_b**2 - 2*r_a*r_b*math.cos(delta_theta)
                )

        for orb_id, po in orbiting_nodes.items():
            r_orb, theta_orb_0 = po["R"], po["theta"]
            for stat_id, pos_stat in static_nodes.items():
                dx_s     = pos_stat[0] - self.sun_x
                dy_s     = pos_stat[1] - self.sun_y
                r_stat   = math.hypot(dx_s, dy_s)
                theta_st = math.atan2(dy_s, dx_s)
                curve    = [0.0] * self.period
                for t in range(self.period):
                    delta    = theta_st - (theta_orb_0 + self.omega * t)
                    curve[t] = math.sqrt(
                        r_orb**2 + r_stat**2 - 2*r_orb*r_stat*math.cos(delta)
                    )
                self.dynamic_distances[orb_id][stat_id] = curve

    def get_distance(self, id_a, id_b, step):
        if id_b in self.static_distances.get(id_a, {}):
            return self.static_distances[id_a][id_b]
        if id_a in self.dynamic_distances and id_b in self.dynamic_distances[id_a]:
            return self.dynamic_distances[id_a][id_b][step % self.period]
        if id_b in self.dynamic_distances and id_a in self.dynamic_distances[id_b]:
            return self.dynamic_distances[id_b][id_a][step % self.period]
        return 999.0


# ─────────────────────────────────────────────────────────────────────────────
# DEFENSIBILITY SCORER
# ─────────────────────────────────────────────────────────────────────────────
def compute_planet_defensibility(target_p, current_step, obs, topology_cache):
    """
    Garrison + proximity-weighted support from same-owner neighbours.
    Used by dispatch_regret_minimized_attacks to avoid under-estimating the
    true capture cost of hardened enemy planets and to prioritise attacking
    the least defensible targets first.
    """
    target_id     = target_p[0]
    target_owner  = target_p[1]
    base_garrison = target_p[5]
    d_critical    = 35.0
    gamma         = 2.0
    network_support = 0.0

    for p in obs.get("planets", []):
        if p[1] != target_owner or p[0] == target_id or p[5] <= 1:
            continue
        dist = topology_cache.get_distance(target_id, p[0], current_step)
        if dist < d_critical:
            network_support += p[5] * (1.0 - (dist / d_critical) ** gamma)

    return base_garrison + network_support


# ─────────────────────────────────────────────────────────────────────────────
# MASTER AGENT ORCHESTRATOR
# Four-pass pipeline:
#   Pass 1 — Reinforcement   (defend planets that will fall)
#   Pass 2 — Parasite        (steal planets mid-capture)
#   Pass 3 — Neutral expansion (fast, minimum-payload neutral grabs)
#   Pass 4 — Regret-minimized attacks (scored offensive vs enemy planets,
#             now using defensibility from NetworkTopologyCache)
# The OutgoingLedger threads through all four passes, giving each an accurate
# view of what ships are already airborne so nothing is double-spent.
# ─────────────────────────────────────────────────────────────────────────────
def agent(obs):
    global _planet_matrix, _fleet_tracker, _planet_radii, _outgoing_ledger, \
           _topology_cache, _game_id

    player_id    = obs.get("player", 0)
    current_step = obs.get("step", 0)

    # FIX 2: hash initial planet positions instead of using id() on the list
    # object.  id() reuses memory addresses after GC, so two different episodes
    # could silently share stale global state.  A content hash is stable.
    obs_game_id = hash(tuple(
        (p[0], p[2], p[3]) for p in obs.get("initial_planets", [])
    ))
    if current_step == 0 or _planet_matrix is None or _game_id != obs_game_id:
        _planet_matrix   = build_world_state_matrix(obs)
        _planet_radii    = build_planet_radii(obs)
        _fleet_tracker   = FleetTracker(_planet_radii)
        _outgoing_ledger = OutgoingLedger()
        _topology_cache  = NetworkTopologyCache(obs)   # FIX 5: initialise
        _game_id         = obs_game_id

    _outgoing_ledger.prune(current_step)
    # FIX 7: pass player_id + ledger so friendly fleet ray-marches are skipped
    _fleet_tracker.update(
        current_step, obs.get("fleets", []), _planet_matrix,
        my_player_id=player_id, outgoing_ledger=_outgoing_ledger
    )

    all_actions    = []
    committed_mass = {}

    # ── Pass 1: Reinforcement ────────────────────────────────────────────────
    reinf_actions, reinf_committed = dispatch_network_reinforcements(
        current_step, obs, _planet_matrix, _fleet_tracker,
        _outgoing_ledger, player_id, _planet_radii
    )
    all_actions.extend(reinf_actions)
    for pid, mass in reinf_committed.items():
        committed_mass[pid] = committed_mass.get(pid, 0) + mass

    # ── Pass 2: Parasite strikes ─────────────────────────────────────────────
    all_planets       = obs.get("planets", [])
    enemy_and_neutral = [p for p in all_planets if p[1] != player_id]

    for target in enemy_and_neutral:
        strike = evaluate_parasite_attack(
            target[0], current_step, obs, _planet_matrix,
            _fleet_tracker, player_id, _planet_radii
        )
        if strike and strike.get("net_gain", 0) > 0:
            from_id  = strike["from_planet_id"]
            airborne = _outgoing_ledger.ships_committed_from(from_id)
            already  = committed_mass.get(from_id, 0)
            launcher = next((p for p in all_planets if p[0] == from_id), None)
            if launcher and (airborne + already + strike["ships"]) <= launcher[5]:
                all_actions.append([from_id, strike["angle"], strike["ships"]])
                committed_mass[from_id] = already + strike["ships"]
                _outgoing_ledger.register_launch(
                    from_id, strike["target_planet_id"],
                    strike["ships"], current_step, strike["impact_step"]
                )

    # ── Pass 3: Neutral expansion ────────────────────────────────────────────
    neutral_actions, neutral_committed = dispatch_neutral_expansion(
        current_step, obs, _planet_matrix, _planet_radii,
        _fleet_tracker, _outgoing_ledger, player_id,
        already_committed_mass=committed_mass
    )
    all_actions.extend(neutral_actions)
    for pid, mass in neutral_committed.items():
        committed_mass[pid] = committed_mass.get(pid, 0) + mass

    # ── Pass 4: Regret-minimized attacks on enemy planets ───────────────────
    expansion_actions = dispatch_regret_minimized_attacks(
        current_step, obs, _planet_matrix, _fleet_tracker,
        _outgoing_ledger, player_id, _planet_radii,
        already_committed_mass=committed_mass,
        topology_cache=_topology_cache   # FIX 5: wire in for defensibility scoring
    )
    all_actions.extend(expansion_actions)

    return all_actions