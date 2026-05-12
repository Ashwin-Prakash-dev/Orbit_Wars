"""
agent based on analysis of a top episode .

Strategy implemented:
1. Phase 1 (0-30): Rush nearest low-ship neutral; build production base
2. Phase 2 (30-200): Expand greedily; target highest prod/ships ratio neutrals
3. Phase 3 (200+): Snowball; mass attack enemy if ahead, defend if behind
4. Always: avoid sun on fleet angles, track orbiting planet positions, don't over-extend
"""

import math
from typing import List, Tuple, Optional

# Constants matching game config
CENTER = (50.0, 50.0)
SUN_RADIUS = 10.0
BOARD_SIZE = 100.0
MAX_SPEED = 6.0


# ─── Named tuples (match README) ───────────────────────────────────────────────
class Planet:
    __slots__ = ['id', 'owner', 'x', 'y', 'radius', 'ships', 'production']
    def __init__(self, id, owner, x, y, radius, ships, production):
        self.id = id; self.owner = owner; self.x = x; self.y = y
        self.radius = radius; self.ships = ships; self.production = production

class Fleet:
    __slots__ = ['id', 'owner', 'x', 'y', 'angle', 'from_planet_id', 'ships']
    def __init__(self, id, owner, x, y, angle, from_planet_id, ships):
        self.id = id; self.owner = owner; self.x = x; self.y = y
        self.angle = angle; self.from_planet_id = from_planet_id; self.ships = ships


# ─── Geometry helpers ──────────────────────────────────────────────────────────

def dist(x1, y1, x2, y2) -> float:
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def angle_to(x1, y1, x2, y2) -> float:
    """Angle in radians from (x1,y1) to (x2,y2). 0=right, pi/2=down."""
    return math.atan2(y2-y1, x2-x1)

def fleet_speed(ships: int) -> float:
    if ships <= 0:
        return 1.0
    s = 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    return min(s, MAX_SPEED)

def travel_time(ships: int, distance: float) -> float:
    speed = fleet_speed(ships)
    return distance / speed if speed > 0 else float('inf')

def segment_min_dist_to_point(x1, y1, x2, y2, px, py) -> float:
    """Minimum distance from segment (x1,y1)-(x2,y2) to point (px,py)."""
    dx, dy = x2-x1, y2-y1
    len_sq = dx*dx + dy*dy
    if len_sq == 0:
        return dist(x1, y1, px, py)
    t = max(0, min(1, ((px-x1)*dx + (py-y1)*dy) / len_sq))
    return dist(x1+t*dx, y1+t*dy, px, py)

def crosses_sun(x1, y1, x2, y2, margin=1.0) -> bool:
    """Does the segment from (x1,y1) to (x2,y2) pass within sun radius + margin?"""
    return segment_min_dist_to_point(x1, y1, x2, y2, CENTER[0], CENTER[1]) < SUN_RADIUS + margin

def safe_angle(fx, fy, tx, ty, num_tries=36) -> Optional[float]:
    """
    Find angle from (fx,fy) toward (tx,ty) that doesn't cross the sun.
    Returns direct angle if clear, otherwise tries arc around sun.
    Returns None if no safe path found (shouldn't happen in practice).
    """
    direct = angle_to(fx, fy, tx, ty)
    if not crosses_sun(fx, fy, tx, ty):
        return direct

    # Try small perturbations in increasing arc around sun
    cx, cy = CENTER
    # Tangent approach: go to a point tangent to the sun's exclusion zone then target
    sun_dist_from = dist(fx, fy, cx, cy)
    if sun_dist_from <= SUN_RADIUS + 2:
        return direct  # Already inside sun zone, just go

    # Find the two tangent points on the exclusion circle from the source
    r_exc = SUN_RADIUS + 2.5
    angle_to_sun = angle_to(fx, fy, cx, cy)
    half_angle = math.asin(min(1.0, r_exc / sun_dist_from))

    # Try going around sun clockwise or counterclockwise
    for sign in [1, -1]:
        waypoint_angle = angle_to_sun + sign * half_angle
        # Waypoint on tangent circle
        wp_dist = math.sqrt(max(0, sun_dist_from**2 - r_exc**2))
        wpx = fx + wp_dist * math.cos(waypoint_angle)
        wpy = fy + wp_dist * math.sin(waypoint_angle)
        # Clamp to board
        wpx = max(2, min(98, wpx))
        wpy = max(2, min(98, wpy))
        # Check if path to waypoint clears sun
        if not crosses_sun(fx, fy, wpx, wpy):
            return angle_to(fx, fy, wpx, wpy)

    # Fallback: direct despite sun (edge case)
    return direct


def predict_planet_position(planet: Planet, initial_planets, ang_vel: float, step: int) -> Tuple[float, float]:
    """
    Predict orbiting planet position at given step offset from now.
    Uses initial planet position + angular velocity.
    """
    # Find initial position
    init = next((p for p in initial_planets if p[0] == planet.id), None)
    if init is None:
        return planet.x, planet.y
    _, _, ix, iy, _, _, _ = init
    cx, cy = CENTER
    # Initial angle from center
    init_angle = math.atan2(iy - cy, ix - cx)
    orbital_radius = dist(ix, iy, cx, cy)
    current_angle = init_angle + ang_vel * step
    return cx + orbital_radius * math.cos(current_angle), cy + orbital_radius * math.sin(current_angle)


def is_orbiting(planet: Planet) -> bool:
    return dist(planet.x, planet.y, CENTER[0], CENTER[1]) + planet.radius < 50.0


# ─── Combat simulation ────────────────────────────────────────────────────────

def ships_needed_to_capture(garrison: int, buffer: int = 3) -> int:
    """Ships needed to capture a planet with `garrison` defenders. Add buffer for safety."""
    return garrison + buffer


# ─── Fleet threat tracking ───────────────────────────────────────────────────

def incoming_enemy_ships(planet: Planet, fleets: List[Fleet], player: int) -> int:
    """Estimate enemy ships heading toward this planet."""
    total = 0
    for f in fleets:
        if f.owner != player and f.owner >= 0:
            # Check if fleet angle roughly points at planet
            a = angle_to(f.x, f.y, planet.x, planet.y)
            if abs(a - f.angle) < 0.3 or abs(abs(a - f.angle) - 2*math.pi) < 0.3:
                total += f.ships
    return total

def friendly_incoming_ships(planet: Planet, fleets: List[Fleet], player: int) -> int:
    """Friendly ships heading toward this planet."""
    total = 0
    for f in fleets:
        if f.owner == player:
            a = angle_to(f.x, f.y, planet.x, planet.y)
            if abs(a - f.angle) < 0.3 or abs(abs(a - f.angle) - 2*math.pi) < 0.3:
                total += f.ships
    return total


# ─── Planet scoring ──────────────────────────────────────────────────────────

def neutral_value_score(source: Planet, target: Planet) -> float:
    """Score for capturing a neutral planet. Higher = more desirable."""
    d = dist(source.x, source.y, target.x, target.y)
    if d == 0:
        d = 0.01
    # Reward production, penalize distance and garrison cost
    prod_weight = target.production * 40
    cost_weight = target.ships
    return prod_weight / (d * 0.5 + cost_weight + 1)


def enemy_attack_score(source: Planet, target: Planet) -> float:
    """Score for attacking an enemy planet. Higher = more desirable."""
    d = dist(source.x, source.y, target.x, target.y)
    if d == 0:
        d = 0.01
    return (target.production * 30 + 10) / (d * 0.5 + target.ships + 1)


# ─── Main agent ───────────────────────────────────────────────────────────────

def agent(obs):
    planets_raw = obs.get("planets", [])
    fleets_raw = obs.get("fleets", [])
    player = obs.get("player", 0)
    ang_vel = obs.get("angular_velocity", 0.0)
    initial_planets = obs.get("initial_planets", [])
    comet_ids = set(obs.get("comet_planet_ids", []))
    step = obs.get("step", 0)

    planets = [Planet(*p) for p in planets_raw]
    fleets = [Fleet(*f) for f in fleets_raw]

    # Categorize planets
    my_planets = [p for p in planets if p.owner == player]
    neutral_planets = [p for p in planets if p.owner == -1]
    enemy_planets = [p for p in planets if p.owner >= 0 and p.owner != player]

    if not my_planets:
        return []

    # Total ship counts for game phase detection
    my_total_ships = sum(p.ships for p in my_planets) + sum(f.ships for f in fleets if f.owner == player)
    enemy_total_ships = sum(p.ships for p in enemy_planets) + sum(f.ships for f in fleets if f.owner != player and f.owner >= 0)

    moves = []
    reserved = {p.id: 0 for p in my_planets}  # ships reserved for defense

    # ── Phase detection ─────────────────────────────────────────────────────
    num_neutrals = len(neutral_planets)
    winning = my_total_ships > enemy_total_ships * 1.3 and len(my_planets) > len(enemy_planets)
    losing = my_total_ships < enemy_total_ships * 0.7 and step > 50

    # ── Defense: reserve ships on contested planets ──────────────────────────
    for p in my_planets:
        incoming_enemy = incoming_enemy_ships(p, fleets, player)
        if incoming_enemy > 0:
            needed = incoming_enemy + 5
            reserved[p.id] = min(p.ships, needed)

    # ── Helper: get available ships from planet ──────────────────────────────
    def available(planet: Planet, keep_min: int = 2) -> int:
        res = reserved.get(planet.id, 0)
        return max(0, planet.ships - max(res, keep_min))

    # ── Target selection ─────────────────────────────────────────────────────
    def find_best_target_for(source: Planet):
        best_target = None
        best_score = -1

        # Check neutrals (always expand while they exist)
        for t in neutral_planets:
            # Skip comets if early game (too risky, they expire)
            if t.id in comet_ids and step < 100:
                continue
            avail = available(source)
            needed = ships_needed_to_capture(t.ships, buffer=5)
            if avail < needed:
                continue
            # Don't send if another friendly fleet is already heading there
            if friendly_incoming_ships(t, fleets, player) >= needed:
                continue
            score = neutral_value_score(source, t)
            if score > best_score:
                best_score = score
                best_target = t

        # If no good neutral, check enemies (especially when winning or no neutrals)
        if (best_target is None or winning or num_neutrals == 0) and enemy_planets:
            for t in enemy_planets:
                avail = available(source)
                needed = ships_needed_to_capture(t.ships, buffer=10)
                if avail < needed:
                    continue
                score = enemy_attack_score(source, t)
                if score > best_score * 0.5:  # enemy attack threshold
                    best_score = score
                    best_target = t

        return best_target, best_score

    # ── Compute orbiting planet current position ─────────────────────────────
    def get_pos(p: Planet):
        if is_orbiting(p):
            return predict_planet_position(p, initial_planets, ang_vel, step)
        return p.x, p.y

    # ── Main targeting loop ───────────────────────────────────────────────────
    # Sort own planets by ships descending (attack from strongest first)
    sorted_my = sorted(my_planets, key=lambda p: -p.ships)

    already_targeting = {}  # target_id -> ships_committed

    for source in sorted_my:
        avail = available(source, keep_min=3)
        if avail < 5:
            continue

        target, score = find_best_target_for(source)
        if target is None:
            continue

        # Don't pile on a target that already has enough ships inbound
        committed = already_targeting.get(target.id, 0)
        needed = ships_needed_to_capture(target.ships, buffer=5)
        if committed >= needed + 5:
            continue

        # Calculate angle, accounting for orbiting planet predicted position
        sx, sy = get_pos(source)
        tx, ty = get_pos(target)

        # If target is orbiting, lead it based on travel time
        if is_orbiting(target):
            d = dist(sx, sy, tx, ty)
            t_time = travel_time(avail, d)
            # Predict where target will be when fleet arrives
            for _ in range(3):  # Iterate for accuracy
                tx, ty = predict_planet_position(target, initial_planets, ang_vel, step + int(t_time))
                d = dist(sx, sy, tx, ty)
                t_time = travel_time(avail, d)

        angle = safe_angle(sx, sy, tx, ty)
        if angle is None:
            continue

        # Ships to send: enough to capture + buffer, but don't denude home
        send = min(avail, needed + 10)
        # In winning phase, send more aggressively
        if winning:
            send = min(avail, max(send, avail // 2))
        # In losing phase, be more conservative; save for defense
        if losing:
            send = min(send, avail // 2)

        if send < 3:
            continue

        # Don't exceed actual available ships
        send = min(send, source.ships - reserved.get(source.id, 0) - 2)
        if send < 3:
            continue

        moves.append([source.id, angle, send])
        already_targeting[target.id] = already_targeting.get(target.id, 0) + send
        # Update reserved so we don't double-spend from same planet
        reserved[source.id] = reserved.get(source.id, 0) + send

    # ── Comet opportunism: cheap to capture if nearby and low garrison ────────
    if step >= 50:
        for comet_id in comet_ids:
            comet = next((p for p in neutral_planets if p.id == comet_id), None)
            if comet is None:
                continue
            # Find closest owned planet with enough ships
            for source in sorted_my:
                avail = available(source)
                needed = ships_needed_to_capture(comet.ships, buffer=3)
                if avail >= needed:
                    d = dist(source.x, source.y, comet.x, comet.y)
                    if d < 30:  # Only go for nearby comets
                        angle = safe_angle(source.x, source.y, comet.x, comet.y)
                        if angle and needed >= 3:
                            # Check not already targeting
                            if already_targeting.get(comet.id, 0) < needed:
                                moves.append([source.id, angle, needed])
                                already_targeting[comet.id] = needed
                                reserved[source.id] = reserved.get(source.id, 0) + needed
                        break

    return moves


# ─── Kaggle submission wrapper ────────────────────────────────────────────────
# The competition runner calls this function directly.
def my_agent(obs, config=None):
    return agent(obs)
