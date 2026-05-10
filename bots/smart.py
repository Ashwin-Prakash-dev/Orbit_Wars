"""
smart.py - Orbit Wars heuristic bot v2

Key improvements over sniper:
1. Orbit prediction: aim where planet WILL BE, not where it is now
2. Production-weighted targeting: prefer high-production planets
3. Accumulate ships before sending: wait until we can send a meaningful fleet
4. Defend: detect incoming enemy fleets and reinforce
5. Comet awareness: grab comets when they appear
6. Sun avoidance: don't aim fleets through the sun
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

# Constants
MAX_SPEED      = 6.0
SUN_X, SUN_Y  = 50.0, 50.0
SUN_RADIUS     = 10.0
BOARD          = 100.0

# Tunable
MIN_SEND       = 3      # never send fewer than this
GARRISON_FRAC  = 0.25   # keep this fraction of ships at home


def fleet_speed(ships):
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(max(ships, 1)) / math.log(1000)) ** 1.5


def travel_turns(dist, ships):
    return dist / fleet_speed(ships)


def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def is_orbiting(planet):
    return dist(planet.x, planet.y, SUN_X, SUN_Y) < 42


def predict_pos(planet, init_map, ang_vel, turns):
    """Predict where an orbiting planet will be after `turns` steps."""
    if not is_orbiting(planet) or abs(ang_vel) < 1e-9:
        return planet.x, planet.y
    init = init_map.get(planet.id)
    if init is None:
        return planet.x, planet.y
    r = dist(init.x, init.y, SUN_X, SUN_Y)
    cur_angle = math.atan2(planet.y - SUN_Y, planet.x - SUN_X)
    future_angle = cur_angle + ang_vel * turns
    return SUN_X + r * math.cos(future_angle), SUN_Y + r * math.sin(future_angle)


def aim_angle(sx, sy, tx, ty):
    return math.atan2(ty - sy, tx - sx)


def path_hits_sun(sx, sy, tx, ty):
    """Does the straight line from (sx,sy) to (tx,ty) pass through the sun?"""
    dx, dy = tx - sx, ty - sy
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return False
    # distance from sun center to line
    ex, ey = dx / length, dy / length
    fx, fy = SUN_X - sx, SUN_Y - sy
    t = fx * ex + fy * ey
    t = max(0, min(length, t))
    closest_x = sx + t * ex
    closest_y = sy + t * ey
    return dist(closest_x, closest_y, SUN_X, SUN_Y) < SUN_RADIUS + 2


def fleet_aimed_at(fleet, planet):
    """Is this fleet heading toward this planet?"""
    dx = planet.x - fleet.x
    dy = planet.y - fleet.y
    dot   =  dx * math.cos(fleet.angle) + dy * math.sin(fleet.angle)
    cross = abs(dx * math.sin(fleet.angle) - dy * math.cos(fleet.angle))
    return dot > 0 and cross < planet.radius + 4


def score_target(src, target, tx, ty, ships, step, ang_vel):
    """
    Score how good it is to attack `target` from `src`.
    Higher = better.
    """
    d = dist(src.x, src.y, tx, ty)
    turns = travel_turns(d, ships)

    # Can't capture
    if ships <= target.ships:
        return -9999

    # Penalise if path goes through sun
    if path_hits_sun(src.x, src.y, tx, ty):
        return -9999

    # Out of bounds destination
    if not (0 <= tx <= BOARD and 0 <= ty <= BOARD):
        return -9999

    turns_left = 500 - step
    # Production gained from owning the planet after capture
    production_value = target.production * max(0, turns_left - turns)

    # Cost = ships used (opportunity cost)
    ship_cost = ships

    # Distance penalty
    dist_penalty = turns * 0.5

    # Bonus for enemy planets (strategic value)
    enemy_bonus = 200 if target.owner >= 0 else 0

    return production_value - ship_cost - dist_penalty + enemy_bonus


def agent(obs):
    moves = []

    player      = obs["player"]           if isinstance(obs, dict) else obs.player
    raw_planets = obs["planets"]          if isinstance(obs, dict) else obs.planets
    raw_fleets  = obs["fleets"]           if isinstance(obs, dict) else obs.fleets
    ang_vel     = obs["angular_velocity"] if isinstance(obs, dict) else obs.angular_velocity
    init_raw    = obs["initial_planets"]  if isinstance(obs, dict) else obs.initial_planets
    step        = obs.get("step", 0)      if isinstance(obs, dict) else getattr(obs, "step", 0)
    comet_ids   = set(obs.get("comet_planet_ids", []) if isinstance(obs, dict) else getattr(obs, "comet_planet_ids", []))

    planets  = [Planet(*p) for p in raw_planets]
    fleets   = [Fleet(*f)  for f in raw_fleets]
    init_map = {Planet(*p).id: Planet(*p) for p in init_raw}

    my_planets  = [p for p in planets if p.owner == player]
    not_mine    = [p for p in planets if p.owner != player]
    enemy_fl    = [f for f in fleets   if f.owner != player]
    my_fl       = [f for f in fleets   if f.owner == player]

    if not my_planets:
        return moves

    # ── Incoming threat per planet ────────────────────────────────────────────
    threat = {}
    for f in enemy_fl:
        for mp in my_planets:
            if fleet_aimed_at(f, mp):
                threat[mp.id] = threat.get(mp.id, 0) + f.ships

    # ── Reinforcement already in flight per planet ────────────────────────────
    reinforce = {}
    for f in my_fl:
        # find which planet this fleet is heading toward
        for mp in my_planets:
            if fleet_aimed_at(f, mp):
                reinforce[mp.id] = reinforce.get(mp.id, 0) + f.ships

    # ── Per planet: decide action ─────────────────────────────────────────────
    for mine in my_planets:

        # How many ships can we spare?
        garrison = max(1, int(mine.ships * GARRISON_FRAC))
        incoming_threat = threat.get(mine.id, 0)
        already_defended = reinforce.get(mine.id, 0)

        # Emergency: need to defend
        net_threat = incoming_threat - mine.ships - already_defended
        if net_threat > 0:
            # Send reinforcements from this planet if it's not the threatened one
            # (handled below for other planets)
            garrison = mine.ships  # lock down — don't send out

        spare = mine.ships - garrison
        if spare < MIN_SEND:
            continue

        # ── Emergency reinforce a teammate under attack ───────────────────────
        critical = [
            p for p in my_planets
            if p.id != mine.id
            and threat.get(p.id, 0) > p.ships + reinforce.get(p.id, 0)
        ]
        if critical:
            target = min(critical, key=lambda p: dist(mine.x, mine.y, p.x, p.y))
            moves.append([mine.id, aim_angle(mine.x, mine.y, target.x, target.y), spare])
            continue

        # ── Score all capturable targets ──────────────────────────────────────
        best_score = -9999
        best_move  = None

        for t in not_mine:
            ships = spare  # send all spare

            # Predict position accounting for orbit
            d0 = dist(mine.x, mine.y, t.x, t.y)
            turns0 = travel_turns(d0, ships)
            tx, ty = predict_pos(t, init_map, ang_vel, turns0)

            # One refinement iteration
            d1 = dist(mine.x, mine.y, tx, ty)
            turns1 = travel_turns(d1, ships)
            tx, ty = predict_pos(t, init_map, ang_vel, turns1)

            sc = score_target(mine, t, tx, ty, ships, step, ang_vel)

            if sc > best_score:
                best_score = sc
                best_move  = [mine.id, aim_angle(mine.x, mine.y, tx, ty), ships]

        if best_move:
            moves.append(best_move)

    return moves
