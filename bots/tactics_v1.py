import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT

SUN_X, SUN_Y   = 50.0, 50.0
SUN_RADIUS      = 10.0
SUN_BUFFER      = 2.0
MAX_SPEED       = 6.0
BOARD_SIZE      = 100.0
MIN_GARRISON    = 12
STRIP_RATIO     = 0.80
SEND_RATIO      = 0.75
MAX_INTERCEPT_ITER = 30

def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5

def path_crosses_sun(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - SUN_X, y1 - SUN_Y
    a = dx * dx + dy * dy
    if a == 0:
        return False
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - SUN_RADIUS ** 2
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)

def tangent_detour_angle(px, py, target_x, target_y):
    dx, dy   = SUN_X - px, SUN_Y - py
    d        = math.sqrt(dx * dx + dy * dy)
    if d <= SUN_RADIUS + SUN_BUFFER:
        return math.atan2(target_y - py, target_x - px)
    base     = math.atan2(dy, dx)
    half     = math.asin(min(1.0, (SUN_RADIUS + SUN_BUFFER) / d))
    left_wp  = _waypoint(px, py, base - half, d * 1.05)
    right_wp = _waypoint(px, py, base + half, d * 1.05)
    cost_left  = dist(px, py, *left_wp)  + dist(*left_wp,  target_x, target_y)
    cost_right = dist(px, py, *right_wp) + dist(*right_wp, target_x, target_y)
    wp = left_wp if cost_left < cost_right else right_wp
    return math.atan2(wp[1] - py, wp[0] - px)

def _waypoint(ox, oy, angle, length):
    return ox + length * math.cos(angle), oy + length * math.sin(angle)

def bearing(x1, y1, x2, y2):
    return math.atan2(y2 - y1, x2 - x1)

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def is_orbiting(planet):
    r = dist(planet.x, planet.y, SUN_X, SUN_Y)
    return (r + planet.radius) < ROTATION_RADIUS_LIMIT

def current_angle(planet):
    return math.atan2(planet.y - SUN_Y, planet.x - SUN_X)

def orbital_radius(planet):
    return dist(planet.x, planet.y, SUN_X, SUN_Y)

def future_pos(planet, turns, angular_velocity):
    if not is_orbiting(planet):
        return planet.x, planet.y
    theta = current_angle(planet) + angular_velocity * turns
    r     = orbital_radius(planet)
    return SUN_X + r * math.cos(theta), SUN_Y + r * math.sin(theta)

def solve_intercept(sx, sy, planet, ships, angular_velocity):
    speed = fleet_speed(ships)
    lo, hi = 1.0, 300.0
    for _ in range(MAX_INTERCEPT_ITER):
        mid    = (lo + hi) / 2.0
        fx, fy = future_pos(planet, mid, angular_velocity)
        d      = dist(sx, sy, fx, fy)
        if d < speed * mid:
            hi = mid
        else:
            lo = mid
    t      = (lo + hi) / 2.0
    fx, fy = future_pos(planet, t, angular_velocity)
    return fx, fy, int(math.ceil(t))

def enemy_fleets_targeting(planet_id, fleets, player):
    total = 0
    for f in fleets:
        if f.owner != player and f.from_planet_id == planet_id:
            total += f.ships
    return total

def friendly_fleets_targeting(planet_id, fleets, player):
    total = 0
    for f in fleets:
        if f.owner == player and f.from_planet_id == planet_id:
            total += f.ships
    return total

def target_score(src_planet, tgt_planet, fleets, player, angular_velocity, comet_ids, step):
    ships_to_send = max(1, int(src_planet.ships * SEND_RATIO))
    ix, iy, eta   = solve_intercept(src_planet.x, src_planet.y, tgt_planet, ships_to_send, angular_velocity)
    d = dist(src_planet.x, src_planet.y, ix, iy)
    friendly_en_route = friendly_fleets_targeting(tgt_planet.id, fleets, player)
    if friendly_en_route > tgt_planet.ships * 1.5:
        return -1.0
    garrison_after_friendly = max(0, tgt_planet.ships - friendly_en_route)
    cost                    = garrison_after_friendly + 1
    prod = tgt_planet.production
    if tgt_planet.id in comet_ids:
        turns_left = 500 - step
        if turns_left < eta + 20:
            return -1.0
        prod = prod * min(1.0, (turns_left - eta) / 80.0)
    return (prod * 10.0) / (cost * (1.0 + d * 0.05))

def is_frontier(planet, all_planets, player, threshold=25.0):
    for p in all_planets:
        if p.owner != player and dist(planet.x, planet.y, p.x, p.y) < threshold:
            return True
    return False

def reinforcement_moves(my_planets, all_planets, fleets, player):
    moves    = []
    frontier = [p for p in my_planets if is_frontier(p, all_planets, player)]
    rear     = [p for p in my_planets if not is_frontier(p, all_planets, player)]
    if not frontier:
        return moves
    for src in rear:
        excess = int((src.ships - MIN_GARRISON) * STRIP_RATIO)
        if excess < 5:
            continue
        dst = min(frontier, key=lambda p: dist(src.x, src.y, p.x, p.y))
        moves.append((src, dst, excess))
    return moves

def defense_moves(my_planets, fleets, player):
    moves = []
    for p in my_planets:
        threat = enemy_fleets_targeting(p.id, fleets, player)
        if threat == 0:
            continue
        deficit = threat - p.ships + 5
        if deficit <= 0:
            continue
        donors = sorted(
            [q for q in my_planets if q.id != p.id and q.ships - MIN_GARRISON > deficit],
            key=lambda q: dist(q.x, q.y, p.x, p.y)
        )
        if donors:
            moves.append((donors[0], p, deficit))
    return moves

def agent(obs):
    planets  = [Planet(*p) for p in obs.get("planets", [])]
    fleets   = [Fleet(*f)  for f in obs.get("fleets",  [])]
    player   = obs.get("player", 0)
    ang_vel  = obs.get("angular_velocity", 0.03)
    step     = obs.get("step", 0)
    comet_ids= set(obs.get("comet_planet_ids", []))

    my_planets      = [p for p in planets if p.owner == player]
    target_planets  = [p for p in planets if p.owner != player]

    if not my_planets:
        return []

    actions = []
    used    = {}

    for src, dst, n in defense_moves(my_planets, fleets, player):
        available = src.ships - used.get(src.id, 0) - MIN_GARRISON
        n         = clamp(n, 1, available)
        if n < 1:
            continue
        ix, iy, _ = solve_intercept(src.x, src.y, dst, n, ang_vel)
        angle     = bearing(src.x, src.y, ix, iy)
        if path_crosses_sun(src.x, src.y, ix, iy):
            angle = tangent_detour_angle(src.x, src.y, ix, iy)
        actions.append([src.id, angle, n])
        used[src.id] = used.get(src.id, 0) + n

    for src, dst, n in reinforcement_moves(my_planets, planets, fleets, player):
        available = src.ships - used.get(src.id, 0) - MIN_GARRISON
        n         = clamp(n, 1, available)
        if n < 1:
            continue
        ix, iy, _ = solve_intercept(src.x, src.y, dst, n, ang_vel)
        angle     = bearing(src.x, src.y, ix, iy)
        if path_crosses_sun(src.x, src.y, ix, iy):
            angle = tangent_detour_angle(src.x, src.y, ix, iy)
        actions.append([src.id, angle, n])
        used[src.id] = used.get(src.id, 0) + n

    for src in my_planets:
        available = src.ships - used.get(src.id, 0) - MIN_GARRISON
        if available < 5:
            continue
        scored = []
        for tgt in target_planets:
            s = target_score(src, tgt, fleets, player, ang_vel, comet_ids, step)
            if s > 0:
                scored.append((s, tgt))
        if not scored:
            continue
        scored.sort(key=lambda x: -x[0])
        _, best = scored[0]
        ships_to_send = clamp(int(available * SEND_RATIO), 1, available)
        needed = best.ships + 5
        ships_to_send = clamp(max(ships_to_send, needed), 1, available)
        if ships_to_send < 1:
            continue
        ix, iy, _ = solve_intercept(src.x, src.y, best, ships_to_send, ang_vel)
        angle     = bearing(src.x, src.y, ix, iy)
        if path_crosses_sun(src.x, src.y, ix, iy):
            angle = tangent_detour_angle(src.x, src.y, ix, iy)
        actions.append([src.id, angle, ships_to_send])
        used[src.id] = used.get(src.id, 0) + ships_to_send

    return actions
