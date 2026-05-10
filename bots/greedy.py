"""
Greedy Bot — targets the highest-production unowned planet it can afford.
Sends 60% of ships from each owned planet every turn.
"""
import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

SEND_RATIO = 0.6  # fraction of ships to send each turn
MIN_GARRISON = 5  # always keep at least this many ships at home


def agent(obs):
    moves = []
    player = obs["player"] if isinstance(obs, dict) else obs.player
    raw_planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]

    my_planets = [p for p in planets if p.owner == player]
    targets = sorted(
        [p for p in planets if p.owner != player],
        key=lambda t: -t.production  # highest production first
    )

    if not targets:
        return moves

    for mine in my_planets:
        ships_to_send = int(mine.ships * SEND_RATIO)
        if ships_to_send <= MIN_GARRISON:
            continue

        # Pick the highest-production target we can capture
        target = None
        for t in targets:
            if ships_to_send > t.ships:
                target = t
                break
        if target is None:
            target = targets[0]  # send anyway to apply pressure

        angle = math.atan2(target.y - mine.y, target.x - mine.x)
        moves.append([mine.id, angle, ships_to_send])

    return moves
