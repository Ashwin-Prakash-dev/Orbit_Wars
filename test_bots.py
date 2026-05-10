#!/usr/bin/env python3
"""
test_bots.py — Local testing harness for Orbit Wars bots.

Usage:
    python test_bots.py                     # run default benchmark
    python test_bots.py --games 20          # more games for accuracy
    python test_bots.py --bot1 main.py --bot2 bots/greedy.py
    python test_bots.py --vs random         # test against the built-in random agent
"""

import argparse
import sys
import os
import time
import statistics
import warnings
warnings.filterwarnings("ignore")

os.environ["KAGGLE_ENVIRONMENTS_DISABLE_LOGGING"] = "1"

# Suppress all the OpenSpiel INFO spam
import logging
logging.disable(logging.CRITICAL)


def run_game(bot1_path, bot2_path, seed=None):
    """Run a single game. Returns (reward_p0, reward_p1, steps_taken)."""
    from kaggle_environments import make

    config = {"episodeSteps": 500}
    if seed is not None:
        config["seed"] = seed

    env = make("orbit_wars", configuration=config, debug=False)

    agents = []
    for path in [bot1_path, bot2_path]:
        if path == "random":
            agents.append("random")
        else:
            agents.append(os.path.abspath(path))

    try:
        env.run(agents)
    except Exception as e:
        print(f"  [ERROR] Game crashed: {e}")
        return None, None, None

    final = env.steps[-1]
    r0 = final[0].reward if final[0].reward is not None else 0
    r1 = final[1].reward if final[1].reward is not None else 0
    return r0, r1, len(env.steps)


def benchmark(bot1, bot2, n_games=10, swap=True):
    """
    Play bot1 vs bot2 for n_games.
    If swap=True, alternate who plays as player 0 to remove positional bias.
    """
    wins = {bot1: 0, bot2: 0, "draw": 0}
    errors = 0
    durations = []
    rewards_p1 = []

    matchups = []
    for i in range(n_games):
        if swap and i % 2 == 1:
            matchups.append((bot2, bot1, True))   # swapped
        else:
            matchups.append((bot1, bot2, False))  # normal

    print(f"\n{'─'*55}")
    print(f"  {short(bot1):>20}  vs  {short(bot2):<20}")
    print(f"  Running {n_games} games {'(with position swap)' if swap else ''}")
    print(f"{'─'*55}")

    for i, (a, b, swapped) in enumerate(matchups):
        t0 = time.time()
        r0, r1, steps = run_game(a, b, seed=i * 7 + 13)
        elapsed = time.time() - t0

        if r0 is None:
            errors += 1
            print(f"  Game {i+1:>3}: ERROR")
            continue

        durations.append(elapsed)

        if swapped:
            # bot1 was player 1, bot2 was player 0
            bot1_r, bot2_r = r1, r0
        else:
            bot1_r, bot2_r = r0, r1

        rewards_p1.append(bot1_r)

        if bot1_r > bot2_r:
            outcome = f"✓ {short(bot1)} wins"
            wins[bot1] += 1
        elif bot2_r > bot1_r:
            outcome = f"✗ {short(bot2)} wins"
            wins[bot2] += 1
        else:
            outcome = "= Draw"
            wins["draw"] += 1

        print(f"  Game {i+1:>3}: {outcome}  "
              f"(scores {bot1_r:+.0f} / {bot2_r:+.0f}, {steps} steps, {elapsed:.1f}s)")

    total = n_games - errors
    if total == 0:
        print("  All games errored out.")
        return

    print(f"{'─'*55}")
    wr = wins[bot1] / total * 100
    print(f"  {short(bot1)} win rate : {wins[bot1]}/{total}  ({wr:.1f}%)")
    print(f"  {short(bot2)} win rate : {wins[bot2]}/{total}  ({wins[bot2]/total*100:.1f}%)")
    print(f"  Draws         : {wins['draw']}/{total}  ({wins['draw']/total*100:.1f}%)")
    if durations:
        print(f"  Avg game time : {statistics.mean(durations):.1f}s  "
              f"(min {min(durations):.1f}s, max {max(durations):.1f}s)")
    print(f"{'─'*55}\n")


def short(path):
    if path == "random":
        return "random"
    return os.path.splitext(os.path.basename(path))[0]


def main():
    parser = argparse.ArgumentParser(description="Orbit Wars local benchmark")
    parser.add_argument("--bot1",  default="main.py",         help="Path to bot 1 (default: main.py)")
    parser.add_argument("--bot2",  default="bots/sniper.py",  help="Path to bot 2 (default: bots/sniper.py)")
    parser.add_argument("--vs",    default=None,              help="Shorthand: test main.py vs this bot (random / path)")
    parser.add_argument("--games", type=int, default=10,      help="Number of games (default: 10)")
    parser.add_argument("--no-swap", action="store_true",     help="Don't swap positions")
    args = parser.parse_args()

    # Change to script directory so relative paths work
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.vs:
        bot2 = args.vs if args.vs == "random" else args.vs
        benchmark("main.py", bot2, n_games=args.games, swap=not args.no_swap)
    else:
        benchmark(args.bot1, args.bot2, n_games=args.games, swap=not args.no_swap)


if __name__ == "__main__":
    main()
