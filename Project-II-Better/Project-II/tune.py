import numpy as np
import concurrent.futures
import itertools
from main import Task
from typing import Tuple, List, Dict
from planners.jerry import PlannerAgent
import time
import os

# === Configurable Parameters ===
history_lengths = [2, 3, 4]
cycle_buffers = [8, 10, 12]
cycle_triggers = [3, 4, 5]

# Best weight configs to test with structural params
WEIGHT_CONFIGS = [
    {'w_target': 12.5, 'w_pursuer': 6.5, 'w_escape': 0.35, 'w_trap': 2},
    {'w_target': 13, 'w_pursuer': 7.5, 'w_escape': 0.3, 'w_trap': 1.5},
]

# Map IDs and run count
map_ids = list(range(10, 30))  # You can extend this to 100 maps for final eval
runs_per_map = 3


# Optional: Limit runtime
MAX_RUNTIME_SECONDS = None  # Or set to a number like 300 for early stopping

# === Core Test Function ===
def test_single_config(config: Dict) -> Dict:
    wins = ties = losses = tom_wins = spike_wins = 0

    for id in map_ids:
        for run in range(runs_per_map):
            agent = PlannerAgent()
            agent.history_length = config['history_length']
            agent.cycle_buffer = config['cycle_buffer']
            agent.cycle_trigger = config['cycle_trigger']
            agent.w_target = config['w_target']
            agent.w_pursuer = config['w_pursuer']
            agent.w_escape = config['w_escape']
            agent.w_trap = config['w_trap']

            T = Task(id, run)
            T.agents[1] = agent  # Jerry
            result = T.run()

            if result[1] == 3:
                wins += 1
            elif result[1] == 1:
                ties += 1
            else:
                losses += 1
                if result[0] == 3:
                    tom_wins += 1
                elif result[2] == 3:
                    spike_wins += 1

    score = wins * 3 + ties - losses
    config_result = {
        'config': config,
        'wins': wins,
        'ties': ties,
        'losses': losses,
        'tom_wins': tom_wins,
        'spike_wins': spike_wins,
        'score': score
    }
    return config_result

# === Main Tuning Logic ===
def main():
    start_time = time.time()
    results = []

    # Generate configs
    param_grid = list(itertools.product(
        history_lengths, cycle_buffers, cycle_triggers, WEIGHT_CONFIGS
    ))

    param_grid = [
        (h, b, t, w) for (h, b, t, w) in param_grid
        if b > t
    ]

    print(f"Testing {len(param_grid)} configurations across {len(map_ids)} maps...")

    configs = []
    for h, b, t, w in param_grid:
        config = {
            'history_length': h,
            'cycle_buffer': b,
            'cycle_trigger': t,
            'w_target': w['w_target'],
            'w_pursuer': w['w_pursuer'],
            'w_escape': w['w_escape'],
            'w_trap': w['w_trap']
        }
        configs.append(config)

    # Parallel execution
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(test_single_config, config): config for config in configs
        }

        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            results.append(result)

            # Print update
            print(f"[{i}/{len(futures)}] Score: {result['score']} | "
                  f"Wins: {result['wins']}, Ties: {result['ties']}, Losses: {result['losses']}")

            # Early stopping if runtime exceeded
            if MAX_RUNTIME_SECONDS and (time.time() - start_time) > MAX_RUNTIME_SECONDS:
                print("Early stopping due to time limit.")
                break

    # Sort results
    results.sort(key=lambda r: (r['score'], r['wins'], r['ties']), reverse=True)


    # Print top 5
    print("\nTop 5 Configurations:")
    for i, result in enumerate(results[:5]):
        c = result['config']
        print(f"{i+1}. Score: {result['score']}, "
              f"Wins: {result['wins']}, Ties: {result['ties']}, Losses: {result['losses']}")
        print(f"   history={c['history_length']}, buffer={c['cycle_buffer']}, "
              f"trigger={c['cycle_trigger']}, target={c['w_target']}, "
              f"pursuer={c['w_pursuer']}, escape={c['w_escape']}, trap={c['w_trap']}")

if __name__ == "__main__":
    main()