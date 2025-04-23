from main import Task
from planners.jerry import PlannerAgent
import numpy as np

# Define both configurations
CONFIG_A = {
    'name': 'A',
    'history_length': 3,
    'cycle_buffer': 10,
    'cycle_trigger': 4,
    'w_target': 12.5,
    'w_pursuer': 6.5,
    'w_escape': 0.35,
    'w_trap': 2.0
}

CONFIG_B = {
    'name': 'B',
    'history_length': 3,
    'cycle_buffer': 10,
    'cycle_trigger': 4,
    'w_target': 13.0,
    'w_pursuer': 7.5,
    'w_escape': 0.3,
    'w_trap': 1.5
}

configs = [CONFIG_A, CONFIG_B]

NUM_TRIALS = 5  # You can change to 3 if needed

def run_config(config):
    wins = 0
    ties = 0
    losses = 0

    for map_id in range(100):
        for run in range(5):
            agent = PlannerAgent()
            agent.history_length = config['history_length']
            agent.cycle_buffer = config['cycle_buffer']
            agent.cycle_trigger = config['cycle_trigger']
            agent.w_target = config['w_target']
            agent.w_pursuer = config['w_pursuer']
            agent.w_escape = config['w_escape']
            agent.w_trap = config['w_trap']

            T = Task(map_id, run)
            T.agents[1] = agent  # Replace Jerry
            result = T.run()

            if result[1] == 3:
                wins += 1
            elif result[1] == 1:
                ties += 1
            else:
                losses += 1

    score = 3 * wins + ties - losses
    return wins, ties, losses, score

if __name__ == '__main__':
    summary = []

    for config in configs:
        total_wins = total_ties = total_losses = total_score = 0
        print(f"\nRunning Config {config['name']} for {NUM_TRIALS} trials...")

        for trial in range(NUM_TRIALS):
            print(f"  Trial {trial+1}/{NUM_TRIALS}")
            wins, ties, losses, score = run_config(config)
            total_wins += wins
            total_ties += ties
            total_losses += losses
            total_score += score

        avg_wins = total_wins / NUM_TRIALS
        avg_ties = total_ties / NUM_TRIALS
        avg_losses = total_losses / NUM_TRIALS
        avg_score = total_score / NUM_TRIALS

        summary.append({
            'name': config['name'],
            'avg_wins': avg_wins,
            'avg_ties': avg_ties,
            'avg_losses': avg_losses,
            'avg_score': avg_score
        })

    # Sort by average score
    summary.sort(key=lambda r: r['avg_score'], reverse=True)

    print("\n===== Final Average Results =====")
    for r in summary:
        print(f"Config {r['name']}: "
              f"{r['avg_wins']:.2f} wins, "
              f"{r['avg_ties']:.2f} ties, "
              f"{r['avg_losses']:.2f} losses → "
              f"Avg Score: {r['avg_score']:.2f}")

    print(f"\n🏆 Best Config: {summary[0]['name']} (Avg Score: {summary[0]['avg_score']:.2f})")