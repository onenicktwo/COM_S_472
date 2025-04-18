import numpy as np
from main import Task
from typing import Tuple, List, Optional

# Importing the tunable PlannerAgent
from planners.jerry import PlannerAgent

# Last Result:
# Testing: history=3, buffer=10, trigger=5, target=12, pursuer=6, escape=0.3, trap=3
# Result: 8 wins, 7 ties, 15 losses (Score: 16)

# Best Result:
# Testing: history=3, buffer=8, trigger=4, target=12, pursuer=8, escape=0.4, trap=5
# Result: 18 wins, 5 ties, 7 losses (Score: 52)

def test_config(history_length, cycle_buffer, cycle_trigger, 
                w_target, w_pursuer, w_escape, w_trap):
    """Test a specific configuration and return performance metrics"""
    wins = 0
    ties = 0
    losses = 0
    tom_wins = 0
    spike_wins = 0
    
    # Test on a subset of maps
    for id in range(10):  # First 10 maps
        for run in range(3):  # 3 runs per map
            # Create a new agent with these parameters
            agent = PlannerAgent()
            agent.history_length = history_length
            agent.cycle_buffer = cycle_buffer
            agent.cycle_trigger = cycle_trigger
            agent.w_target = w_target
            agent.w_pursuer = w_pursuer
            agent.w_escape = w_escape
            agent.w_trap = w_trap
            
            # Run the test
            T = Task(id, run)
            T.agents[1] = agent  # Replace Jerry with our tuned agent
            result = T.run()
            
            # Record results
            if result[1] == 3:  # Jerry wins
                wins += 1
            elif result[1] == 1:  # Tie
                ties += 1
            else:  # Jerry loses
                losses += 1
                if result[0] == 3:  # Tom wins
                    tom_wins += 1
                elif result[2] == 3:  # Spike wins
                    spike_wins += 1
    
    return wins, ties, losses, tom_wins, spike_wins

def main():
    """Run parameter tuning experiments"""
    best_score = -1
    best_config = None
    results = []
    
    # Testing different history lengths
    history_lengths = [3, 4, 5]
    # Testing different cycle detection parameters
    cycle_buffers = [8, 10, 12]
    cycle_triggers = [3, 4, 5]
    # Testing different weighting parameters
    target_weights = [10, 12, 14]  # Pursuit weight
    pursuer_weights = [6, 8, 10]   # Evasion weight  
    escape_weights = [0.3, 0.4, 0.5]  # Escape route weight
    trap_weights = [3, 4, 5]  # Trap penalty weight
    
    # Grid search over parameters
    for history_length in history_lengths:
        for cycle_buffer in cycle_buffers:
            for cycle_trigger in cycle_triggers:
                for w_target in target_weights:
                    for w_pursuer in pursuer_weights:
                        for w_escape in escape_weights:
                            for w_trap in trap_weights:
                                # Skip configurations where cycle buffer is too small compared to trigger
                                if cycle_buffer <= cycle_trigger:
                                    continue
                                    
                                print(f"Testing: history={history_length}, buffer={cycle_buffer}, " +
                                      f"trigger={cycle_trigger}, target={w_target}, pursuer={w_pursuer}, " +
                                      f"escape={w_escape}, trap={w_trap}")
                                
                                wins, ties, losses, tom_wins, spike_wins = test_config(
                                    history_length, cycle_buffer, cycle_trigger,
                                    w_target, w_pursuer, w_escape, w_trap
                                )
                                
                                # Calculate a score (prioritizing wins)
                                score = wins * 3 + ties - losses
                                
                                # Store result
                                config = {
                                    'history_length': history_length,
                                    'cycle_buffer': cycle_buffer,
                                    'cycle_trigger': cycle_trigger,
                                    'w_target': w_target,
                                    'w_pursuer': w_pursuer,
                                    'w_escape': w_escape,
                                    'w_trap': w_trap
                                }
                                
                                results.append({
                                    'config': config,
                                    'wins': wins,
                                    'ties': ties,
                                    'losses': losses,
                                    'tom_wins': tom_wins,
                                    'spike_wins': spike_wins,
                                    'score': score
                                })
                                
                                print(f"Result: {wins} wins, {ties} ties, {losses} losses (Score: {score})")
                                
                                # Update best configuration
                                if score > best_score:
                                    best_score = score
                                    best_config = config
    
    # Sort results by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Print top 5 configurations
    print("\nTop 5 Configurations:")
    for i, result in enumerate(results[:5]):
        config = result['config']
        print(f"{i+1}. Score: {result['score']} ({result['wins']} wins, {result['ties']} ties, {result['losses']} losses)")
        print(f"   history={config['history_length']}, buffer={config['cycle_buffer']}, trigger={config['cycle_trigger']}")
        print(f"   target={config['w_target']}, pursuer={config['w_pursuer']}, escape={config['w_escape']}, trap={config['w_trap']}")
    
    # Print best configuration
    print("\nBest Configuration:")
    print(f"history_length = {best_config['history_length']}")
    print(f"cycle_buffer = {best_config['cycle_buffer']}")
    print(f"cycle_trigger = {best_config['cycle_trigger']}")
    print(f"w_target = {best_config['w_target']}")
    print(f"w_pursuer = {best_config['w_pursuer']}")
    print(f"w_escape = {best_config['w_escape']}")
    print(f"w_trap = {best_config['w_trap']}")

if __name__ == "__main__":
    main()