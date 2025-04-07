import numpy as np
from typing import Tuple, Optional, List
import heapq

class PlannerAgent:
    last_positions = []
    tom_history = []  # Store recent Tom positions
    spike_history = [] # Store recent Spike positions
    history_length = 3  # How many past positions to track

    @staticmethod
    def get_all_actions():
        return [np.array([dr, dc]) for dr in [-1, 0, 1] for dc in [-1, 0, 1]]

    @staticmethod
    def is_valid(pos: Tuple[int, int], world: np.ndarray) -> bool:
        r, c = pos
        return 0 <= r < world.shape[0] and 0 <= c < world.shape[1] and world[r, c] == 0

    @staticmethod
    def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def neighbors(pos: Tuple[int, int], world: np.ndarray) -> List[Tuple[int, int]]:
        return [
            (pos[0] + a[0], pos[1] + a[1])
            for a in PlannerAgent.get_all_actions()
            if PlannerAgent.is_valid((pos[0] + a[0], pos[1] + a[1]), world)
        ]

    @staticmethod
    def predict_agent(current_pos: Tuple[int, int], history: List[Tuple[int, int]], target: Tuple[int, int], world: np.ndarray) -> Tuple[int, int]:
        """Predicts the next position of an agent (Tom or Spike)."""

        # 1. One-step greedy prediction (towards their target)
        best_pos = current_pos
        best_dist = PlannerAgent.manhattan(current_pos, target)
        for n in PlannerAgent.neighbors(current_pos, world):
            d = PlannerAgent.manhattan(n, target)
            if d < best_dist:
                best_dist = d
                best_pos = n

        # 2. Velocity-based prediction (using recent history)
        if len(history) >= 2:
            # Calculate the average velocity over the last few steps
            velocity = np.array(history[-1], dtype=float) - np.array(history[-2], dtype=float)
            predicted_pos_vel = tuple((np.array(current_pos) + velocity).astype(int))

            # If the velocity-predicted position is valid, use it with a higher weight
            if PlannerAgent.is_valid(predicted_pos_vel, world):
              return predicted_pos_vel

        return best_pos # Otherwise return the one-step greedy

    @staticmethod
    def count_escape_routes(pos: Tuple[int, int], world: np.ndarray) -> int:
        return sum(1 for a in PlannerAgent.get_all_actions()
                   if np.any(a) and PlannerAgent.is_valid((pos[0] + a[0], pos[1] + a[1]), world))

    @staticmethod
    def a_star(start: Tuple[int, int], goal: Tuple[int, int], world: np.ndarray,
               predicted_tom: Tuple[int, int], avoid_tom: bool = True) -> Optional[np.ndarray]:  # Use predicted Tom
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break

            for neighbor in PlannerAgent.neighbors(current, world):
                new_cost = cost_so_far[current] + 1
                if avoid_tom and PlannerAgent.manhattan(neighbor, predicted_tom) <= 2:
                    new_cost += 5  # heavily discourage being near Tom

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + PlannerAgent.manhattan(neighbor, goal)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        if goal not in came_from:
            return np.array([0, 0])  # No path

        # Reconstruct path to get first move
        current = goal
        while came_from[current] != start:
            current = came_from[current]
            if current is None:
                return np.array([0, 0])
        return np.array([current[0] - start[0], current[1] - start[1]])


    @staticmethod
    def plan_action(world: np.ndarray,
                    current: np.ndarray,
                    pursued: np.ndarray,  # Spike
                    pursuer: np.ndarray   # Tom
                    ) -> Optional[np.ndarray]:

        cur = tuple(current)
        spike = tuple(pursued)
        tom = tuple(pursuer)

        # --- Update Histories ---
        PlannerAgent.tom_history.append(tom)
        if len(PlannerAgent.tom_history) > PlannerAgent.history_length:
            PlannerAgent.tom_history.pop(0)

        PlannerAgent.spike_history.append(spike)
        if len(PlannerAgent.spike_history) > PlannerAgent.history_length:
            PlannerAgent.spike_history.pop(0)

        # --- Predict Positions ---
        predicted_spike = PlannerAgent.predict_agent(spike, PlannerAgent.spike_history, tom, world)  # Spike chases Tom
        predicted_tom = PlannerAgent.predict_agent(tom, PlannerAgent.tom_history, cur, world)    # Tom chases Jerry

        # --- Loop Detection (Jerry-Spike) ---
        PlannerAgent.last_positions.append((cur, spike)) # Use original positions, not predictions
        if len(PlannerAgent.last_positions) > 6:
            PlannerAgent.last_positions.pop(0)
        if PlannerAgent.last_positions.count((cur, spike)) >= 3:
            valid_moves = [
                a for a in PlannerAgent.get_all_actions()
                if PlannerAgent.is_valid((cur[0] + a[0], cur[1] + a[1]), world)
            ]
            if valid_moves:
                return valid_moves[np.random.randint(len(valid_moves))]
            return np.array([0, 0])

        # --- Tom Avoidance ---
        dist_to_tom = PlannerAgent.manhattan(cur, predicted_tom) # Use predicted_tom
        avoid_tom = dist_to_tom <= 4

        # --- Trap Detection (Spike) ---
        spike_escape_routes = PlannerAgent.count_escape_routes(predicted_spike, world) # Use predicted_spike
        trap_mode = spike_escape_routes <= 3

        # --- Flanking Bias ---
        vec_spike_to_tom = np.array(predicted_tom) - np.array(predicted_spike)
        vec_jerry_to_spike = np.array(predicted_spike) - np.array(cur)
        flanking_penalty = 0
        if np.dot(vec_spike_to_tom, vec_jerry_to_spike) > 0:
            flanking_penalty = 1.0

        # --- Time-Based Aggression ---
        time_aggression = max(0, len(PlannerAgent.last_positions) - 50)

        # --- Plan Path to Predicted Spike (using A*) ---
        move = PlannerAgent.a_star(cur, predicted_spike, world, predicted_tom, avoid_tom=avoid_tom)

        if move is None:
            return np.array([0, 0])

        return move