import numpy as np
import heapq
from typing import Tuple, Optional, List

class PlannerAgent:
    pursuer_history = []
    target_history = []
    last_positions = []
    history_length = 3

    @staticmethod
    def get_all_actions() -> List[np.ndarray]:
        return [np.array([dr, dc]) for dr in [-1, 0, 1] for dc in [-1, 0, 1]]

    @staticmethod
    def is_valid(pos: Tuple[int, int], world: np.ndarray) -> bool:
        r, c = pos
        return (
            0 <= r < world.shape[0] and
            0 <= c < world.shape[1] and
            world[r, c] == 0
        )

    @staticmethod
    def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def neighbors(pos: Tuple[int, int], world: np.ndarray) -> List[Tuple[int, int]]:
        return [
            (pos[0] + move[0], pos[1] + move[1])
            for move in PlannerAgent.get_all_actions()
            if PlannerAgent.is_valid((pos[0] + move[0], pos[1] + move[1]), world)
        ]

    @staticmethod
    def predict_agent(current_pos: Tuple[int, int], history: List[Tuple[int, int]],
                      target: Tuple[int, int], world: np.ndarray) -> Tuple[int, int]:
        # Greedy prediction (1-step toward target)
        best_pos = current_pos
        best_dist = PlannerAgent.manhattan(current_pos, target)
        for n in PlannerAgent.neighbors(current_pos, world):
            d = PlannerAgent.manhattan(n, target)
            if d < best_dist:
                best_pos = n
                best_dist = d

        # Velocity-based prediction (if enough history)
        if len(history) >= 2:
            velocity = np.array(history[-1]) - np.array(history[-2])
            predicted = tuple((np.array(current_pos) + velocity).astype(int))
            if PlannerAgent.is_valid(predicted, world):
                return predicted

        return best_pos

    @staticmethod
    def count_escape_routes(pos: Tuple[int, int], world: np.ndarray) -> int:
        return sum(
            1 for move in PlannerAgent.get_all_actions()
            if np.any(move) and PlannerAgent.is_valid((pos[0] + move[0], pos[1] + move[1]), world)
        )

    @staticmethod
    def a_star(start: Tuple[int, int], goal: Tuple[int, int], world: np.ndarray,
               predicted_pursuer: Tuple[int, int], avoid_pursuer: bool = True,
               trap_mode: bool = False, flanking_penalty: float = 0.0) -> Optional[np.ndarray]:
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

                # pursuer avoidance
                if avoid_pursuer and PlannerAgent.manhattan(neighbor, predicted_pursuer) <= 2:
                    new_cost += 5

                # Penalize low escape options (dead ends)
                escape = PlannerAgent.count_escape_routes(neighbor, world)
                if escape < 3:
                    new_cost += (3 - escape) * 2

                # Flanking penalty (if we're between pursuer and target)
                new_cost += flanking_penalty * 3

                # Encourage aggression if target is trapped
                if trap_mode:
                    new_cost -= 1

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + PlannerAgent.manhattan(neighbor, goal)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        if goal not in came_from:
            return np.array([0, 0])  # No path

        # Reconstruct path
        current = goal
        while came_from[current] != start:
            current = came_from[current]
            if current is None:
                return np.array([0, 0])
        return np.array([current[0] - start[0], current[1] - start[1]])

    @staticmethod
    def plan_action(world: np.ndarray,
                    current: np.ndarray,
                    pursued: np.ndarray,  # target
                    pursuer: np.ndarray   # pursuer
                    ) -> Optional[np.ndarray]:

        cur = tuple(current)
        target = tuple(pursued)
        pursuer = tuple(pursuer)

        # --- Update Histories ---
        PlannerAgent.pursuer_history.append(pursuer)
        if len(PlannerAgent.pursuer_history) > PlannerAgent.history_length:
            PlannerAgent.pursuer_history.pop(0)

        PlannerAgent.target_history.append(target)
        if len(PlannerAgent.target_history) > PlannerAgent.history_length:
            PlannerAgent.target_history.pop(0)

        # --- Predict Positions ---
        predicted_pursuer = PlannerAgent.predict_agent(pursuer, PlannerAgent.pursuer_history, cur, world)

        # --- Loop Detection ---
        PlannerAgent.last_positions.append((cur, target))
        if len(PlannerAgent.last_positions) > 6:
            PlannerAgent.last_positions.pop(0)
        if PlannerAgent.last_positions.count((cur, target)) >= 3:
            valid_moves = [
                a for a in PlannerAgent.get_all_actions()
                if PlannerAgent.is_valid((cur[0] + a[0], cur[1] + a[1]), world)
            ]
            if valid_moves:
                return valid_moves[np.random.randint(len(valid_moves))]
            return np.array([0, 0])

        # --- 2-Step Lookahead Evaluation ---
        best_score = -float('inf')
        best_move = np.array([0, 0])

        for move in PlannerAgent.get_all_actions():
            next = tuple(current + move)
            if not PlannerAgent.is_valid(next, world):
                continue

            # Predict target's move: greedy toward pursuer
            best_target = target
            best_dist = PlannerAgent.manhattan(target, pursuer)
            for smove in PlannerAgent.get_all_actions():
                next_target = (target[0] + smove[0], target[1] + smove[1])
                if not PlannerAgent.is_valid(next_target, world):
                    continue
                d = PlannerAgent.manhattan(next_target, pursuer)
                if d < best_dist:
                    best_target = next_target
                    best_dist = d

            # Score the resulting state
            dist_to_target = PlannerAgent.manhattan(next, best_target)
            dist_to_pursuer = PlannerAgent.manhattan(next, predicted_pursuer)

            escape_score = PlannerAgent.count_escape_routes(next, world)

            # Heuristic: prioritize closing in on target, but stay away from pursuer
            score = 0
            score += 5 / (dist_to_target + 1)       # reward chasing target
            score -= 5 / (dist_to_pursuer + 1)         # penalize being near pursuer
            score += 0.5 * escape_score            # reward flexibility

            if score > best_score:
                best_score = score
                best_move = move

        return best_move