import numpy as np
import heapq
from typing import Tuple, Optional, List


class PlannerAgent:
    def __init__(self):
        self.pursuer_history = []
        self.target_history = []
        self.last_positions = []
        self.history_length = 3

    def get_all_actions(self) -> List[np.ndarray]:
        return [np.array([dr, dc]) for dr in [-1, 0, 1] for dc in [-1, 0, 1]]

    def is_valid(self, pos: Tuple[int, int], world: np.ndarray) -> bool:
        r, c = pos
        return (
            0 <= r < world.shape[0] and
            0 <= c < world.shape[1] and
            world[r, c] == 0
        )

    def manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def neighbors(self, pos: Tuple[int, int], world: np.ndarray) -> List[Tuple[int, int]]:
        return [
            (pos[0] + move[0], pos[1] + move[1])
            for move in self.get_all_actions()
            if self.is_valid((pos[0] + move[0], pos[1] + move[1]), world)
        ]

    def predict_agent(self, current_pos: Tuple[int, int], history: List[Tuple[int, int]],
                      goal: Tuple[int, int], world: np.ndarray) -> Tuple[int, int]:
        # Greedy 1-step prediction
        best_pos = current_pos
        best_dist = self.manhattan(current_pos, goal)
        for n in self.neighbors(current_pos, world):
            d = self.manhattan(n, goal)
            if d < best_dist:
                best_pos = n
                best_dist = d

        # Velocity-based prediction
        if len(history) >= 2:
            velocity = np.array(history[-1]) - np.array(history[-2])
            predicted = tuple(np.array(current_pos) + velocity)
            if self.is_valid(predicted, world):
                return predicted

        return best_pos

    def count_escape_routes(self, pos: Tuple[int, int], world: np.ndarray) -> int:
        return sum(
            1 for move in self.get_all_actions()
            if np.any(move) and self.is_valid((pos[0] + move[0], pos[1] + move[1]), world)
        )

    def a_star(self,
               start: Tuple[int, int],
               goal: Tuple[int, int],
               world: np.ndarray,
               predicted_pursuer: Tuple[int, int],
               avoid_pursuer: bool = True,
               trap_mode: bool = False,
               flanking_penalty: float = 0.0
               ) -> Optional[np.ndarray]:
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break

            for neighbor in self.neighbors(current, world):
                new_cost = cost_so_far[current] + 1

                if avoid_pursuer and self.manhattan(neighbor, predicted_pursuer) <= 2:
                    new_cost += 5

                escape = self.count_escape_routes(neighbor, world)
                if escape < 3:
                    new_cost += (3 - escape) * 2

                new_cost += flanking_penalty * 3

                if trap_mode:
                    new_cost -= 1

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.manhattan(neighbor, goal)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        if goal not in came_from:
            return np.array([0, 0])  # No path

        current = goal
        while came_from[current] != start:
            current = came_from[current]
            if current is None:
                return np.array([0, 0])
        return np.array([current[0] - start[0], current[1] - start[1]])

    def plan_action(self,
                    world: np.ndarray,
                    current: np.ndarray,
                    pursued: np.ndarray,
                    pursuer: np.ndarray
                    ) -> Optional[np.ndarray]:

        cur = tuple(current)
        target = tuple(pursued)
        purs = tuple(pursuer)

        #  Update Histories 
        self.pursuer_history.append(purs)
        if len(self.pursuer_history) > self.history_length:
            self.pursuer_history.pop(0)

        self.target_history.append(target)
        if len(self.target_history) > self.history_length:
            self.target_history.pop(0)

        #  Predict Pursuer 
        predicted_pursuer = self.predict_agent(purs, self.pursuer_history, cur, world)

        #  Loop Detection 
        self.last_positions.append((cur, target))
        if len(self.last_positions) > 6:
            self.last_positions.pop(0)
        if self.last_positions.count((cur, target)) >= 3:
            valid_moves = [
                a for a in self.get_all_actions()
                if self.is_valid((cur[0] + a[0], cur[1] + a[1]), world)
            ]
            if valid_moves:
                return valid_moves[np.random.randint(len(valid_moves))]
            return np.array([0, 0])

        #  Lookahead Evaluation
        best_score = -float('inf')
        best_move = np.array([0, 0])

        for move in self.get_all_actions():
            if np.all(move == 0):
                continue

            next_pos = tuple(current + move)
            if not self.is_valid(next_pos, world):
                continue

            # Predict where the target might go (greedy toward pursuer)
            predicted_target = target
            best_dist = self.manhattan(target, purs)
            for t_move in self.get_all_actions():
                nt = (target[0] + t_move[0], target[1] + t_move[1])
                if not self.is_valid(nt, world):
                    continue
                d = self.manhattan(nt, purs)
                if d < best_dist:
                    predicted_target = nt
                    best_dist = d

            # Score the move
            dist_to_target = self.manhattan(next_pos, predicted_target)
            dist_to_pursuer = self.manhattan(next_pos, predicted_pursuer)
            escape_score = self.count_escape_routes(next_pos, world)

            trap_penalty = 0
            if escape_score <= 2 and dist_to_pursuer < 4:
                trap_penalty = (3 - escape_score) * 4  # Strong deterrent

            score = (
                10 / (dist_to_target + 1) -
                8 / (dist_to_pursuer + 1) +
                0.4 * escape_score -
                trap_penalty)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move