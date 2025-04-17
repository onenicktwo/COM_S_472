import numpy as np
import heapq
from typing import Tuple, Optional, List


class PlannerAgent:
    def __init__(self):
        self.spike_history = []
        self.pursuer_history = []
        self.history_length = 3

    def get_all_actions(self) -> List[np.ndarray]:
        return [np.array([dr, dc]) for dr in [-1, 0, 1] for dc in [-1, 0, 1]]

    def is_valid(self, pos: Tuple[int, int], world: np.ndarray) -> bool:
        r, c = pos
        return 0 <= r < world.shape[0] and 0 <= c < world.shape[1] and world[r, c] == 0

    def manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def neighbors(self, pos: Tuple[int, int], world: np.ndarray) -> List[Tuple[int, int]]:
        return [
            (pos[0] + move[0], pos[1] + move[1])
            for move in self.get_all_actions()
            if self.is_valid((pos[0] + move[0], pos[1] + move[1]), world)
        ]

    def a_star(self, start: Tuple[int, int], goal: Tuple[int, int], world: np.ndarray,
               avoid: Tuple[int, int]) -> Optional[np.ndarray]:
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break

            for next_pos in self.neighbors(current, world):
                new_cost = cost_so_far[current] + 1
                # Light penalty for being near Tom
                if self.manhattan(next_pos, avoid) < 2:
                    new_cost += 4

                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.manhattan(next_pos, goal)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current

        if goal not in came_from:
            return None

        cur = goal
        while came_from[cur] != start:
            if came_from[cur] is None:
                return None
            cur = came_from[cur]
        return np.array(cur) - np.array(start)

    def predict_spike(self, spike: Tuple[int, int], jerry: Tuple[int, int], world: np.ndarray) -> List[Tuple[int, int]]:
        path = []
        cur = spike
        for _ in range(3):
            best = cur
            best_dist = self.manhattan(cur, jerry)
            for move in self.get_all_actions():
                if np.all(move == 0):
                    continue
                nxt = (cur[0] + move[0], cur[1] + move[1])
                if self.is_valid(nxt, world):
                    d = self.manhattan(nxt, jerry)
                    if d > best_dist:
                        best = nxt
                        best_dist = d
            cur = best
            path.append(cur)
        return path

    def escape_options(self, pos: Tuple[int, int], world: np.ndarray) -> int:
        return sum(1 for move in self.get_all_actions()
                   if np.any(move) and self.is_valid((pos[0] + move[0], pos[1] + move[1]), world))

    def plan_action(
        self,
        world: np.ndarray,
        current: np.ndarray,
        pursued: np.ndarray,
        pursuer: np.ndarray
    ) -> Optional[np.ndarray]:

        jerry = tuple(current)
        spike = tuple(pursued)
        tom = tuple(pursuer)

        self.spike_history.append(spike)
        if len(self.spike_history) > self.history_length:
            self.spike_history.pop(0)

        self.pursuer_history.append(tom)
        if len(self.pursuer_history) > self.history_length:
            self.pursuer_history.pop(0)

        # Immediate capture
        if self.manhattan(jerry, spike) == 1:
            return np.array(spike) - np.array(jerry)

        # Commit if Spike is cornered
        if self.escape_options(spike, world) <= 3 and self.manhattan(jerry, spike) <= 3:
            move = self.a_star(jerry, spike, world, tom)
            if move is not None:
                return move

        # Predict Spike movements and intercept
        predicted_path = self.predict_spike(spike, jerry, world)
        for predicted in predicted_path:
            move = self.a_star(jerry, predicted, world, tom)
            if move is not None:
                return move

        # Fallback: chase Spike directly
        move = self.a_star(jerry, spike, world, tom)
        if move is not None:
            return move

        # Final fallback: greedy
        best_move = np.array([0, 0])
        best_dist = self.manhattan(jerry, spike)
        for move in self.get_all_actions():
            if np.all(move == 0):
                continue
            pos = (jerry[0] + move[0], jerry[1] + move[1])
            if self.is_valid(pos, world):
                d = self.manhattan(pos, spike)
                if d < best_dist:
                    best_dist = d
                    best_move = move

        return best_move