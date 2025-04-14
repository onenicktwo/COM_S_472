import numpy as np
from typing import Tuple, Optional, List, Dict

class PlannerAgent:
    """
    A competitor agent that uses a reactive, 1-step lookahead heuristic
    to decide its action, balancing pursuit and evasion based on immediate scores.
    Based on the original logic in the user's jerry.py plan_action.
    """
    def __init__(self, history_length: int = 5):
        """
        Initializes the planner with instance-specific history.
        """
        self.pursuer_history: List[Tuple[int, int]] = []
        self.target_history: List[Tuple[int, int]] = []
        self.self_history: List[Tuple[int, int]] = [] # Track own positions for loop detection
        self.history_length: int = history_length

    # --- Static Helper Methods ---

    @staticmethod
    def get_all_actions() -> List[np.ndarray]:
        """Returns all possible move actions, including staying put."""
        return [np.array([dr, dc]) for dr in [-1, 0, 1] for dc in [-1, 0, 1]]

    @staticmethod
    def is_valid(pos: Tuple[int, int], world: np.ndarray) -> bool:
        """Checks if a position is within bounds and not an obstacle."""
        r, c = pos
        rows, cols = world.shape
        return 0 <= r < rows and 0 <= c < cols and world[r, c] == 0

    @staticmethod
    def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculates Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # --- Instance Helper Methods ---

    def predict_agent_position(self, current_pos: Tuple[int, int], history: List[Tuple[int, int]],
                           target_for_prediction: Tuple[int, int], world: np.ndarray) -> Tuple[int, int]:
        """
        Predicts the next position of an agent based on history or greedy movement.
        (Same prediction logic as the A* agent for consistency in comparison)
        """
        # --- Velocity-based prediction (if enough history) ---
        if len(history) >= 2:
            velocity = np.array(history[-1]) - np.array(history[-2])
            predicted_pos = tuple((np.array(current_pos) + velocity).astype(int))
            if self.is_valid(predicted_pos, world):
                return predicted_pos

        # --- Fallback: Greedy prediction (1-step towards target_for_prediction) ---
        best_pos = current_pos
        min_dist = self.manhattan(current_pos, target_for_prediction)
        for move in self.get_all_actions():
            next_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
            if self.is_valid(next_pos, world):
                dist = self.manhattan(next_pos, target_for_prediction)
                if dist < min_dist:
                    min_dist = dist
                    best_pos = next_pos
        return best_pos

    def count_escape_routes(self, pos: Tuple[int, int], world: np.ndarray) -> int:
        """Counts the number of valid moves from a position (excluding staying put)."""
        count = 0
        for move in self.get_all_actions():
            if move[0] != 0 or move[1] != 0:
                next_pos = (pos[0] + move[0], pos[1] + move[1])
                if self.is_valid(next_pos, world):
                    count += 1
        return count

    # --- Main Planning Method ---

    def plan_action(self, world: np.ndarray,
                      current: np.ndarray,
                      pursued: np.ndarray,  # target
                      pursuer: np.ndarray   # pursuer
                      ) -> Optional[np.ndarray]:
        """
        Determines the agent's next move using a 1-step lookahead heuristic score.

        Args:
            world: 2D numpy array representing the environment.
            current: (row, col) position of this agent.
            pursued: (row, col) position of the target agent.
            pursuer: (row, col) position of the agent chasing this agent.

        Returns:
            A numpy array [dr, dc] representing the direction of movement.
        """
        cur_pos = tuple(current)
        target_pos = tuple(pursued)
        pursuer_pos = tuple(pursuer)

        # --- Update Histories ---
        self.pursuer_history.append(pursuer_pos)
        if len(self.pursuer_history) > self.history_length:
            self.pursuer_history.pop(0)

        self.target_history.append(target_pos)
        if len(self.target_history) > self.history_length:
            self.target_history.pop(0)

        self.self_history.append(cur_pos)
        if len(self.self_history) > self.history_length + 2:
             self.self_history.pop(0)

        # --- Predict Positions ---
        # Predict pursuer's next move (likely towards us)
        predicted_pursuer = self.predict_agent_position(pursuer_pos, self.pursuer_history, cur_pos, world)
        # Predict target's next move (could be complex, let's predict towards its target - our pursuer)
        predicted_target = self.predict_agent_position(target_pos, self.target_history, pursuer_pos, world)

        # --- Loop Detection ---
        loop_threshold = 3
        recent_history = self.self_history[-(loop_threshold * 2):]
        if recent_history.count(cur_pos) >= loop_threshold:
            # If looping, try a random valid move away from the predicted pursuer
            valid_moves = []
            for move in self.get_all_actions():
                if np.any(move): # Exclude [0, 0]
                    next_pos_loop = tuple(current + move)
                    if self.is_valid(next_pos_loop, world) and self.manhattan(next_pos_loop, predicted_pursuer) > 1:
                         valid_moves.append(move)
            if valid_moves:
                idx = np.random.choice(len(valid_moves))
                # print(f"Competitor Loop detected at {cur_pos}! Making random safe move: {valid_moves[idx]}") # Debug print
                return valid_moves[idx]
            else:
                 # print(f"Competitor Loop detected at {cur_pos} but no safe random move! Staying put.") # Debug print
                 return np.array([0, 0]) # Stay put if no safe random move

        # --- 1-Step Lookahead Evaluation (Original Logic) ---
        best_score = -float('inf')
        best_move = np.array([0, 0]) # Default to staying put

        # Evaluate each possible action
        for move in self.get_all_actions():
            next_pos = tuple(current + move)

            # Skip invalid moves
            if not self.is_valid(next_pos, world):
                continue

            # Score the potential next state based on distances and escape routes
            dist_to_target = self.manhattan(next_pos, predicted_target) # Use predicted target
            dist_to_pursuer = self.manhattan(next_pos, predicted_pursuer) # Use predicted pursuer
            escape_score = self.count_escape_routes(next_pos, world)

            # Heuristic score calculation (similar to original jerry.py)
            score = 0.0
            # Reward for getting closer to the target (higher score for smaller distance)
            # Add small epsilon to avoid division by zero
            score += 5.0 / (dist_to_target + 1e-6)
            # Penalty for getting closer to the pursuer (lower score for smaller distance)
            score -= 5.0 / (dist_to_pursuer + 1e-6)
            # Reward for having more escape routes (flexibility)
            score += 0.5 * escape_score

            # If this move yields a better score, update the best move
            if score > best_score:
                best_score = score
                best_move = move

        return best_move
