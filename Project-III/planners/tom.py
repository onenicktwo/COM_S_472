import numpy as np

# Define all 4 movements (up, down, left, right) + staying still
ACTIONS = [np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0]), np.array([0, 0])]

# Probabilities for left, correct, right
PROBS = [0.3, 0.3, 0.4]

class PlannerAgent:
    def __init__(self):
        pass

    def rotate_left(self, action):
        return np.array([-action[1], action[0]])

    def rotate_right(self, action):
        return np.array([action[1], -action[0]])

    def in_bounds(self, pos, world):
        return 0 <= pos[0] < world.shape[0] and 0 <= pos[1] < world.shape[1]

    def is_free(self, pos, world):
        return self.in_bounds(pos, world) and world[pos[0], pos[1]] == 0

    def eval_move(self, move, current, pursued, pursuer, world):
        """Returns a reward score for a given move after accounting for stochasticity"""
        # Compute 3 possible outcomes after rotation
        left = self.rotate_left(move)
        right = self.rotate_right(move)
        outcomes = [left, move, right]

        value = 0
        for i, a in enumerate(outcomes):
            new_pos = current + a
            if not self.is_free(new_pos, world):
                penalty = -100  # hitting wall
                value += PROBS[i] * penalty
                continue

            dist_to_target = np.linalg.norm(new_pos - pursued, ord=1)
            dist_from_pursuer = np.linalg.norm(new_pos - pursuer, ord=1)

            # Reward being closer to pursued, and farther from pursuer
            score = -dist_to_target + 0.5 * dist_from_pursuer
            value += PROBS[i] * score

        return value

    def plan_action(self, world, current, pursued, pursuer):
        best_action = np.array([0, 0])
        best_value = float('-inf')

        for move in ACTIONS:
            val = self.eval_move(move, current, pursued, pursuer, world)
            if val > best_value:
                best_value = val
                best_action = move

        return best_action