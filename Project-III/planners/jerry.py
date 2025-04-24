import numpy as np

# Current Scoring
# jerry=123 * 3 + 102=471 tom=46 * 3 + (275 - 46)=367 spike=40 * 3 + (275 - 40)=355
# jerry wins=123 jerry ties=102 jerry losses=275
# tom wins=46 spike wins=40
# Self kills = 275 - 46 - 40 = 189


# Possible Optimizations

# target trapping:  Add a function that scores positions based on: 
# How many directions Spike can still move 
# Whether Jerry is cutting off those directions

# tie avoidance reward: Add a “tie prediction” heuristic:
# If we’re within 10 steps of max_iter
# And distance to Spike is constant
# And no captures are likely
# Impl: Add a step_count input to plan_action If near timeout, relax crash threshold to allow bolder moves

# Rollout Diversity: Try sampling different actions in the rollout itself (i.e., mini rollouts from the resulting state).

# Stochastic-Aware Greedy Move: When choosing a move toward a target prefer directions that are safe even if rotated
# Make greedy_move() prefer directions with lower crash risk in adjacent cells.

class PlannerAgent:
    def __init__(self, depth=3, rollouts=5):
        self.depth = depth
        self.rollouts = rollouts

        self.actions = [
            np.array([0, 1]),  # right
            np.array([1, 0]),  # down
            np.array([0, -1]), # left
            np.array([-1, 0]), # up
            np.array([0, 0])   # stay
        ]
        self.prob = [0.3, 0.3, 0.4]
        self.last_positions = []

    def plan_action(self, world, current, pursued, pursuer):
        best_score = -np.inf
        best_action = np.array([0, 0])

        self.last_dist_to_target = np.sum(np.abs(current - pursued))
        self.last_dist_from_chaser = np.sum(np.abs(current - pursuer))
        self.last_positions.append(tuple(current))
        if len(self.last_positions) > 15:
            self.last_positions.pop(0)

        for action in self.actions:
            crash_risk = self.expected_crash_risk(world, current, action)
            dist_to_target = np.sum(np.abs(current - pursued))
            safe_moves = self.count_safe_directions(world, current)

            # Dynamic crash threshold
            if dist_to_target <= 2:
                crash_threshold = 0.6
            elif dist_to_target <= 4:
                crash_threshold = 0.4
            else:
                crash_threshold = 0.3

            # Absolute safety checks
            if crash_risk > 0.7:
                continue  # always skip very risky actions
            if safe_moves <= 2 and crash_risk > 0.3:
                continue  # be ultra conservative in tight spaces
            if crash_risk > crash_threshold:
                continue  # dynamic threshold

            total_score = 0.0
            for _ in range(self.rollouts):
                score = self.simulate(world, current, pursued, pursuer, action)
                total_score += score

            avg_score = total_score / self.rollouts

            # Crash penalty scaled by reward potential
            if avg_score > 500:
                avg_score -= 100 * crash_risk
            else:
                avg_score -= 300 * crash_risk

            if avg_score > best_score:
                best_score = avg_score
                best_action = action

        # Final safety check: ensure best action is not likely to crash in any rotation
        if self.expected_crash_risk(world, current, best_action) > 0.7:
            # Try to find a safer alternative (fallback)
            for fallback in self.actions:
                if self.expected_crash_risk(world, current, fallback) < 0.2:
                    return fallback
            return np.array([0, 0])  # worst case: stay put
        else:
            return best_action

    def simulate(self, world, current, pursued, pursuer, base_action):
        pos = np.array(current)
        tar = np.array(pursued)
        chaser = np.array(pursuer)

        for _ in range(self.depth):
            a = self.sample_stochastic_action(base_action)
            next_pos = pos + a

            if not self.valid(world, next_pos):
                return -1000

            pos = next_pos

            if self.near_wall(world, tar):
                tar = tar  # Spike stays still when near wall
            elif np.random.rand() < 0.2:
                tar = self.random_valid_move(world, tar)
            else:
                tar = self.greedy_move(world, tar, pos)

            chaser = self.greedy_move(world, chaser, pos)

            if np.array_equal(pos, tar):
                return 1000
            if np.array_equal(pos, chaser):
                return -1000
            if world[pos[0], pos[1]] == 1:
                return -1000

        return self.evaluate(pos, tar, chaser, world, current)

    def evaluate(self, pos, target, chaser, world, current):
        dist_to_target = np.sum(np.abs(pos - target))
        dist_from_chaser = np.sum(np.abs(pos - chaser))

        score = 0
        score += -1.0 * dist_to_target
        score += 1.5 * dist_from_chaser

        # Progress toward Spike
        progress = self.last_dist_to_target - dist_to_target
        score += 2.0 * progress

        # Escape from Tom
        if dist_from_chaser > self.last_dist_from_chaser:
            score += 2.0  # reward retreat
        self.last_dist_from_chaser = dist_from_chaser

        # Open space
        escape_routes = self.count_safe_directions(world, pos)
        score += escape_routes * 2

        # Danger zone penalty
        score -= self.danger_score(pos, chaser)

        # Bonus for proximity to Spike
        if dist_to_target <= 2:
            # Only give bonus if not adjacent to wall (no easy crash)
            if not self.near_wall(world, pos):
                score += 300
            else:
                score += 100  # smaller bonus — be cautious

        # Wall proximity penalty
        if self.near_wall(world, pos):
            score -= 2

        # Discourage staying still
        if np.array_equal(pos, current):
            score -= 1.5

        # Loop penalty
        if self.last_positions.count(tuple(pos)) > 1:
            score -= 10

        score += np.random.uniform(-0.5, 0.5)

        if dist_to_target <= 2 and self.near_wall(world, pos):
            score -= 50

        return score

    def random_valid_move(self, world, pos):
        np.random.shuffle(self.actions)
        for d in self.actions:
            new = pos + d
            if self.valid(world, new):
                return new
        return pos

    def count_safe_directions(self, world, pos):
        count = 0
        for d in self.actions[:-1]:
            neighbor = pos + d
            if self.valid(world, neighbor):
                count += 1
        return count

    def danger_score(self, pos, chaser):
        dist = np.sum(np.abs(pos - chaser))
        if dist <= 1:
            return 100
        elif dist == 2:
            return 30
        elif dist == 3:
            return 10
        else:
            return 0

    def expected_crash_risk(self, world, current, action):
        variants = [
            (np.array([-action[1], action[0]]), self.prob[0]),  # left
            (action, self.prob[1]),                             # straight
            (np.array([action[1], -action[0]]), self.prob[2])   # right
        ]
        crash_risk = 0.0
        for a, p in variants:
            next_pos = current + a
            if not self.valid(world, next_pos):
                crash_risk += p
        return crash_risk

    def sample_stochastic_action(self, action):
        choice = np.random.choice(3, p=self.prob)
        if choice == 1:
            return action
        elif choice == 0:
            return np.array([-action[1], action[0]])
        else:
            return np.array([action[1], -action[0]])

    def valid(self, world, pos):
        r, c = pos
        rows, cols = world.shape
        return 0 <= r < rows and 0 <= c < cols and world[r, c] != 1

    def near_wall(self, world, pos):
        wall_count = 0
        for d in self.actions[:-1]:
            neighbor = pos + d
            if not self.valid(world, neighbor):
                wall_count += 1
        return wall_count >= 2

    def greedy_move(self, world, source, target):
        best = source
        best_dist = np.sum(np.abs(source - target))
        for d in self.actions:
            new = source + d
            if self.valid(world, new):
                dist = np.sum(np.abs(new - target))
                if dist < best_dist:
                    best = new
                    best_dist = dist
        return best