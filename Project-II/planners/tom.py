# planner_opponent.py
# (Save this as planner_tom.py and planner_spike.py for testing against your Jerry)

import numpy as np
import heapq
import random
from typing import Tuple, Optional, List, Dict

# --- Constants ---
# A* Penalties
PURSUER_PROXIMITY_PENALTY_WEIGHT = 15.0 # Penalty strength near predicted pursuer
PURSUER_COLLISION_PENALTY = 1000.0     # Penalty for direct collision with predicted pursuer
# Distance Thresholds
PURSUER_DANGER_ZONE = 4     # Distance within which pursuer penalty ramps up
# Prediction parameters
PREDICT_AVOID_DIST = 1      # How close opponent avoids their pursuer in prediction
PREDICT_AVOID_PENALTY = 10.0  # Penalty strength for opponent avoidance

class PlannerAgent:
    """
    An improved baseline planner using A* with prediction and evasion.
    Designed to be used for Tom and Spike as opponents.
    """

    @staticmethod
    def get_valid_moves_static(pos: Tuple[int, int], world: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """ Static helper to get valid action vectors and resulting positions. """
        moves = []
        r, c = pos
        rows, cols = world.shape
        is_current_valid = 0 <= r < rows and 0 <= c < cols and world[r, c] == 0

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                next_r, next_c = r + dr, c + dc
                if 0 <= next_r < rows and 0 <= next_c < cols and world[next_r, next_c] == 0:
                    moves.append((np.array([dr, dc], dtype=np.int8), (next_r, next_c)))

        stay_put_action = np.array([0, 0], dtype=np.int8)
        # Add stay_put if current is valid and it wasn't added by the loop
        if is_current_valid and not any(np.array_equal(m[0], stay_put_action) for m in moves):
             if world[r, c] == 0: # Double check current spot is free
                 moves.append((stay_put_action, pos))
        elif is_current_valid and not moves: # Trapped in a valid spot
             moves.append((stay_put_action, pos))

        return moves

    @staticmethod
    def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """ Static Manhattan distance calculation. """
        if a is None or b is None: return 9999
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def predict_move_static(agent_current_pos: Tuple[int, int],
                            agent_target_pos: Tuple[int, int],
                            agent_pursuer_pos: Optional[Tuple[int, int]], # The one chasing *this* agent
                            world: np.ndarray) -> Tuple[int, int]:
        """ Predicts the single next position for an agent based on current info only. """
        possible_moves = PlannerAgent.get_valid_moves_static(agent_current_pos, world)
        if not possible_moves: return agent_current_pos # Cannot move

        best_next_pos = agent_current_pos
        min_score = float('inf')

        # Evaluate staying put first
        score_stay = float(PlannerAgent.manhattan(agent_current_pos, agent_target_pos))
        if agent_pursuer_pos:
            dist_pursuer_stay = PlannerAgent.manhattan(agent_current_pos, agent_pursuer_pos)
            if dist_pursuer_stay <= PREDICT_AVOID_DIST:
                score_stay += PREDICT_AVOID_PENALTY / max(0.1, float(dist_pursuer_stay))
        min_score = score_stay

        # Evaluate other moves
        for _, next_pos in possible_moves:
            if next_pos == agent_current_pos: continue

            score_move = float(PlannerAgent.manhattan(next_pos, agent_target_pos))
            if agent_pursuer_pos:
                dist_pursuer_move = PlannerAgent.manhattan(next_pos, agent_pursuer_pos)
                if dist_pursuer_move <= PREDICT_AVOID_DIST:
                    score_move += PREDICT_AVOID_PENALTY / max(0.1, float(dist_pursuer_move))

            if score_move < min_score:
                min_score = score_move
                best_next_pos = next_pos

        return best_next_pos

    @staticmethod
    def a_star_static(start: Tuple[int, int],
                      goal: Tuple[int, int],
                      world: np.ndarray,
                      predicted_pursuer_next_pos: Tuple[int, int], # Agent's own predicted pursuer
                      current_pursuer_pos: Tuple[int, int] # Pass current pursuer pos for heuristic? No, keep heuristic simple.
                     ) -> Optional[np.ndarray]:
        """ A* search implementation modified to avoid the predicted pursuer position. """

        open_set = []
        start_h = PlannerAgent.manhattan(start, goal)
        heapq.heappush(open_set, (start_h, 0, start)) # f = g + h

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {start: None}
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        closed_set = set()

        max_iter = world.shape[0] * world.shape[1] * 1.5 # Allow slightly more iterations
        iter_count = 0

        while open_set and iter_count < max_iter:
            iter_count += 1

            current_f, current_g, current_pos = heapq.heappop(open_set)

            if current_pos in closed_set: continue
            closed_set.add(current_pos)

            # Goal Condition: Reached adjacent (or on top) of the target's position
            if PlannerAgent.manhattan(current_pos, goal) <= 1: # Allow being adjacent as goal
                path = []
                temp = current_pos
                while temp is not None and temp != start:
                    path.append(temp)
                    temp = came_from.get(temp, None)
                    if temp is None and current_pos != start : return None # Path reconstruction failed

                if not path and current_pos == start: # Started adjacent to goal
                     # Find the move that gets directly to goal if valid
                     action_to_goal = np.clip(np.array(goal) - np.array(start), -1, 1).astype(np.int8)
                     if PlannerAgent.manhattan(tuple(np.array(start)+action_to_goal), goal) < PlannerAgent.manhattan(start, goal):
                          # Check if this action is actually valid
                          if any(np.array_equal(action_to_goal, m[0]) for m in PlannerAgent.get_valid_moves_static(start, world)):
                               return action_to_goal
                     # If direct move not best/valid, maybe stay put or A* found other adjacent
                     return np.array([0,0], dtype=np.int8) # Prefer stay if already adjacent

                if not path: return None # Should not happen if loop finished

                first_step_pos = path[-1]
                action = np.array(first_step_pos) - np.array(start)
                return action.astype(np.int8)

            # --- Explore Neighbors ---
            possible_moves = PlannerAgent.get_valid_moves_static(current_pos, world)
            for _, neighbor_pos in possible_moves:
                if neighbor_pos in closed_set: continue

                # --- Calculate Cost (g) ---
                cost_to_move = 1.0 # Base cost

                # Penalty for proximity to predicted pursuer
                dist_to_pursuer = PlannerAgent.manhattan(neighbor_pos, predicted_pursuer_next_pos)
                if dist_to_pursuer <= PURSUER_DANGER_ZONE:
                    cost_to_move += PURSUER_PROXIMITY_PENALTY_WEIGHT / max(0.1, float(dist_to_pursuer))

                # Huge penalty if moving onto the exact square pursuer is predicted to occupy
                if neighbor_pos == predicted_pursuer_next_pos:
                    cost_to_move += PURSUER_COLLISION_PENALTY

                tentative_g_score = current_g + cost_to_move

                if tentative_g_score < g_score.get(neighbor_pos, float('inf')):
                    came_from[neighbor_pos] = current_pos
                    g_score[neighbor_pos] = tentative_g_score
                    h_score = PlannerAgent.manhattan(neighbor_pos, goal) # Heuristic: distance to final target
                    f_score = tentative_g_score + h_score
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor_pos))

        return None # No path found

    # The required interface method
    @staticmethod
    def plan_action(
        world: np.ndarray,
        current: Tuple[int, int], # This agent's position (e.g., Tom)
        pursued: Tuple[int, int], # The agent being chased (e.g., Jerry)
        pursuer: Tuple[int, int]  # The agent chasing this one (e.g., Spike)
    ) -> Optional[np.ndarray]:
        """
        Plans action for Tom/Spike using A* with prediction and evasion.
        Pursues 'pursued', avoids 'pursuer'.
        """
        agent_pos = tuple(current)
        target_pos = tuple(pursued)
        pursuer_pos = tuple(pursuer) # This agent's own pursuer

        # --- Predict Pursuer's Move ---
        # Predict where *our* pursuer will move.
        # Pursuer targets us (agent_pos), and avoids *their* own pursuer (which is our target_pos).
        predicted_pursuer_next_pos = PlannerAgent.predict_move_static(
            pursuer_pos, agent_pos, target_pos, world)

        # --- Immediate Action Checks ---
        possible_agent_moves = PlannerAgent.get_valid_moves_static(agent_pos, world)

        # Check for immediate capture opportunity
        for action, next_agent_pos in possible_agent_moves:
            if next_agent_pos == target_pos: # Potential capture move for *us*
                 # Is it safe from *our* pursuer? Check predicted pursuer pos.
                 if next_agent_pos != predicted_pursuer_next_pos:
                      # print(f"Opponent {agent_pos}: Immediate safe capture.")
                      return action # Take the capture

        # Check for immediate danger (predicted pursuer capture)
        must_escape = False
        if predicted_pursuer_next_pos == agent_pos: # Pursuer predicted to capture us if we stay put
             must_escape = True
             # print(f"Opponent {agent_pos}: Must escape predicted capture.")

        best_escape_action = None
        max_escape_dist = -1
        if must_escape:
            for action, next_agent_pos in possible_agent_moves:
                 if next_agent_pos == agent_pos: continue # Cannot stay put

                 # Check if this move avoids immediate capture by predicted pursuer
                 if next_agent_pos != predicted_pursuer_next_pos:
                      dist_after_escape = PlannerAgent.manhattan(next_agent_pos, predicted_pursuer_next_pos)
                      # Also consider distance to target as tie-breaker for escape
                      dist_to_target = PlannerAgent.manhattan(next_agent_pos, target_pos)

                      # Prioritize maximizing distance, then minimizing target dist
                      current_escape_score = (dist_after_escape, -dist_to_target)
                      best_escape_score = (max_escape_dist, -PlannerAgent.manhattan(tuple(np.array(agent_pos)+(best_escape_action if best_escape_action is not None else [0,0])), target_pos) if best_escape_action is not None else (-1,-9999) )


                      if current_escape_score > best_escape_score:
                           max_escape_dist = dist_after_escape
                           best_escape_action = action

            if best_escape_action is not None:
                 # print(f"Opponent {agent_pos}: Escaping with {best_escape_action}")
                 return best_escape_action
            else:
                 # print(f"Opponent {agent_pos}: Trapped! Cannot escape predicted capture.")
                 pass # Proceed to A*/Fallback

        # --- A* Path Planning ---
        # Goal for A* is the current position of the pursued agent
        astar_goal = target_pos
        astar_action = PlannerAgent.a_star_static(agent_pos, astar_goal, world,
                                                  predicted_pursuer_next_pos, pursuer_pos)

        if astar_action is not None:
             planned_next_pos = tuple(np.array(agent_pos) + astar_action)
             # Final safety check: Don't move into predicted pursuer pos even if A* suggests it
             if planned_next_pos == predicted_pursuer_next_pos:
                  # print(f"Opponent {agent_pos}: A* suggested unsafe move into {predicted_pursuer_next_pos}, fallback.")
                  astar_action = None # Force fallback
             else:
                  # print(f"Opponent {agent_pos}: A* action {astar_action}")
                  return astar_action

        # --- Fallback Strategy ---
        # Trigger fallback if: A* failed OR A* suggested unsafe move
        # print(f"Opponent {agent_pos}: Using Fallback.")

        best_fallback_action = np.array([0, 0], dtype=np.int8) # Default stay put
        # Use predicted pursuer pos for fallback evaluation
        eval_pursuer_pos = predicted_pursuer_next_pos
        max_dist_from_pursuer = PlannerAgent.manhattan(agent_pos, eval_pursuer_pos)
        min_dist_to_target = PlannerAgent.manhattan(agent_pos, target_pos)

        fallback_options = []
        for action, next_pos in possible_agent_moves:
             # Avoid moving directly into predicted pursuer's square
             if next_pos == eval_pursuer_pos: continue

             dist_pursuer = PlannerAgent.manhattan(next_pos, eval_pursuer_pos)
             dist_target = PlannerAgent.manhattan(next_pos, target_pos)
             fallback_options.append({'action': action, 'dist_pursuer': dist_pursuer, 'dist_target': dist_target})

        if not fallback_options:
             # print(f"Opponent {agent_pos}: Fallback - No safe moves.")
             return best_fallback_action # Stay put

        # Priority 1: Maximize distance from predicted pursuer
        escape_moves = [opt for opt in fallback_options if opt['dist_pursuer'] > max_dist_from_pursuer]
        if escape_moves:
            escape_moves.sort(key=lambda x: x['dist_target']) # Tie-break: closest to target
            best_fallback_action = escape_moves[0]['action']
            # print(f"Opponent {agent_pos}: Fallback Escape: {best_fallback_action}")
            return best_fallback_action

        # Priority 2: Maintain distance from pursuer, minimize distance to target
        neutral_moves = [opt for opt in fallback_options if opt['dist_pursuer'] >= max_dist_from_pursuer]
        if neutral_moves:
            neutral_moves.sort(key=lambda x: x['dist_target'])
            best_fallback_action = neutral_moves[0]['action']
            # print(f"Opponent {agent_pos}: Fallback Neutral: {best_fallback_action}")
            return best_fallback_action

        # Priority 3: If all safe moves decrease distance to pursuer, minimize distance to target
        fallback_options.sort(key=lambda x: x['dist_target'])
        best_fallback_action = fallback_options[0]['action']
        # print(f"Opponent {agent_pos}: Fallback Desperate: {best_fallback_action}")
        return best_fallback_action