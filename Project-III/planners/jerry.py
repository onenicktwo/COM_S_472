from heapq import heappush, heappop
from collections import deque
import numpy as np
from typing import Tuple, Optional
from concurrent.futures import ProcessPoolExecutor
import atexit

POOL = ProcessPoolExecutor(max_workers=9)
atexit.register(lambda: POOL.shutdown(wait=False))

# stochastic rotations
P_LEFT, P_STR, P_RIGHT = 0.4, 0.3, 0.3
ROT_P = np.array([P_LEFT, P_STR, P_RIGHT], dtype=np.float32)

# moves (stay, 4-axis, 4-diag)
MOVES = np.array(
    [[ 0, 0],[-1, 0],[ 1, 0],[ 0,-1],[ 0, 1],
     [-1,-1],[-1, 1],[ 1,-1],[ 1, 1]], dtype=np.int8)
N_MOVE = len(MOVES)

# pre-build rotation tables
ROT = np.stack([
    np.stack([[-d[1],  d[0]] for d in MOVES]),   # left
    MOVES.copy(),                                # straight
    np.stack([[ d[1], -d[0]] for d in MOVES])    # right
])

# utility constants
CAPTURE_BONUS =  10.0
CRASH_PENALTY = -12.0
G_DIST        =   0.6

# risk policy: static 12 % threshold
RISK_TH = 0.12

# misc parameters
STALL_LIMIT = 25
HISTORY     = 30

# helpers
def in_bounds(p): return 0 <= p[0] < 30 and 0 <= p[1] < 30
def mhd(a,b):     return abs(int(a[0]-b[0])) + abs(int(a[1]-b[1]))

# worker for the process pool
def ev_worker(args):
    mv, C, S, T, world_flat = args
    world = world_flat.reshape(30, 30)
    mv_idx = np.where((MOVES == mv).all(axis=1))[0][0]

    e_vals = []
    for r in range(3):
        nxt = C + ROT[r, mv_idx]

        # hard outcomes
        if (not in_bounds(nxt)) or world[nxt[0], nxt[1]]:
            e_vals.append(CRASH_PENALTY)
        elif (nxt == T).all():
            e_vals.append(CRASH_PENALTY)
        elif (nxt == S).all():
            e_vals.append(CAPTURE_BONUS)
        else:  # soft heuristic
            e_vals.append(G_DIST * (mhd(nxt, T) - mhd(nxt, S)) / 30.0)

    e_vals = np.asarray(e_vals, np.float32)
    ev  = (ROT_P * e_vals).sum()
    var = (ROT_P * (e_vals - ev) ** 2).sum()
    return mv, ev, var


class PlannerAgent:
    def __init__(self):
        self.safe      = None
        self.sig       = None
        self.open_ast  = []
        self.exit_hist = deque(maxlen=HISTORY)
        self.stall_cnt = 0

    # main API
    def plan_action(
        self,
        world : np.ndarray,
        cur   : Tuple[int,int],
        prey  : Tuple[int,int],
        purs  : Tuple[int,int]) -> Optional[np.ndarray]:

        # rebuild safe table if grid changes
        if self.sig != hash(world.tobytes()):
            self._build_safe_table(world)
            self.exit_hist.clear()
            self.stall_cnt = 0

        C = np.asarray(cur,  np.int8)
        S = np.asarray(prey, np.int8)
        T = np.asarray(purs, np.int8)
        safe_row = self.safe[C[0], C[1]]

        # stall bookkeeping
        exits_now = self._exit_count(world, S, C)
        self.exit_hist.append(exits_now)
        if len(self.exit_hist) == self.exit_hist.maxlen:
            if exits_now >= min(self.exit_hist):
                self.stall_cnt += 1
            else:
                self.stall_cnt = 0

        # 1. flee if pursuer is close
        if mhd(C, T) <= 3:
            return self._flee_move(C, S, T, safe_row)

        # 2. safe-A* step toward prey
        mv_ast = self._astar_first_step(world, C, S)
        if mv_ast is not None and safe_row[self._mv_idx(mv_ast)]:
            return mv_ast

        # 3. choke-move when stalled
        if self.stall_cnt >= STALL_LIMIT:
            mv = self._choke_move(C, S, safe_row, world)
            self.stall_cnt = 0
            return mv

        # 4. expectimax local choice
        return self._expectimax_move(world, C, S, T)

    # tables
    def _build_safe_table(self, world):
        wall  = world == 1
        safe  = np.zeros((30,30,N_MOVE), np.bool_)

        for r in range(30):
            for c in range(30):
                base = np.array([r, c], np.int8)
                trio = base + ROT
                rows, cols = trio[...,0], trio[...,1]

                in_bd  = (rows>=0)&(rows<30)&(cols>=0)&(cols<30)
                hit    = np.logical_not(in_bd) | wall[rows.clip(0,29),
                                                      cols.clip(0,29)]

                risk = (ROT_P[:,None] * hit).sum(axis=0)
                safe[r,c] = risk <= RISK_TH

        self.safe = safe
        self.sig  = hash(world.tobytes())

    # A* (bounds-safe)
    def _astar_first_step(self, world, start, goal):
        if np.array_equal(start, goal):
            return np.array([0, 0], np.int8)

        seen = np.full((30, 30), 99, np.int8)
        self.open_ast.clear()
        heappush(self.open_ast, (mhd(start, goal), 0, tuple(start), -1))

        while self.open_ast:
            f, g, node, first = heappop(self.open_ast)
            if g >= 40: continue
            if seen[node] <= g: continue
            seen[node] = g

            if node == tuple(goal):
                return MOVES[first] if first >= 0 else np.array([0, 0], np.int8)

            r, c = node
            for idx, mv in enumerate(MOVES):
                if not self.safe[r, c, idx]:
                    continue
                nxt = (r + mv[0], c + mv[1])
                if not in_bounds(nxt) or world[nxt]:
                    continue
                if seen[nxt] <= g + 1:
                    continue
                heappush(self.open_ast,
                         (g + 1 + mhd(nxt, goal), g + 1, nxt,
                          idx if first == -1 else first))
        return None

    # local heuristics
    def _flee_move(self, C, S, T, safe_row):
        best_mv, best_val = MOVES[0], -1e9
        for idx, mv in enumerate(MOVES):
            if not safe_row[idx]: continue
            nxt = C + mv
            val = 2 * mhd(nxt, T) - mhd(nxt, S)
            if val > best_val:
                best_val, best_mv = val, mv
        return best_mv

    def _choke_move(self, C, S, safe_row, world):
        base_exits = self._exit_count(world, S, C)
        best_mv, best_val = MOVES[0], base_exits
        for idx, mv in enumerate(MOVES):
            if not safe_row[idx]: continue
            nxt = C + mv
            ex  = self._exit_count(world, S, nxt)
            if ex < best_val or (ex == best_val and mhd(nxt, S) < mhd(best_mv + C, S)):
                best_mv, best_val = mv, ex
        return best_mv

    def _expectimax_move(self, world, C, S, T):
        world_flat = world.ravel()
        futures = [POOL.submit(ev_worker, (mv, C, S, T, world_flat))
                   for mv in MOVES]

        best_mv, best_ev, best_var = MOVES[0], -1e9, 1e9
        for f in futures:
            mv, ev, var = f.result()
            if ev > best_ev + 1e-6 or (abs(ev - best_ev) <= 1e-6 and var < best_var):
                best_mv, best_ev, best_var = mv, ev, var
        return best_mv

    # minor utils
    @staticmethod
    def _exit_count(world, S, pos):
        exits = 0
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = S[0] + dr, S[1] + dc
            if 0 <= nr < 30 and 0 <= nc < 30 and world[nr, nc] == 0 and (nr, nc) != tuple(pos):
                exits += 1
        return exits

    @staticmethod
    def _mv_idx(mv):
        return int(np.where((MOVES == mv).all(axis=1))[0][0])