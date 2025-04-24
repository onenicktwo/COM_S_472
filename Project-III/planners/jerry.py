from heapq import heappush, heappop
from collections import deque
import numpy as np
from typing import Tuple, Optional

P_LEFT, P_STR, P_RIGHT = 0.4, 0.3, 0.3
ROT_P = np.array([P_LEFT, P_STR, P_RIGHT], dtype=np.float32)

MOVES = np.array(
    [[ 0, 0],[-1, 0],[ 1, 0],[ 0,-1],[ 0, 1],
     [-1,-1],[-1, 1],[ 1,-1],[ 1, 1]],
    dtype=np.int8)
N_MOVE = len(MOVES)

ROT = np.stack([
    np.stack([[-d[1],  d[0]] for d in MOVES]),   # left 90°
    MOVES.copy(),                                # straight
    np.stack([[ d[1], -d[0]] for d in MOVES])    # right 90°
])                                               # (3,9,2)

def inb(p): return 0 <= p[0] < 30 and 0 <= p[1] < 30
def mhd(a,b): return abs(int(a[0]-b[0])) + abs(int(a[1]-b[1]))

class PlannerAgent:
    """
    Decision flow
    1.  Build / reuse SAFE[r,c,move] :  True  iff *all three* noisy
        outcomes are obstacle-free and on the board.
    2.  If Tom is within 3 Manhattan →  ‘panic mode’: choose the safe move that
        maximises   2·dist_to_Tom – dist_to_Spike .
    3.  Else compute an A* path to Spike **using only safe moves**.
        If path found  → take its first step.
    4.  If no A* path (Spike boxed)   → doorway-closing heuristic:
        pick safe move that minimises Spike’s 4-neighbour exit count,
        breaking ties with distance heuristic.
    All steps are O(900) or less ⇒  < 1 ms.
    """

    def __init__(self):
        self.safe  = None     # 30x30x9 bool
        self.sig   = None
        self.OPEN  = []       # A* priority queue reused to avoid realloc

    def plan_action(self,
                    world : np.ndarray,
                    cur   : Tuple[int,int],
                    prey  : Tuple[int,int],
                    purs  : Tuple[int,int]) -> Optional[np.ndarray]:

        if self.sig != hash(world.tobytes()):
            self._build_safe_table(world)

        C = np.asarray(cur,   np.int8)
        S = np.asarray(prey,  np.int8)
        T = np.asarray(purs,  np.int8)
        safe_row = self.safe[C[0],C[1]]           # (9,) bool

        # panic mode if Tom is close
        if mhd(C, T) <= 3:
            best, best_val = MOVES[0], -1e9
            for idx, mv in enumerate(MOVES):
                if not safe_row[idx]: continue
                nxt = C + mv
                val = 2*mhd(nxt, T) - mhd(nxt, S)
                if val > best_val:
                    best_val = val; best = mv
            return best

        # A* path (safe moves only)
        path_mv = self._astar_first_step(world, C, S)
        if path_mv is not None and safe_row[self._mv_idx(path_mv)]:
            return path_mv

        # doorway-closing heuristic
        exits0 = self._exit_count(world, S, C)
        best,b_exit,b_score = MOVES[0], 9, -1e9
        for idx, mv in enumerate(MOVES):
            if not safe_row[idx]: continue
            nxt = C + mv
            ex  = self._exit_count(world, S, nxt)
            score = 1.2*mhd(nxt, T) - mhd(nxt, S)
            if ex < b_exit or (ex == b_exit and score > b_score):
                best,b_exit,b_score = mv,ex,score
        return best

    def _build_safe_table(self, world):
        self.safe = np.zeros((30,30,N_MOVE), np.bool_)
        wall = world==1
        for r in range(30):
            for c in range(30):
                base = np.array([r,c], np.int8)
                trio = base + ROT
                rows, cols = trio[...,0], trio[...,1]
                legal = (rows>=0)&(rows<30)&(cols>=0)&(cols<30)
                legal &= ~wall[rows.clip(0,29), cols.clip(0,29)]
                self.safe[r,c] = legal.all(axis=0)
        self.sig = hash(world.tobytes())


    def _astar_first_step(self, world, start, goal):
        if np.array_equal(start, goal): return np.array([0,0], np.int8)
        seen = np.full((30,30), 99, np.int8)
        self.OPEN.clear()
        heappush(self.OPEN, (mhd(start,goal), 0, tuple(start), -1))
        while self.OPEN:
            f,g,node,first_idx = heappop(self.OPEN)
            if g >= 40: continue                      # depth cutoff
            if seen[node] <= g: continue
            seen[node] = g
            if node == tuple(goal):
                return MOVES[first_idx] if first_idx>=0 else np.array([0,0],np.int8)
            r,c = node
            for idx,mv in enumerate(MOVES):
                if not self.safe[r,c,idx]: continue
                nxt = (r+mv[0], c+mv[1])
                if seen[nxt] <= g+1: continue
                h = mhd(nxt, goal)
                heappush(self.OPEN, (g+1+h, g+1, nxt, idx if first_idx==-1 else first_idx))
        return None   # no path

    @staticmethod
    def _exit_count(world, S, J_new):
        exits = 0
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr,nc = S[0]+dr, S[1]+dc
            if 0<=nr<30 and 0<=nc<30 and world[nr,nc]==0 and (nr,nc)!=tuple(J_new):
                exits += 1
        return exits

    @staticmethod
    def _mv_idx(mv): return int(np.where((MOVES==mv).all(axis=1))[0][0])