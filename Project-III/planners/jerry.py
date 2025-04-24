from heapq import heappush, heappop
from collections import deque
import numpy as np
from typing import Tuple, Optional

# stochastic model
P_LEFT, P_STR, P_RIGHT = 0.4, 0.3, 0.3
ROT_P = np.array([P_LEFT, P_STR, P_RIGHT], dtype=np.float32)

# move set
MOVES = np.array(
    [[ 0, 0],[-1, 0],[ 1, 0],[ 0,-1],[ 0, 1],
     [-1,-1],[-1, 1],[ 1,-1],[ 1, 1]], dtype=np.int8)
N_MOVE = len(MOVES)

ROT = np.stack([
    np.stack([[-d[1],  d[0]] for d in MOVES]),
    MOVES.copy(),
    np.stack([[ d[1], -d[0]] for d in MOVES])
])

def in_bounds(p): return 0 <= p[0] < 30 and 0 <= p[1] < 30
def mhd(a,b):     return abs(int(a[0]-b[0])) + abs(int(a[1]-b[1]))

# stall parameters
STALL_LIMIT   = 25      # frames without exit reduction -> stall
HISTORY       = 30      # keep this many past exit counts

class PlannerAgent:
    def __init__(self):
        self.safe      = None
        self.sig       = None
        self.open_ast  = []
        # pocket stall bookkeeping
        self.exit_hist = deque(maxlen=HISTORY)
        self.stall_cnt = 0

    def plan_action(self,
                    world : np.ndarray,
                    cur   : Tuple[int,int],
                    prey  : Tuple[int,int],
                    purs  : Tuple[int,int]) -> Optional[np.ndarray]:

        # build / reuse safe table
        if self.sig != hash(world.tobytes()):
            self._build_safe_table(world)
            self.exit_hist.clear(); self.stall_cnt = 0

        C = np.asarray(cur,   np.int8)
        S = np.asarray(prey,  np.int8)
        T = np.asarray(purs,  np.int8)
        safe_row = self.safe[C[0], C[1]]

        # exit statistics for stall detection
        exits_now = self._exit_count(world, S, C)
        self.exit_hist.append(exits_now)
        if len(self.exit_hist) == self.exit_hist.maxlen:
            if exits_now >= min(self.exit_hist):
                self.stall_cnt += 1
            else:
                self.stall_cnt = 0   # progress achieved
        
        #  flee Tom if too near
        if mhd(C, T) <= 3:
            return self._flee_move(world, C, S, T, safe_row)

        # A* to Spike (safe moves only)
        mv_ast = self._astar_first_step(world, C, S)
        if mv_ast is not None and safe_row[self._mv_idx(mv_ast)]:
            return mv_ast

        # stall-breaker choke move 
        if self.stall_cnt >= STALL_LIMIT:
            mv = self._choke_move(world, C, S, safe_row)
            self.stall_cnt = 0        # reset counter after forcing move
            return mv

        # doorway heuristic
        return self._doorway_move(world, C, S, T, safe_row)

    def _build_safe_table(self, world):
        self.safe = np.zeros((30,30,N_MOVE), np.bool_)
        wall = world == 1
        for r in range(30):
            for c in range(30):
                base = np.array([r,c], np.int8)
                trio = base + ROT
                rows, cols = trio[...,0], trio[...,1]
                legal = (rows>=0)&(rows<30)&(cols>=0)&(cols<30)
                legal &= ~wall[rows.clip(0,29), cols.clip(0,29)]
                self.safe[r,c] = legal.all(axis=0)
        self.sig = hash(world.tobytes())

    # flee when Tom close 
    def _flee_move(self, world, C, S, T, safe_row):
        best_mv, best_val = MOVES[0], -1e9
        for idx,mv in enumerate(MOVES):
            if not safe_row[idx]: continue
            nxt = C + mv
            val = 2*mhd(nxt, T) - mhd(nxt, S)
            if val > best_val:
                best_val, best_mv = val, mv
        return best_mv

    # A* path using only safe moves 
    def _astar_first_step(self, world, start, goal):
        if np.array_equal(start, goal): return np.array([0,0], np.int8)
        seen = np.full((30,30), 99, np.int8)
        self.open_ast.clear()
        heappush(self.open_ast, (mhd(start,goal), 0, tuple(start), -1))
        while self.open_ast:
            f,g,node,first = heappop(self.open_ast)
            if g >= 40:           continue
            if seen[node] <= g:   continue
            seen[node] = g
            if node == tuple(goal):
                return MOVES[first] if first>=0 else np.array([0,0],np.int8)
            r,c = node
            for idx,mv in enumerate(MOVES):
                if not self.safe[r,c,idx]: continue
                nxt = (r+mv[0], c+mv[1])
                if seen[nxt] <= g+1: continue
                heappush(self.open_ast,(g+1+mhd(nxt,goal), g+1,
                                         nxt, idx if first==-1 else first))
        return None

    #  doorway heuristic when not stalled 
    def _doorway_move(self, world, C, S, T, safe_row):
        best,b_exit,b_score = MOVES[0], 9, -1e9
        for idx,mv in enumerate(MOVES):
            if not safe_row[idx]: continue
            nxt = C + mv
            ex  = self._exit_count(world, S, nxt)
            score = 1.2*mhd(nxt, T) - mhd(nxt, S)
            if ex < b_exit or (ex==b_exit and score>b_score):
                best,b_exit,b_score = mv,ex,score
        return best

    # STALL-BREAKER choke move
    def _choke_move(self, world, C, S, safe_row):
        base_exits = self._exit_count(world, S, C)
        best_mv, best_val = MOVES[0], base_exits
        for idx,mv in enumerate(MOVES):
            if not safe_row[idx]: continue
            nxt = C + mv
            ex  = self._exit_count(world, S, nxt)
            if ex < best_val or (ex==best_val and mhd(nxt,S)<mhd(best_mv+C,S)):
                best_mv, best_val = mv, ex
        return best_mv

    # Spike exits (orthogonal)
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