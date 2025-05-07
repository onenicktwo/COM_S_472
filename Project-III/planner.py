import numpy as np
from heapq import heappush, heappop
from collections import deque
from typing import Tuple, Optional

# environment probabilities
P_LEFT, P_STR, P_RIGHT = 0.4, 0.3, 0.3
ROT_P = np.array([P_LEFT, P_STR, P_RIGHT], dtype=np.float32)

# motion tables
MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0,-1], [0, 1],
                  [-1,-1],[-1, 1],[1,-1],[1, 1]], dtype=np.int8)
ROT = np.stack([
    np.stack([[-d[1],  d[0]] for d in MOVES]),
    MOVES,
    np.stack([[ d[1], -d[0]] for d in MOVES])
])                                               # 3×9×2

# heuristic weights
CAPTURE_BONUS  = 64.0
CRASH_PENALTY  = -1200.0
W_PREY         =  1.0
W_PURS         =  1.2
STAY_BIAS      = -0.04
RISK_COST      =  40.0

# safety policy
RISK_TH_BASE   = 0.12
RISK_TH_MAX    = 0.28
ADAPT_RATE     = 0.03

# misc
STALL_LIMIT = 10
HISTORY     = 30
def in_bounds(p): return 0 <= p[0] < 30 and 0 <= p[1] < 30
def mhd(a,b):    return abs(int(a[0]-b[0])) + abs(int(a[1]-b[1]))

class PlannerAgent:
    def __init__(self):
        self.risk=None; self.sig=None
        self.open_ast=[]; self.exit_hist=deque(maxlen=HISTORY)
        self.stall_cnt=0; self.idle_steps=0

    # main API
    def plan_action(self, world:np.ndarray,
                    cur:Tuple[int,int],
                    prey:Tuple[int,int],
                    purs:Tuple[int,int]) -> Optional[np.ndarray]:

        if self.sig != hash(world.tobytes()):
            self._build_risk(world)
            self.exit_hist.clear(); self.stall_cnt=0; self.idle_steps=0

        C=np.asarray(cur ,np.int8)
        S=np.asarray(prey,np.int8)
        T=np.asarray(purs,np.int8)

        dyn_th=min(RISK_TH_BASE+ADAPT_RATE*self.idle_steps,RISK_TH_MAX)
        if (self.risk[C[0],C[1],1:]<=dyn_th).sum()==0:
            dyn_th=min(self.risk[C[0],C[1],1:].min()+1e-4,RISK_TH_MAX)

        risk_row=self.risk[C[0],C[1]]
        safe_row=risk_row<=dyn_th

        # EV-aware capture lunge
        mv_lunge=self._lunge_if_profitable(C,S,T,risk_row)
        if mv_lunge is not None:
            return mv_lunge

        # bookkeeping
        self.idle_steps = self.idle_steps+1 if safe_row[1:].sum()==0 else 0
        exits_now=self._exit_count(world,S,C)
        self.exit_hist.append(exits_now)
        if len(self.exit_hist)==self.exit_hist.maxlen:
            self.stall_cnt=self.stall_cnt+1 if exits_now>=min(self.exit_hist) else 0

        # 1. flee
        if mhd(C,T)<=3:
            return self._flee(C,S,T,safe_row)

        # 2. safe A*
        step=self._astar(world,C,S,dyn_th)
        if step is not None and safe_row[self._idx(step)]:
            return step

        # 3. choke
        if self.stall_cnt>=STALL_LIMIT:
            self.stall_cnt=0
            return self._choke(C,S,safe_row,world)

        # 4. expectimax with EV filter
        return self._expectimax(world,C,S,T,safe_row,risk_row)

    # risk table
    def _build_risk(self, world):
        wall=world==1
        base=np.indices((30,30)).transpose(1,2,0)
        trio=base[...,None,None,:]+ROT
        r,c=trio[...,0],trio[...,1]
        hit=(~((r>=0)&(r<30)&(c>=0)&(c<30)) | wall[r.clip(0,29),c.clip(0,29)])
        self.risk=(ROT_P[None,None,:,None]*hit).sum(2).astype(np.float32)
        self.sig=hash(world.tobytes())

    # lunge rule
    def _lunge_if_profitable(self,C,S,T,risk_row):
        idx_best=int(risk_row.argmin())
        if idx_best==0 or risk_row[idx_best]>RISK_TH_MAX: return None
        p_crash=risk_row[idx_best]
        p_cap=0.0
        for rot,p in enumerate(ROT_P):
            if np.array_equal(C+ROT[rot,idx_best],S):
                p_cap+=p
        if p_cap==0: return None
        if mhd(C,T)<=2: delta=3
        else:            delta=2
        if p_cap*delta > p_crash:
            return MOVES[idx_best]
        return None

    # A* search
    def _astar(self,world,start,goal,dyn_th):
        if np.array_equal(start,goal): return MOVES[0]
        seen=np.full((30,30),99,np.int8)
        self.open_ast.clear()
        heappush(self.open_ast,(mhd(start,goal),0,tuple(start),-1))
        while self.open_ast:
            f,g,node,first=heappop(self.open_ast)
            if g>=40 or seen[node]<=g: continue
            seen[node]=g
            if node==tuple(goal):
                return MOVES[first] if first>=0 else MOVES[0]
            r,c=node
            for idx,mv in enumerate(MOVES):
                if self.risk[r,c,idx]>dyn_th: continue
                nxt=(r+mv[0],c+mv[1])
                if not in_bounds(nxt) or world[nxt]: continue
                if seen[nxt]<=g+1: continue
                heappush(self.open_ast,(g+1+mhd(nxt,goal),g+1,nxt,
                         idx if first==-1 else first))
        return None

    # expectimax
    def _expectimax(self,world,C,S,T,safe_row,risk_row):
        ev1,_=self._ev_all(world,C,S,T)
        risk_pen=self._adaptive_risk_cost(C,S)
        ev1 -= risk_pen*risk_row
        ev1[0]+=STAY_BIAS
        best3=np.argsort(-ev1)[:3]
        for idx in best3:
            if risk_row[idx]>RISK_TH_MAX or not safe_row[idx]:
                ev1[idx]=-1e9;continue
            C2=C+MOVES[idx]
            if not in_bounds(C2) or world[tuple(C2)]:
                ev1[idx]+=0.25*CRASH_PENALTY;continue
            ev2,_=self._ev_all(world,C2,S,T)
            ev2-=risk_pen*self.risk[C2[0],C2[1]]
            ev1[idx]=0.7*ev1[idx]+0.3*ev2.max()
        return MOVES[int(ev1.argmax())]

    # EV table
    def _ev_all(self,world,C,S,T):
        nxt=C+ROT
        r,c=nxt[...,0],nxt[...,1]
        hit=(~((r>=0)&(r<30)&(c>=0)&(c<30))|
             world[r.clip(0,29),c.clip(0,29)])
        d_prey=np.abs(r-S[0])+np.abs(c-S[1])
        d_purs=np.abs(r-T[0])+np.abs(c-T[1])
        h=np.where(hit,CRASH_PENALTY,
           np.where((r==S[0])&(c==S[1]),CAPTURE_BONUS,
           np.where((r==T[0])&(c==T[1]),CRASH_PENALTY,
                    -W_PREY*d_prey+W_PURS*d_purs)))
        ev=(ROT_P[:,None]*h).sum(0)
        var=(ROT_P[:,None]*(h-ev)**2).sum(0)
        return ev,var

    # adaptive risk
    def _adaptive_risk_cost(self,C,S):
        d=mhd(C,S)
        return RISK_COST * (0.5 if d<=6 else 1.0)

    # local heuristics
    def _flee(self,C,S,T,safe_row):
        best=-1e9; mv=MOVES[0]
        for idx,m in enumerate(MOVES):
            if self.risk[C[0],C[1],idx]>RISK_TH_MAX or not safe_row[idx]:continue
            nxt=C+m; v=2*mhd(nxt,T)-mhd(nxt,S)
            if v>best: best=v; mv=m
        return mv

    def _choke(self,C,S,safe_row,world):
        base=self._exit_count(world,S,C)
        best=base; mv=MOVES[0]
        for idx,m in enumerate(MOVES):
            if self.risk[C[0],C[1],idx]>RISK_TH_MAX or not safe_row[idx]:continue
            nxt=C+m; ex=self._exit_count(world,S,nxt)
            if ex<best or (ex==best and mhd(nxt,S)<mhd(mv+C,S)):
                best=ex; mv=m
        return mv

    # misc
    @staticmethod
    def _exit_count(world,S,pos):
        e=0
        for dr,dc in((1,0),(-1,0),(0,1),(0,-1)):
            nr,nc=S[0]+dr,S[1]+dc
            if 0<=nr<30 and 0<=nc<30 and world[nr,nc]==0 and (nr,nc)!=tuple(pos):
                e+=1
        return e
    @staticmethod
    def _idx(mv):
        return int(np.where((MOVES==mv).all(1))[0][0])