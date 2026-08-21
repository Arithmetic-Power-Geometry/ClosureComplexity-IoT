from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import heapq, math
import numpy as np
import pandas as pd

AREA=1000.0
SENSE_R=120.0
COMM_R=260.0
GRID_N=22

@dataclass(frozen=True)
class Target:
    coverage: float=0.60
    reliability: float=0.60
    latency_ms: float=120.0
    energy_mj: float=7.0

@dataclass(frozen=True)
class Action:
    kind: str
    x: float
    y: float
    cost: float
    label: str

@dataclass
class Scenario:
    seed:int
    n_nodes:int
    damage:float
    failure_mode:str
    nodes:np.ndarray
    active:np.ndarray
    failed:np.ndarray
    gateways:np.ndarray
    actions:List[Action]


def _grid():
    xs=np.linspace(0,AREA,GRID_N); ys=np.linspace(0,AREA,GRID_N)
    return np.array([(x,y) for x in xs for y in ys],dtype=float)
GRID=_grid()
_METRIC_CACHE={}


def clear_cache():
    _METRIC_CACHE.clear()


def _coverage(points:np.ndarray)->float:
    if len(points)==0:return 0.0
    d=((GRID[:,None,:]-points[None,:,:])**2).sum(axis=2)
    return float((d.min(axis=1)<=SENSE_R**2).mean())


def _state_metrics(points:np.ndarray,gws:np.ndarray,relays:List[List[float]]|None=None,boost:bool=False)->Dict[str,float]:
    relays=relays or []
    cov=_coverage(points)
    if len(points)==0 or len(gws)==0:
        return dict(coverage=cov,reliability=0.0,latency_ms=9999.,energy_mj=9999.)
    aps=np.vstack([gws,np.asarray(relays,float)]) if relays else gws.copy()
    ng=len(gws)
    ds=np.sqrt(((points[:,None,:]-aps[None,:,:])**2).sum(axis=2))
    r=COMM_R*(1.18 if boost else 1.0)
    probs=1/(1+np.exp((ds-r)/(0.14*r)))
    best_idx=np.argmax(probs,axis=1)
    best_p=probs[np.arange(len(points)),best_idx]
    reliability=float(best_p.mean())
    relay_hop=(best_idx>=ng).astype(float)
    latency=float(np.mean(18.0 + 24.0*relay_hop + 35.0*(1-best_p)))
    best_d=ds[np.arange(len(points)),best_idx]
    energy=float(np.mean(0.06 + 1.2e-6*(best_d**2) + 0.03*relay_hop))
    return dict(coverage=cov,reliability=reliability,latency_ms=latency,energy_mj=energy)


def healthy_metrics(s:Scenario)->Dict[str,float]:
    return _state_metrics(s.nodes,s.gateways)


def metrics(s:Scenario, mask:int)->Dict[str,float]:
    key=(s.seed,s.n_nodes,round(s.damage,4),s.failure_mode,len(s.actions),mask)
    if key in _METRIC_CACHE: return _METRIC_CACHE[key].copy()
    pts=s.nodes[s.active].copy(); gws=s.gateways.copy(); relays=[]; boost=False
    for i,a in enumerate(s.actions):
        if not(mask>>i)&1: continue
        if a.kind=='restore': pts=np.vstack([pts,[a.x,a.y]])
        elif a.kind=='relay':
            pts=np.vstack([pts,[a.x,a.y]]); relays.append([a.x,a.y])
        elif a.kind=='gateway': gws=np.vstack([gws,[a.x,a.y]])
        elif a.kind=='boost': boost=True
    out=_state_metrics(pts,gws,relays,boost)
    _METRIC_CACHE[key]=out
    return out.copy()


def cost(s:Scenario, mask:int)->float:
    if mask < 0:return float('nan')
    return float(sum(a.cost for i,a in enumerate(s.actions) if (mask>>i)&1))

def steps(mask:int)->int:return int(mask.bit_count()) if mask>=0 else -1


def selected_actions(s:Scenario, mask:int)->List[str]:
    """Return the exact labels of actions selected by a repair mask."""
    if mask < 0:
        return []
    return [a.label for i,a in enumerate(s.actions) if (mask>>i)&1]

def selected_action_kinds(s:Scenario, mask:int)->List[str]:
    """Return human-readable action kinds for a repair mask, preserving action order."""
    if mask < 0:
        return []
    return [a.kind for i,a in enumerate(s.actions) if (mask>>i)&1]

def satisfies(m:Dict[str,float], t:Target)->bool:
    return m['coverage']>=t.coverage and m['reliability']>=t.reliability and m['latency_ms']<=t.latency_ms and m['energy_mj']<=t.energy_mj

def deficit(m:Dict[str,float],t:Target)->float:
    vals=[max(0,t.coverage-m['coverage'])/max(t.coverage,1e-9),max(0,t.reliability-m['reliability'])/max(t.reliability,1e-9),max(0,m['latency_ms']-t.latency_ms)/max(t.latency_ms,1e-9),max(0,m['energy_mj']-t.energy_mj)/max(t.energy_mj,1e-9)]
    return float(sum(vals))


def _failure_indices(nodes:np.ndarray,k:int,mode:str,rng:np.random.Generator)->np.ndarray:
    n=len(nodes); k=min(max(1,k),n)
    center=np.array([AREA/2,AREA/2],float)
    if mode=='random':
        return rng.choice(n,size=k,replace=False)
    dcenter=np.linalg.norm(nodes-center,axis=1)
    if mode=='gateway-near':
        return np.argsort(dcenter)[:k]
    if mode=='peripheral':
        return np.argsort(dcenter)[-k:]
    if mode=='clustered':
        anchor=nodes[int(rng.integers(n))]
        return np.argsort(np.linalg.norm(nodes-anchor,axis=1))[:k]
    if mode=='coverage-critical':
        # Nodes ranked by exclusive grid coverage contribution in the healthy network.
        d=((GRID[:,None,:]-nodes[None,:,:])**2).sum(axis=2)<=SENSE_R**2
        counts=d.sum(axis=1)
        unique=((d)&(counts[:,None]==1)).sum(axis=0)
        # tie-break with centrality in sensing coverage to keep deterministic ranking
        return np.lexsort((dcenter,-unique))[:k]
    raise ValueError(f'unknown failure mode: {mode}')


def _candidate_holes(activepts:np.ndarray,needed:int)->List[np.ndarray]:
    if len(activepts)==0:
        order=np.arange(len(GRID))
    else:
        d=((GRID[:,None,:]-activepts[None,:,:])**2).sum(axis=2).min(axis=1)
        order=np.argsort(d)[::-1]
    chosen=[]
    for gi in order:
        p=GRID[gi]
        if all(np.linalg.norm(p-q)>150 for q in chosen): chosen.append(p)
        if len(chosen)>=needed:break
    return chosen


def generate_scenario(seed:int,n_nodes:int,damage:float,failure_mode:str='random',n_actions:int=8)->Scenario:
    rng=np.random.default_rng(seed)
    nodes=rng.uniform(20,AREA-20,size=(n_nodes,2))
    k=max(1,int(round(n_nodes*damage)))
    fail=np.asarray(_failure_indices(nodes,k,failure_mode,rng),dtype=int)
    active=np.ones(n_nodes,dtype=bool); active[fail]=False
    gateways=np.array([[AREA/2,AREA/2]],float)
    # Allocate actions deterministically. Always include one range boost.
    restore_n=min(len(fail), max(2, min(4, n_actions//4)))
    gateway_n=2 if n_actions<=9 else (3 if n_actions<=11 else 4)
    gateway_n=min(gateway_n,max(0,n_actions-restore_n-2))
    relay_n=max(1,n_actions-restore_n-gateway_n-1)
    actions=[]
    # Restore failed nodes with greatest healthy coverage contribution first.
    failed_pts=nodes[fail]
    if len(failed_pts):
        dg=((GRID[:,None,:]-failed_pts[None,:,:])**2).sum(axis=2)<=SENSE_R**2
        scores=dg.sum(axis=0)
        for j,local in enumerate(np.argsort(scores)[::-1][:restore_n]):
            idx=fail[int(local)]; x,y=nodes[idx]
            actions.append(Action('restore',float(x),float(y),1.0,f'R{j+1}'))
    holes=_candidate_holes(nodes[active],relay_n)
    for j,p in enumerate(holes): actions.append(Action('relay',float(p[0]),float(p[1]),1.25,f'L{j+1}'))
    gw_positions=[(AREA*.25,AREA*.25),(AREA*.75,AREA*.75),(AREA*.25,AREA*.75),(AREA*.75,AREA*.25)]
    for j,(x,y) in enumerate(gw_positions[:gateway_n]): actions.append(Action('gateway',x,y,2.4,f'G{j+1}'))
    actions.append(Action('boost',0,0,1.6,'B1'))
    # exact length guard
    actions=actions[:n_actions]
    return Scenario(seed,n_nodes,damage,failure_mode,nodes,active,fail,gateways,actions)


def exact_cc(s:Scenario,t:Target)->Tuple[int,Dict[str,float]]:
    pq=[(0.0,0)]; seen=set()
    while pq:
        c,mask=heapq.heappop(pq)
        if mask in seen: continue
        seen.add(mask); m=metrics(s,mask)
        if satisfies(m,t): return mask,m
        for i,a in enumerate(s.actions):
            if (mask>>i)&1:continue
            heapq.heappush(pq,(c+a.cost,mask|(1<<i)))
    return -1,metrics(s,0)


def exact_single_target(s:Scenario,t:Target,which:str)->Tuple[int,Dict[str,float]]:
    def ok(m):
        if which=='coverage': return m['coverage']>=t.coverage
        if which=='reliability': return m['reliability']>=t.reliability
        if which=='latency': return m['latency_ms']<=t.latency_ms
        if which=='energy': return m['energy_mj']<=t.energy_mj
        raise ValueError(which)
    pq=[(0.,0)];seen=set()
    while pq:
        c,mask=heapq.heappop(pq)
        if mask in seen:continue
        seen.add(mask);m=metrics(s,mask)
        if ok(m):return mask,m
        for i,a in enumerate(s.actions):
            if not((mask>>i)&1):heapq.heappush(pq,(c+a.cost,mask|(1<<i)))
    return -1,metrics(s,0)


def greedy(s:Scenario,t:Target)->Tuple[int,Dict[str,float]]:
    mask=0
    for _ in range(len(s.actions)):
        m=metrics(s,mask)
        if satisfies(m,t):return mask,m
        base=deficit(m,t);best=None
        for i,a in enumerate(s.actions):
            if (mask>>i)&1:continue
            nm=mask|(1<<i); mm=metrics(s,nm); gain=(base-deficit(mm,t))/a.cost
            cand=(gain,-a.cost,-i,nm,mm)
            if best is None or cand[:3]>best[:3]:best=cand
        if best is None:break
        mask=best[3]
    return mask,metrics(s,mask)


def random_search(s:Scenario,t:Target,seed:int,budget:int=300)->Tuple[int,Dict[str,float]]:
    rng=np.random.default_rng(seed); best=None
    for _ in range(budget):
        mask=int(rng.integers(0,1<<len(s.actions))); m=metrics(s,mask)
        score=(0 if satisfies(m,t) else 1, cost(s,mask) if satisfies(m,t) else deficit(m,t), steps(mask))
        if best is None or score<best[0]:best=(score,mask,m)
    return best[1],best[2]


def genetic(s:Scenario,t:Target,seed:int,pop:int=28,generations:int=24)->Tuple[int,Dict[str,float]]:
    rng=np.random.default_rng(seed); d=len(s.actions); P=rng.integers(0,2,size=(pop,d),dtype=np.int8)
    def evalrow(row):
        mask=sum(int(v)<<i for i,v in enumerate(row));m=metrics(s,mask)
        return (cost(s,mask) if satisfies(m,t) else 100+50*deficit(m,t)+0.1*cost(s,mask),mask,m)
    for _ in range(generations):
        ranked=sorted((evalrow(r) for r in P),key=lambda x:x[0]); elites=[]
        for _,mask,_ in ranked[:max(2,pop//3)]: elites.append(np.array([(mask>>i)&1 for i in range(d)],dtype=np.int8))
        new=elites.copy()
        while len(new)<pop:
            a,b=elites[rng.integers(len(elites))],elites[rng.integers(len(elites))]
            cut=int(rng.integers(1,d)); child=np.r_[a[:cut],b[cut:]].copy(); mut=rng.random(d)<0.08; child[mut]=1-child[mut];new.append(child)
        P=np.stack(new[:pop])
    ranked=sorted((evalrow(r) for r in P),key=lambda x:x[0]);return ranked[0][1],ranked[0][2]


def pso_binary(s:Scenario,t:Target,seed:int,pop:int=26,iters:int=24)->Tuple[int,Dict[str,float]]:
    rng=np.random.default_rng(seed); d=len(s.actions); x=rng.integers(0,2,size=(pop,d));v=rng.normal(0,0.2,size=(pop,d))
    def score(row):
        mask=sum(int(q)<<i for i,q in enumerate(row));m=metrics(s,mask)
        return (cost(s,mask) if satisfies(m,t) else 100+50*deficit(m,t)+0.1*cost(s,mask),mask,m)
    p=x.copy(); ps=np.array([score(r)[0] for r in x]); g=p[int(ps.argmin())].copy()
    for _ in range(iters):
        r1=rng.random((pop,d));r2=rng.random((pop,d));v=.72*v+1.45*r1*(p-x)+1.45*r2*(g-x)
        prob=1/(1+np.exp(-v));x=(rng.random((pop,d))<prob).astype(int)
        for i in range(pop):
            sc=score(x[i])[0]
            if sc<ps[i]:ps[i]=sc;p[i]=x[i].copy()
        g=p[int(ps.argmin())].copy()
    q=score(g);return q[1],q[2]


def _dominates(a,b):
    return (a[0]<=b[0] and a[1]<=b[1]) and (a[0]<b[0] or a[1]<b[1])

def _rank_and_crowding(objs:np.ndarray):
    n=len(objs); dom_count=np.zeros(n,int); dominates=[[] for _ in range(n)]; fronts=[[]]
    for i in range(n):
        for j in range(n):
            if i==j:continue
            if _dominates(objs[i],objs[j]):dominates[i].append(j)
            elif _dominates(objs[j],objs[i]):dom_count[i]+=1
        if dom_count[i]==0:fronts[0].append(i)
    rank=np.full(n,10**6,int);k=0
    while k<len(fronts) and fronts[k]:
        nxt=[]
        for i in fronts[k]:
            rank[i]=k
            for j in dominates[i]:
                dom_count[j]-=1
                if dom_count[j]==0:nxt.append(j)
        k+=1
        if nxt:fronts.append(nxt)
    crowd=np.zeros(n,float)
    for fr in fronts:
        if not fr:continue
        if len(fr)<=2:
            crowd[fr]=np.inf;continue
        vals=objs[fr]
        for q in range(vals.shape[1]):
            order=np.argsort(vals[:,q]); crowd[fr[order[0]]]=np.inf; crowd[fr[order[-1]]]=np.inf
            lo,hi=vals[order[0],q],vals[order[-1],q]
            if hi<=lo:continue
            for u in range(1,len(order)-1):crowd[fr[order[u]]]+=(vals[order[u+1],q]-vals[order[u-1],q])/(hi-lo)
    return rank,crowd


def nsga2_binary(s:Scenario,t:Target,seed:int,pop:int=32,generations:int=28)->Tuple[int,Dict[str,float]]:
    """Generic binary NSGA-II baseline on (repair cost, normalized target deficit).
    This is an independent standard-algorithm implementation, not source code from any cited paper.
    """
    rng=np.random.default_rng(seed); d=len(s.actions)
    P=rng.integers(0,2,size=(pop,d),dtype=np.int8)
    def maskrow(row):return sum(int(v)<<i for i,v in enumerate(row))
    def objective(row):
        mask=maskrow(row);m=metrics(s,mask);return np.array([cost(s,mask),deficit(m,t)],float)
    for _ in range(generations):
        objs=np.array([objective(r) for r in P]); rank,crowd=_rank_and_crowding(objs)
        children=[]
        while len(children)<pop:
            def tour():
                a,b=int(rng.integers(pop)),int(rng.integers(pop))
                if rank[a]<rank[b] or (rank[a]==rank[b] and crowd[a]>=crowd[b]):return P[a]
                return P[b]
            a,b=tour(),tour(); cut=int(rng.integers(1,d)); child=np.r_[a[:cut],b[cut:]].copy(); mut=rng.random(d)<(1/max(d,1));child[mut]=1-child[mut];children.append(child)
        C=np.stack(children); R=np.vstack([P,C]); robj=np.array([objective(r) for r in R]); rrank,rcrowd=_rank_and_crowding(robj)
        order=sorted(range(len(R)),key=lambda i:(rrank[i],-rcrowd[i]));P=R[order[:pop]]
    candidates=[]
    for r in P:
        mask=maskrow(r);m=metrics(s,mask);candidates.append((0 if satisfies(m,t) else 1,cost(s,mask) if satisfies(m,t) else deficit(m,t),steps(mask),mask,m))
    best=min(candidates,key=lambda z:z[:3]);return best[3],best[4]


def enumerate_states(s:Scenario,t:Target)->pd.DataFrame:
    rows=[]
    for mask in range(1<<len(s.actions)):
        m=metrics(s,mask);rows.append(dict(mask=mask,cost=cost(s,mask),steps=steps(mask),**m,satisfies=satisfies(m,t)))
    return pd.DataFrame(rows)


def pareto(df:pd.DataFrame)->pd.DataFrame:
    sat=df[df.satisfies].copy()
    if sat.empty:return sat
    vals=np.c_[sat.cost,sat.energy_mj,sat.latency_ms,-sat.coverage,-sat.reliability]
    keep=np.ones(len(vals),dtype=bool)
    for i in range(len(vals)):
        if not keep[i]:continue
        dom=np.all(vals<=vals[i],axis=1)&np.any(vals<vals[i],axis=1)
        if dom.any():keep[i]=False
    return sat.iloc[np.where(keep)[0]].sort_values(['cost','energy_mj']).reset_index(drop=True)
