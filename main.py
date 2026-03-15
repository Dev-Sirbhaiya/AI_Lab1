"""
CZ3005 / SC3000 Artificial Intelligence — Lab Assignment 1
===========================================================
Part 1 : NYC Road Network shortest path with energy budget
  Task 1 : Dijkstra  (no energy constraint)
  Task 2 : UCS       (energy budget = 287,932   uninformed)
  Task 3 : A*        (energy budget = 287,932   Euclidean heuristic)

Part 2 : 5x5 Grid World MDP & Reinforcement Learning
  Task 1 : Value Iteration  + Policy Iteration  (gamma = 0.9)
  Task 2 : First-Visit Monte Carlo Control      (epsilon = 0.1)
  Task 3 : Tabular Q-Learning                   (epsilon = 0.1, alpha = 0.1)

Run   : python main.py
Output: main_results.txt  +  9 PNG plots
"""

import json, heapq, math, random, time, collections, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

# ─────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────
SOURCE        = "1"
TARGET        = "50"
ENERGY_BUDGET = 287_932
THETA         = 1e-6
GAMMA         = 0.9
MC_EPISODES   = 10_000
QL_EPISODES   = 10_000
EPSILON       = 0.1
ALPHA         = 0.1
MAX_STEPS     = 1_000

# ─────────────────────────────────────────────────────────────
# TEE OUTPUT TO FILE AND CONSOLE
# ─────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, s):
        for f in self.files: f.write(s)
    def flush(self):
        for f in self.files: f.flush()

_log = open("main_results.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log)

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
print("Loading NYC road network ...")
_t0 = time.time()
with open("G.json")     as f: G     = json.load(f)
with open("Coord.json") as f: Coord = json.load(f)
with open("Dist.json")  as f: Dist  = json.load(f)
with open("Cost.json")  as f: Cost  = json.load(f)
print(f"  {len(G):,} nodes  |  {len(Dist):,} edges  ({time.time()-_t0:.1f}s)\n")


# =============================================================
# PART 1 — NYC ROAD NETWORK
# =============================================================

def _reconstruct(parent):
    path, n = [], TARGET
    while n is not None:
        path.append(n)
        n = parent[n][0] if n in parent else None
    path.reverse()
    return path

def fmt_path(path, show=5):
    if len(path) <= 2*show: return "->".join(path)
    return "->".join(path[:show]) + " -> ... -> " + "->".join(path[-show:])


# ── Task 1 : Dijkstra ────────────────────────────────────────
def task1_dijkstra():
    parent   = {SOURCE: (None, 0.0, 0.0)}
    visited  = set()
    heap     = [(0.0, SOURCE)]
    expanded = 0
    while heap:
        dist, node = heapq.heappop(heap)
        if node in visited: continue
        visited.add(node); expanded += 1
        if node == TARGET: break
        pd, pe = parent[node][1], parent[node][2]
        for nb in G.get(node, []):
            edge = f"{node},{nb}"
            if nb not in visited and edge in Dist:
                nd = pd + Dist[edge]
                ne = pe + Cost.get(edge, 0.0)
                if nb not in parent or parent[nb][1] > nd:
                    parent[nb] = (node, nd, ne)
                heapq.heappush(heap, (nd, nb))
    return _reconstruct(parent), parent[TARGET][1], parent[TARGET][2], expanded


# ── Task 2 : UCS with energy constraint ──────────────────────
def task2_ucs():
    # Standard Dijkstra on distance with energy as a hard budget prune.
    # A node is settled on its FIRST pop (minimum distance guaranteed by heap).
    # Energy is only used to filter out neighbours that would exceed the budget.
    parent   = {SOURCE: (None, 0.0, 0.0)}
    visited  = set()
    heap     = [(0.0, SOURCE, 0.0)]   # (dist, node, energy)
    expanded = 0
    while heap:
        dist, node, energy = heapq.heappop(heap)
        if node in visited: continue          # already settled at min-distance
        visited.add(node); expanded += 1
        if node == TARGET: break
        pd, pe = parent[node][1], parent[node][2]
        for nb in G.get(node, []):
            edge = f"{node},{nb}"
            if edge not in Dist: continue
            nd = pd + Dist[edge]
            ne = pe + Cost.get(edge, 0.0)
            if ne > ENERGY_BUDGET: continue   # hard budget prune
            if nb not in visited:
                if nb not in parent or parent[nb][1] > nd:
                    parent[nb] = (node, nd, ne)
                heapq.heappush(heap, (nd, nb, ne))
    return _reconstruct(parent), parent[TARGET][1], parent[TARGET][2], expanded


# ── Task 3 : A* with energy constraint ───────────────────────
def _h(node):
    x1,y1 = Coord[node]; x2,y2 = Coord[TARGET]
    return math.hypot(x1-x2, y1-y2)

def task3_astar():
    # A* on distance with f(n) = g(n) + h(n), h = Euclidean distance (admissible).
    # Standard visited set: a node is settled on first pop (min-distance guaranteed).
    # Energy is only used as a hard budget prune when pushing neighbours.
    parent   = {SOURCE: (None, 0.0, 0.0)}
    visited  = set()
    heap     = [(_h(SOURCE), 0.0, SOURCE, 0.0)]   # (f, g, node, energy)
    expanded = 0
    while heap:
        f, g, node, energy = heapq.heappop(heap)
        if node in visited: continue              # already settled at min-distance
        visited.add(node); expanded += 1
        if node == TARGET: break
        pg, pe = parent[node][1], parent[node][2]
        for nb in G.get(node, []):
            edge = f"{node},{nb}"
            if edge not in Dist: continue
            ng = pg + Dist[edge]
            ne = pe + Cost.get(edge, 0.0)
            if ne > ENERGY_BUDGET: continue       # hard budget prune
            if nb not in visited:
                if nb not in parent or parent[nb][1] > ng:
                    parent[nb] = (node, ng, ne)
                heapq.heappush(heap, (ng + _h(nb), ng, nb, ne))
    return _reconstruct(parent), parent[TARGET][1], parent[TARGET][2], expanded


# ── Run Part 1 ───────────────────────────────────────────────
def run_part1():
    sep = "="*64
    print(f"\n{sep}\n  PART 1 — NYC Road Network\n{sep}")

    rows = []
    all_results = []
    for tag, fn, has_budget in [
        ("Task 1 — Dijkstra (no energy constraint)", task1_dijkstra, False),
        (f"Task 2 — UCS  (budget={ENERGY_BUDGET:,})",  task2_ucs,   True),
        (f"Task 3 — A*   (budget={ENERGY_BUDGET:,})",  task3_astar, True),
    ]:
        t0 = time.time()
        path, dist, energy, exp = fn()
        elapsed = time.time() - t0
        feasible = (not has_budget) or (energy <= ENERGY_BUDGET)
        print(f"\n  {tag}")
        print(f"  {'─'*60}")
        print(f"  Shortest path    : {fmt_path(path)}.")
        print(f"  Shortest distance: {dist:,.4f}.")
        print(f"  Total energy cost: {energy:,.4f}.")
        print(f"  Nodes in path    : {len(path)}")
        print(f"  Nodes expanded   : {exp:,}")
        print(f"  Time             : {elapsed:.4f}s")
        print(f"  Budget feasible  : {'YES' if feasible else 'NO (no constraint)'}")
        r = dict(path=path, dist=dist, energy=energy, expanded=exp, elapsed=elapsed)
        all_results.append(r)
        rows.append(r)

    t1r, t2r, t3r = rows
    print(f"\n  {'─'*64}\n  {'SUMMARY':^64}\n  {'─'*64}")
    print(f"  {'Algorithm':<22} {'Distance':>12} {'Energy':>12} {'Expanded':>10} {'Time':>8}")
    print(f"  {'─'*64}")
    for lbl, r in [("Dijkstra (Task 1)", t1r), ("UCS (Task 2)", t2r), ("A* (Task 3)", t3r)]:
        print(f"  {lbl:<22} {r['dist']:>12,.2f} {r['energy']:>12,.0f} {r['expanded']:>10,} {r['elapsed']:>7.3f}s")
    print(f"  {'─'*64}")
    di = (t2r["dist"]-t3r["dist"])/t2r["dist"]*100
    ei = (t2r["energy"]-t3r["energy"])/t2r["energy"]*100
    ni = (t2r["expanded"]-t3r["expanded"])/t2r["expanded"]*100
    print(f"\n  A* vs UCS: distance {di:+.1f}%  |  energy {ei:+.1f}%  |  nodes expanded {ni:+.1f}%")
    return t1r, t2r, t3r


# ── Part 1 Plots ─────────────────────────────────────────────
def plot_p1_map(t1, t2, t3):
    fig, ax = plt.subplots(figsize=(15, 11))
    fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#16213e")
    random.seed(0)
    sample = random.sample(list(Coord.keys()), 8000)
    ax.scatter([Coord[n][0] for n in sample], [Coord[n][1] for n in sample],
               s=0.3, c="#4a4a6a", alpha=0.6, linewidths=0)
    for r, col, lbl, lw in [
        (t1,"#00b4d8",f"Task 1 Dijkstra  dist={t1['dist']:,.0f}  energy={t1['energy']:,.0f}",2.5),
        (t2,"#f77f00",f"Task 2 UCS       dist={t2['dist']:,.0f}  energy={t2['energy']:,.0f}",2.0),
        (t3,"#06d6a0",f"Task 3 A*        dist={t3['dist']:,.0f}  energy={t3['energy']:,.0f}",1.5),
    ]:
        px=[Coord[n][0] for n in r["path"] if n in Coord]
        py=[Coord[n][1] for n in r["path"] if n in Coord]
        ax.plot(px, py, "-", color=col, lw=lw, label=lbl, alpha=0.9)
    for coord, c, lbl in [(Coord[SOURCE],"#ff006e",f"Start ({SOURCE})"),
                           (Coord[TARGET],"#ffbe0b",f"End ({TARGET})")]:
        ax.scatter([coord[0]],[coord[1]],s=300,c=c,zorder=8,marker="*",
                   edgecolors="white",linewidths=0.8,label=lbl)
    ax.legend(loc="lower right",fontsize=9,facecolor="#0f3460",
              edgecolor="#00b4d8",labelcolor="white",framealpha=0.9)
    ax.set_title("Part 1 — NYC Road Network: Path Comparison",
                 fontsize=13,color="white",fontweight="bold",pad=14)
    ax.set_xlabel("Longitude (×10⁻⁷)",color="white")
    ax.set_ylabel("Latitude (×10⁻⁷)",color="white")
    ax.tick_params(colors="white"); ax.spines[:].set_color("#00b4d8")
    plt.tight_layout()
    plt.savefig("part1_path_map.png",dpi=160,bbox_inches="tight"); plt.close()
    print("  Saved: part1_path_map.png")


def plot_p1_stats(t1, t2, t3):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Part 1 — Algorithm Performance Comparison", fontsize=14, fontweight="bold")
    tasks  = ["Task 1\nDijkstra", "Task 2\nUCS", "Task 3\nA*"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    metrics = [
        ("Shortest Distance",        [t1["dist"],     t2["dist"],     t3["dist"]],     "Distance"),
        ("Total Energy Cost",         [t1["energy"],   t2["energy"],   t3["energy"]],   "Energy"),
        ("Nodes Expanded",            [t1["expanded"], t2["expanded"], t3["expanded"]],  "Nodes"),
        ("Execution Time (s)",        [t1["elapsed"],  t2["elapsed"],  t3["elapsed"]],   "Seconds"),
    ]
    for ax, (title, vals, ylabel) in zip(axes.flat, metrics):
        bars = ax.bar(tasks, vals, color=colors, alpha=0.85, edgecolor="black", lw=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                    f"{v:,.2f}" if v < 1e4 else f"{v:,.0f}",
                    ha="center", va="bottom", fontsize=9)
        if "Energy" in title:
            ax.axhline(ENERGY_BUDGET, color="red", ls="--", lw=1.5, label=f"Budget {ENERGY_BUDGET:,}")
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", alpha=0.35); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("part1_algorithm_stats.png", dpi=160, bbox_inches="tight"); plt.close()
    print("  Saved: part1_algorithm_stats.png")


def plot_p1_profiles(t1, t2, t3):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Part 1 — Cumulative Distance & Energy Along Each Path",
                 fontsize=13, fontweight="bold")
    for r, col, lbl in [
        (t1,"#1f77b4","Task 1 Dijkstra"),
        (t2,"#ff7f0e","Task 2 UCS"),
        (t3,"#2ca02c","Task 3 A*"),
    ]:
        p = r["path"]
        cd, ce = [0.0], [0.0]
        for i in range(len(p)-1):
            e = f"{p[i]},{p[i+1]}"
            cd.append(cd[-1]+Dist.get(e,0.0))
            ce.append(ce[-1]+Cost.get(e,0.0))
        ax1.plot(range(len(cd)), cd, color=col, lw=2, label=lbl)
        ax2.plot(range(len(ce)), ce, color=col, lw=2, label=lbl)
    ax1.set_title("Cumulative Distance", fontweight="bold")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Distance"); ax1.legend(); ax1.grid(alpha=0.35)
    ax2.axhline(ENERGY_BUDGET, color="red", ls="--", lw=1.5, label=f"Budget {ENERGY_BUDGET:,}")
    ax2.set_title("Cumulative Energy", fontweight="bold")
    ax2.set_xlabel("Step"); ax2.set_ylabel("Energy"); ax2.legend(); ax2.grid(alpha=0.35)
    plt.tight_layout()
    plt.savefig("part1_path_profiles.png", dpi=160, bbox_inches="tight"); plt.close()
    print("  Saved: part1_path_profiles.png")


# =============================================================
# PART 2 — GRID WORLD
# =============================================================

ACTIONS    = ["UP","DOWN","LEFT","RIGHT"]
DELTAS     = {"UP":(0,1),"DOWN":(0,-1),"LEFT":(-1,0),"RIGHT":(1,0)}
ARROWS     = {"UP":"↑","DOWN":"↓","LEFT":"←","RIGHT":"→"}
ROADBLOCKS = frozenset([(2,1),(2,3)])
GOAL       = (4,4)
START      = (0,0)
PERP       = {
    "UP":   [(-1,0),(1,0)],
    "DOWN": [(1,0),(-1,0)],
    "LEFT": [(0,-1),(0,1)],
    "RIGHT":[(0,1),(0,-1)],
}
ALL_STATES = [(x,y) for x in range(5) for y in range(5)
              if (x,y) not in ROADBLOCKS and (x,y) != GOAL]


def _clip(x,y,dx,dy):
    nx,ny = x+dx, y+dy
    if 0<=nx<5 and 0<=ny<5 and (nx,ny) not in ROADBLOCKS: return (nx,ny)
    return (x,y)

def transitions(s, a):
    if s == GOAL: return [(1.0,GOAL,0)]
    x,y = s; dx,dy = DELTAS[a]; p1,p2 = PERP[a]
    return [(0.8,_clip(x,y,dx,dy),  10 if _clip(x,y,dx,dy)==GOAL else -1),
            (0.1,_clip(x,y,*p1), 10 if _clip(x,y,*p1)==GOAL else -1),
            (0.1,_clip(x,y,*p2), 10 if _clip(x,y,*p2)==GOAL else -1)]

def sample_step(s, a):
    if s==GOAL: return GOAL,0
    rng,cum = random.random(),0.0
    for p,ns,r in transitions(s,a):
        cum+=p
        if rng<=cum: return ns,r
    return GOAL,0

def qval(s,a,V):
    return sum(p*(r+GAMMA*V.get(ns,0.0)) for p,ns,r in transitions(s,a))

def best_act(s,V):
    return max(ACTIONS, key=lambda a: qval(s,a,V))


# ── Task 1A : Value Iteration ────────────────────────────────
def value_iteration():
    V={s:0.0 for s in ALL_STATES}; V[GOAL]=0.0
    deltas=[]; iters=0
    while True:
        delta=0.0; Vn={}
        for s in ALL_STATES:
            v=max(qval(s,a,V) for a in ACTIONS)
            delta=max(delta,abs(v-V[s])); Vn[s]=v
        V.update(Vn); V[GOAL]=0.0
        deltas.append(delta); iters+=1
        if delta<THETA: break
    return V, {s:best_act(s,V) for s in ALL_STATES}, deltas


# ── Task 1B : Policy Iteration ───────────────────────────────
def policy_iteration():
    policy={s:"UP" for s in ALL_STATES}
    V={s:0.0 for s in ALL_STATES}; V[GOAL]=0.0
    pi_iters=0; eval_counts=[]
    while True:
        evals=0
        while True:
            delta=0.0
            for s in ALL_STATES:
                old=V[s]; V[s]=qval(s,policy[s],V)
                delta=max(delta,abs(V[s]-old))
            evals+=1
            if delta<THETA: break
        eval_counts.append(evals)
        stable=True
        for s in ALL_STATES:
            old=policy[s]
            policy[s]=max(ACTIONS,key=lambda a:qval(s,a,V))
            if policy[s]!=old: stable=False
        pi_iters+=1
        if stable: break
    return V, policy, pi_iters, eval_counts


# ── Task 2 : Monte Carlo ─────────────────────────────────────
def monte_carlo_control():
    Q=collections.defaultdict(float)
    ret_s=collections.defaultdict(float)
    ret_n=collections.defaultdict(int)
    ep_rews=[]; ep_lens=[]; snaps={}

    def eps(s):
        if random.random()<EPSILON: return random.choice(ACTIONS)
        mx=max(Q[(s,a)] for a in ACTIONS)
        return random.choice([a for a in ACTIONS if Q[(s,a)]==mx])

    for ep in range(1, MC_EPISODES+1):
        traj=[]; s=START
        for _ in range(MAX_STEPS):
            a=eps(s); ns,r=sample_step(s,a)
            traj.append((s,a,r)); s=ns
            if s==GOAL: break
        G=0.0; seen=set()
        for t in range(len(traj)-1,-1,-1):
            st,at,rt=traj[t]; G=GAMMA*G+rt
            if (st,at) not in seen:
                seen.add((st,at)); ret_s[(st,at)]+=G; ret_n[(st,at)]+=1
                Q[(st,at)]=ret_s[(st,at)]/ret_n[(st,at)]
        ep_rews.append(sum(r for _,_,r in traj)); ep_lens.append(len(traj))
        if ep in {500,1000,2000,5000,10000}:
            snaps[ep]={s:max(ACTIONS,key=lambda a:Q[(s,a)]) for s in ALL_STATES}

    policy={s:max(ACTIONS,key=lambda a:Q[(s,a)]) for s in ALL_STATES}
    V={s:max(Q[(s,a)] for a in ACTIONS) for s in ALL_STATES}
    return policy, V, dict(Q), ep_rews, ep_lens, snaps


# ── Task 3 : Q-Learning ──────────────────────────────────────
def q_learning():
    Q=collections.defaultdict(float)
    ep_rews=[]; ep_lens=[]; snaps={}

    def eps(s):
        if random.random()<EPSILON: return random.choice(ACTIONS)
        mx=max(Q[(s,a)] for a in ACTIONS)
        return random.choice([a for a in ACTIONS if Q[(s,a)]==mx])

    for ep in range(1, QL_EPISODES+1):
        s=START; total=0.0; steps=0
        for _ in range(MAX_STEPS):
            a=eps(s); ns,r=sample_step(s,a)
            bq=0.0 if ns==GOAL else max(Q[(ns,a2)] for a2 in ACTIONS)
            Q[(s,a)]+=ALPHA*(r+GAMMA*bq-Q[(s,a)])
            s=ns; total+=r; steps+=1
            if s==GOAL: break
        ep_rews.append(total); ep_lens.append(steps)
        if ep in {500,1000,2000,5000,10000}:
            snaps[ep]={s:max(ACTIONS,key=lambda a:Q[(s,a)]) for s in ALL_STATES}

    policy={s:max(ACTIONS,key=lambda a:Q[(s,a)]) for s in ALL_STATES}
    V={s:max(Q[(s,a)] for a in ACTIONS) for s in ALL_STATES}
    return policy, V, dict(Q), ep_rews, ep_lens, snaps


# ── Part 2 text helpers ──────────────────────────────────────
def _vgrid(V):
    rows=[]
    for y in range(4,-1,-1):
        c=[]
        for x in range(5):
            if (x,y)==GOAL:           c.append("  GOAL ")
            elif (x,y) in ROADBLOCKS: c.append("  BLOK ")
            else:                     c.append(f" {V.get((x,y),0):6.2f}")
        rows.append(f"  y={y} |{''.join(c)}")
    return "\n".join(rows)

def _pgrid(pol):
    rows=[]
    for y in range(4,-1,-1):
        c=[]
        for x in range(5):
            s=(x,y)
            if s==GOAL:           c.append("  G")
            elif s in ROADBLOCKS: c.append("  X")
            else:                 c.append(f"  {ARROWS.get(pol.get(s,'?'),'?')}")
        rows.append(f"  y={y} |{''.join(c)}")
    return "\n".join(rows)


# ── Run Part 2 ───────────────────────────────────────────────
def run_part2():
    sep="="*64
    print(f"\n{sep}\n  PART 2 — Grid World MDP & Reinforcement Learning\n{sep}")

    # Task 1A
    print("\n-- Task 1A : Value Iteration ------------------------------------")
    t0=time.time()
    V_vi,pol_vi,vi_d=value_iteration(); vi_t=time.time()-t0
    print(f"  Converged in {len(vi_d)} iterations  (delta_final={vi_d[-1]:.2e})  [{vi_t:.3f}s]")
    print(f"\n  Value Function (VI):\n{_vgrid(V_vi)}\n  Optimal Policy (VI):\n{_pgrid(pol_vi)}")

    # Task 1B
    print("\n-- Task 1B : Policy Iteration -----------------------------------")
    t0=time.time()
    V_pi,pol_pi,pi_out,pi_ec=policy_iteration(); pi_t=time.time()-t0
    print(f"  Outer iters={pi_out}  total inner sweeps={sum(pi_ec)}  [{pi_t:.3f}s]")
    print(f"\n  Value Function (PI):\n{_vgrid(V_pi)}\n  Optimal Policy (PI):\n{_pgrid(pol_pi)}")
    ag=sum(1 for s in ALL_STATES if pol_vi.get(s)==pol_pi.get(s))
    print(f"\n  VI vs PI: {ag}/{len(ALL_STATES)} states identical  "
          f"({'IDENTICAL' if ag==len(ALL_STATES) else 'minor ties'})")
    print(f"  Max |V_VI - V_PI| = {max(abs(V_vi.get(s,0)-V_pi.get(s,0)) for s in ALL_STATES):.2e}")

    # Task 2
    print(f"\n-- Task 2 : Monte Carlo  (eps={EPSILON}, {MC_EPISODES:,} episodes) --------")
    t0=time.time()
    pol_mc,V_mc,Q_mc,mc_r,mc_l,mc_s=monte_carlo_control(); mc_t=time.time()-t0
    print(f"  Done [{mc_t:.1f}s]  final avg reward={np.mean(mc_r[-100:]):.3f}  "
          f"avg length={np.mean(mc_l[-100:]):.1f}")
    print(f"\n  Learned Policy (MC):\n{_pgrid(pol_mc)}")
    mc_ag=sum(1 for s in ALL_STATES if pol_vi.get(s)==pol_mc.get(s))
    print(f"  MC vs VI: {mc_ag}/{len(ALL_STATES)} ({100*mc_ag/len(ALL_STATES):.0f}%)")

    # Task 3
    print(f"\n-- Task 3 : Q-Learning  (eps={EPSILON}, alpha={ALPHA}, {QL_EPISODES:,} episodes) --")
    t0=time.time()
    pol_ql,V_ql,Q_ql,ql_r,ql_l,ql_s=q_learning(); ql_t=time.time()-t0
    print(f"  Done [{ql_t:.1f}s]  final avg reward={np.mean(ql_r[-100:]):.3f}  "
          f"avg length={np.mean(ql_l[-100:]):.1f}")
    print(f"\n  Learned Policy (Q-Learning):\n{_pgrid(pol_ql)}")
    ql_ag=sum(1 for s in ALL_STATES if pol_vi.get(s)==pol_ql.get(s))
    print(f"  QL vs VI: {ql_ag}/{len(ALL_STATES)} ({100*ql_ag/len(ALL_STATES):.0f}%)")

    # 3-way comparison
    print("\n-- 3-Way Policy Comparison : VI | MC | Q-Learning ---------------")
    for y in range(4,-1,-1):
        row=f"  y={y} |"
        for x in range(5):
            s=(x,y)
            if s==GOAL:           row+="  G/G/G  "
            elif s in ROADBLOCKS: row+="  X/X/X  "
            else:
                a1=ARROWS.get(pol_vi.get(s,'?'),'?')
                a2=ARROWS.get(pol_mc.get(s,'?'),'?')
                a3=ARROWS.get(pol_ql.get(s,'?'),'?')
                row+=f" {a1}/{a2}/{a3}{'*' if a1==a2==a3 else ' '} "
        print(row)
    all3=sum(1 for s in ALL_STATES if pol_vi.get(s)==pol_mc.get(s)==pol_ql.get(s))
    print(f"  All three agree: {all3}/{len(ALL_STATES)} ({100*all3/len(ALL_STATES):.0f}%)")

    # Convergence
    def conv(rews,thr=-5,w=100):
        ma=[np.mean(rews[max(0,i-w+1):i+1]) for i in range(len(rews))]
        for i,v in enumerate(ma):
            if v>=thr: return i+1
        return len(rews)
    mc_c=conv(mc_r); ql_c=conv(ql_r)
    print(f"\n-- Convergence Analysis (avg reward >= -5, 100-ep window) -------")
    print(f"  Monte Carlo  : episode {mc_c}")
    print(f"  Q-Learning   : episode {ql_c}")
    winner="Q-Learning" if ql_c<mc_c else "Monte Carlo"
    print(f"  {winner} converges ~{abs(mc_c-ql_c)} episodes faster")
    print("  Reason: Q-Learning updates every step (TD bootstrapping) vs MC needs full return")

    return (V_vi,pol_vi,vi_d, V_pi,pol_pi,pi_out,
            pol_mc,V_mc,Q_mc,mc_r,mc_l,mc_s,
            pol_ql,V_ql,Q_ql,ql_r,ql_l,ql_s,
            mc_ag,ql_ag,all3)


# ── Part 2 Plots ─────────────────────────────────────────────
_CELL_CMAP = LinearSegmentedColormap.from_list(
    "g2r", ["#d32f2f","#ff8f00","#fdd835","#81c784","#2e7d32"])

def _draw_cell(ax, pol, V, show_val=True):
    vmin=min((V.get(s,0) for s in ALL_STATES),default=-15)
    vmax=max((V.get(s,0) for s in ALL_STATES),default=5)
    norm=Normalize(vmin=vmin,vmax=vmax)
    for x in range(5):
        for y in range(5):
            s=(x,y)
            if s==GOAL:
                ax.add_patch(plt.Rectangle([x,y],1,1,fc="#388E3C",ec="#111",lw=1.5))
                ax.text(x+.5,y+.5,"GOAL\n+10",ha="center",va="center",
                        fontsize=9,fontweight="bold",color="white")
            elif s in ROADBLOCKS:
                ax.add_patch(plt.Rectangle([x,y],1,1,fc="#263238",ec="#111",lw=1.5))
                ax.text(x+.5,y+.5,"X",ha="center",va="center",fontsize=16,color="#78909C")
            else:
                v=V.get(s,0.0)
                fc="#BBDEFB" if s==START else _CELL_CMAP(norm(v))
                ax.add_patch(plt.Rectangle([x,y],1,1,fc=fc,ec="#111",lw=1.5))
                ax.text(x+.5,y+.62,ARROWS.get(pol.get(s,""),"?"),ha="center",
                        va="center",fontsize=22,color="black")
                if show_val:
                    ax.text(x+.5,y+.2,f"{v:.2f}",ha="center",va="center",fontsize=7.5)
                if s==START:
                    ax.text(x+.5,y+.87,"S",ha="center",va="center",
                            fontsize=7,color="#0D47A1",fontweight="bold")
    ax.set_xlim(0,5);ax.set_ylim(0,5)
    ax.set_xticks(np.arange(.5,5,1)); ax.set_xticklabels([f"x={i}" for i in range(5)],fontsize=8)
    ax.set_yticks(np.arange(.5,5,1)); ax.set_yticklabels([f"y={i}" for i in range(5)],fontsize=8)
    sm=ScalarMappable(norm=norm,cmap=_CELL_CMAP); sm.set_array([])
    plt.colorbar(sm,ax=ax,fraction=0.04,pad=0.02,label="V(s)")


def plot_p2_vi_pi(V_vi,pol_vi,V_pi,pol_pi):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,7))
    fig.suptitle("Part 2 Task 1 — VI vs PI: Value Functions & Optimal Policies",
                 fontsize=13,fontweight="bold")
    _draw_cell(ax1,pol_vi,V_vi); ax1.set_title("Value Iteration",fontsize=11,fontweight="bold",pad=8)
    _draw_cell(ax2,pol_pi,V_pi); ax2.set_title("Policy Iteration",fontsize=11,fontweight="bold",pad=8)
    plt.tight_layout()
    plt.savefig("part2_vi_pi_comparison.png",dpi=160,bbox_inches="tight"); plt.close()
    print("  Saved: part2_vi_pi_comparison.png")


def plot_p2_convergence(vi_d):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
    fig.suptitle("Part 2 Task 1A — Value Iteration Convergence",fontsize=13,fontweight="bold")
    iters=range(1,len(vi_d)+1)
    ax1.plot(iters,vi_d,"b-o",ms=4,lw=1.5)
    ax1.axhline(THETA,color="red",ls="--",lw=1.2,label=f"theta={THETA}")
    ax1.set_xlabel("Iteration");ax1.set_ylabel("Max |DeltaV|")
    ax1.set_title("Linear Scale");ax1.legend();ax1.grid(alpha=0.35)
    ax2.semilogy(iters,vi_d,"b-o",ms=4,lw=1.5)
    ax2.axhline(THETA,color="red",ls="--",lw=1.2,label=f"theta={THETA}")
    ax2.set_xlabel("Iteration");ax2.set_ylabel("Max |DeltaV| (log)")
    ax2.set_title("Log Scale");ax2.legend();ax2.grid(alpha=0.35,which="both")
    plt.tight_layout()
    plt.savefig("part2_vi_convergence.png",dpi=160,bbox_inches="tight"); plt.close()
    print("  Saved: part2_vi_convergence.png")


def _ma(d,w=100):
    return np.array([np.mean(d[max(0,i-w+1):i+1]) for i in range(len(d))])


def plot_p2_learning(mc_r,mc_l,ql_r,ql_l):
    fig=plt.figure(figsize=(16,12))
    fig.suptitle("Part 2 Tasks 2 & 3 — Reinforcement Learning Curves",
                 fontsize=14,fontweight="bold")
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.38,wspace=0.35)
    eps=np.arange(1,len(mc_r)+1)

    # Panel 1: rewards
    ax1=fig.add_subplot(gs[0,:2])
    ax1.plot(eps,mc_r,alpha=0.12,color="steelblue",lw=0.6)
    ax1.plot(eps,_ma(mc_r),color="steelblue",lw=2.2,label="MC (100-ep avg)")
    ax1.plot(eps,ql_r,alpha=0.12,color="tomato",lw=0.6)
    ax1.plot(eps,_ma(ql_r),color="tomato",lw=2.2,label="Q-Learning (100-ep avg)")
    ax1.axhline(0,color="black",ls="--",lw=0.8,alpha=0.4)
    ax1.set_title("Total Reward per Episode",fontweight="bold")
    ax1.set_xlabel("Episode");ax1.set_ylabel("Reward")
    ax1.legend(fontsize=9);ax1.grid(alpha=0.3)

    # Panel 2: reward histogram (last 2000)
    ax2=fig.add_subplot(gs[0,2])
    ax2.hist(mc_r[-2000:],bins=30,alpha=0.65,color="steelblue",label="MC last 2000")
    ax2.hist(ql_r[-2000:],bins=30,alpha=0.65,color="tomato",   label="QL last 2000")
    ax2.set_title("Reward Distribution\n(last 2000 eps)",fontweight="bold")
    ax2.set_xlabel("Reward");ax2.set_ylabel("Frequency")
    ax2.legend(fontsize=8);ax2.grid(alpha=0.3)

    # Panel 3: episode lengths
    ax3=fig.add_subplot(gs[1,:2])
    ax3.plot(eps,_ma(mc_l),color="steelblue",lw=2.2,label="MC (100-ep avg length)")
    ax3.plot(eps,_ma(ql_l),color="tomato",   lw=2.2,label="Q-Learning (100-ep avg length)")
    ax3.set_title("Episode Length (steps to goal)",fontweight="bold")
    ax3.set_xlabel("Episode");ax3.set_ylabel("Steps")
    ax3.legend(fontsize=9);ax3.grid(alpha=0.3)

    # Panel 4: early convergence zoom
    ax4=fig.add_subplot(gs[1,2])
    ax4.plot(eps[:3000],_ma(mc_r)[:3000],color="steelblue",lw=2,label="MC")
    ax4.plot(eps[:3000],_ma(ql_r)[:3000],color="tomato",   lw=2,label="Q-Learning")
    ax4.axhline(-5,color="green",ls="--",lw=1.2,label="Conv. threshold (-5)")
    ax4.set_title("Early Convergence (first 3000 eps)",fontweight="bold")
    ax4.set_xlabel("Episode");ax4.set_ylabel("Avg Reward")
    ax4.legend(fontsize=8);ax4.grid(alpha=0.3)

    plt.savefig("part2_learning_curves.png",dpi=160,bbox_inches="tight"); plt.close()
    print("  Saved: part2_learning_curves.png")


def plot_p2_policy_compare(pol_vi,pol_mc,pol_ql):
    flat={s:0.0 for s in ALL_STATES}
    fig,axes=plt.subplots(1,3,figsize=(17,6))
    fig.suptitle("Part 2 — Policy Comparison  (red = differs from VI)",
                 fontsize=13,fontweight="bold")
    _draw_cell(axes[0],pol_vi,flat,show_val=False)
    axes[0].set_title("Task 1: Value Iteration\n(Optimal)",fontsize=10,fontweight="bold",pad=6)
    for ax,pol,lbl in [
        (axes[1],pol_mc, f"Task 2: Monte Carlo\n(eps={EPSILON}, {MC_EPISODES:,} eps)"),
        (axes[2],pol_ql, f"Task 3: Q-Learning\n(eps={EPSILON}, alpha={ALPHA}, {QL_EPISODES:,} eps)"),
    ]:
        _draw_cell(ax,pol,flat,show_val=False)
        ax.set_title(lbl,fontsize=10,fontweight="bold",pad=6)
        for x in range(5):
            for y in range(5):
                s=(x,y)
                if s in ALL_STATES and pol.get(s)!=pol_vi.get(s):
                    ax.add_patch(plt.Rectangle([x,y],1,1,fc="none",
                                               ec="red",lw=2.5,zorder=10))
    plt.tight_layout()
    plt.savefig("part2_policy_comparison.png",dpi=160,bbox_inches="tight"); plt.close()
    print("  Saved: part2_policy_comparison.png")


def plot_p2_qtable(Q_ql):
    fig,axes=plt.subplots(2,2,figsize=(13,11))
    fig.suptitle("Part 2 Task 3 — Q-Table Heatmaps per Action (Q-Learning)",
                 fontsize=13,fontweight="bold")
    for ax,action in zip(axes.flat,ACTIONS):
        grid=np.full((5,5),np.nan)
        for x in range(5):
            for y in range(5):
                s=(x,y)
                if s not in ROADBLOCKS and s!=GOAL:
                    grid[4-y][x]=Q_ql.get((s,action),0.0)
        im=ax.imshow(grid,cmap="RdYlGn",aspect="equal",
                     vmin=-15,vmax=10,interpolation="nearest")
        for x in range(5):
            for y in range(5):
                gy=4-y; s=(x,y)
                if s==GOAL:
                    ax.add_patch(plt.Rectangle([x-.5,gy-.5],1,1,fc="#388E3C",zorder=2))
                    ax.text(x,gy,"G",ha="center",va="center",fontsize=11,
                            color="white",fontweight="bold",zorder=3)
                elif s in ROADBLOCKS:
                    ax.add_patch(plt.Rectangle([x-.5,gy-.5],1,1,fc="#263238",zorder=2))
                    ax.text(x,gy,"X",ha="center",va="center",fontsize=12,color="#90A4AE",zorder=3)
                else:
                    v=Q_ql.get((s,action),0.0)
                    ax.text(x,gy,f"{v:.2f}",ha="center",va="center",fontsize=8,zorder=3)
        plt.colorbar(im,ax=ax,label="Q(s,a)")
        ax.set_title(f"Action: {ARROWS[action]}  {action}",fontsize=11,fontweight="bold")
        ax.set_xticks(range(5));ax.set_xticklabels([f"x={i}" for i in range(5)])
        ax.set_yticks(range(5));ax.set_yticklabels([f"y={4-i}" for i in range(5)])
    plt.tight_layout()
    plt.savefig("part2_qtable_heatmaps.png",dpi=160,bbox_inches="tight"); plt.close()
    print("  Saved: part2_qtable_heatmaps.png")


def plot_p2_evolution(mc_s,ql_s,pol_vi):
    cps=[500,1000,2000,5000,10000]
    fig,axes=plt.subplots(2,5,figsize=(22,9))
    fig.suptitle("Part 2 — Policy Evolution During Training  (red = differs from VI)",
                 fontsize=13,fontweight="bold")
    for row,(lbl,snaps) in enumerate([("MC",mc_s),("Q-Learning",ql_s)]):
        for col,ep in enumerate(cps):
            ax=axes[row][col]; pol=snaps.get(ep,{})
            for x in range(5):
                for y in range(5):
                    s=(x,y)
                    if s==GOAL:
                        ax.add_patch(plt.Rectangle([x,y],1,1,fc="#388E3C",ec="black",lw=1))
                        ax.text(x+.5,y+.5,"G",ha="center",va="center",
                                fontsize=10,color="white",fontweight="bold")
                    elif s in ROADBLOCKS:
                        ax.add_patch(plt.Rectangle([x,y],1,1,fc="#263238",ec="black",lw=1))
                        ax.text(x+.5,y+.5,"X",ha="center",va="center",fontsize=12,color="#90A4AE")
                    else:
                        diff=pol.get(s)!=pol_vi.get(s)
                        ax.add_patch(plt.Rectangle([x,y],1,1,
                                                   fc="#ffcdd2" if diff else "white",
                                                   ec="red" if diff else "black",
                                                   lw=2.0 if diff else 1.0))
                        ax.text(x+.5,y+.5,ARROWS.get(pol.get(s,"?"),"?"),
                                ha="center",va="center",fontsize=18)
            ag=sum(1 for s in ALL_STATES if pol.get(s)==pol_vi.get(s))
            ax.set_xlim(0,5);ax.set_ylim(0,5)
            ax.set_xticks([]);ax.set_yticks([])
            ax.set_title(f"{lbl}\nEp {ep:,}  {ag}/{len(ALL_STATES)}",
                         fontsize=9,fontweight="bold")
    plt.tight_layout()
    plt.savefig("part2_policy_evolution.png",dpi=150,bbox_inches="tight"); plt.close()
    print("  Saved: part2_policy_evolution.png")


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # ── Part 1 ────────────────────────────────────────────────
    t1r, t2r, t3r = run_part1()
    print("\n-- Part 1 Plots --------------------------------------------------")
    plot_p1_map(t1r, t2r, t3r)
    plot_p1_stats(t1r, t2r, t3r)
    plot_p1_profiles(t1r, t2r, t3r)

    # ── Part 2 ────────────────────────────────────────────────
    (V_vi,pol_vi,vi_d, V_pi,pol_pi,pi_out,
     pol_mc,V_mc,Q_mc,mc_r,mc_l,mc_s,
     pol_ql,V_ql,Q_ql,ql_r,ql_l,ql_s,
     mc_ag,ql_ag,all3) = run_part2()

    print("\n-- Part 2 Plots --------------------------------------------------")
    plot_p2_vi_pi(V_vi,pol_vi,V_pi,pol_pi)
    plot_p2_convergence(vi_d)
    plot_p2_learning(mc_r,mc_l,ql_r,ql_l)
    plot_p2_policy_compare(pol_vi,pol_mc,pol_ql)
    plot_p2_qtable(Q_ql)
    plot_p2_evolution(mc_s,ql_s,pol_vi)

    # ── Final summary ─────────────────────────────────────────
    print(f"\n{'='*64}")
    print("  ALL COMPLETE")
    print(f"{'='*64}")
    print("  main_results.txt         (all printed output)")
    print("  part1_path_map.png       (NYC map 3 paths)")
    print("  part1_algorithm_stats.png (bar charts: dist/energy/nodes/time)")
    print("  part1_path_profiles.png  (cumulative dist & energy per step)")
    print("  part2_vi_pi_comparison.png (VI vs PI side-by-side)")
    print("  part2_vi_convergence.png (max delta-V per iteration)")
    print("  part2_learning_curves.png (MC + QL rewards/lengths/convergence)")
    print("  part2_policy_comparison.png (VI | MC | QL with red diffs)")
    print("  part2_qtable_heatmaps.png (Q-values per action heatmap)")
    print("  part2_policy_evolution.png (policy snapshots at checkpoints)")
    print(f"{'='*64}")

    sys.stdout = sys.__stdout__
    _log.close()