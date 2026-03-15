# CZ3005 / SC3000 Artificial Intelligence — Lab Assignment 1

A single-file Python implementation covering two major AI topics:
- **Part 1**: Shortest path search on the NYC road network with an energy budget constraint
- **Part 2**: Grid World MDP solved via dynamic programming and reinforcement learning

---

## Requirements

- Python 3.8+
- `numpy`
- `matplotlib`

Install dependencies:
```bash
pip install numpy matplotlib
```

---

## How to Run

```bash
python main.py
```

That's it. The script will:
1. Load the NYC road network from the JSON files
2. Run all Part 1 search algorithms
3. Run all Part 2 RL algorithms
4. Save all results to `main_results.txt`
5. Generate 9 PNG plots in the same directory

> **Note:** Do not use `pythonw.exe` — use `python.exe` so terminal output is visible.

---

## Input Files (required in same directory)

| File | Description |
|------|-------------|
| `G.json` | Adjacency list of the NYC road network (264,346 nodes) |
| `Coord.json` | (x, y) coordinates for each node |
| `Dist.json` | Edge distances |
| `Cost.json` | Edge energy costs |

---

## Output Files

| File | Description |
|------|-------------|
| `main_results.txt` | Full printed output of all results |
| `part1_path_map.png` | NYC map with all 3 paths overlaid |
| `part1_algorithm_stats.png` | Bar charts: distance / energy / nodes expanded / time |
| `part1_path_profiles.png` | Cumulative distance & energy along each path |
| `part2_vi_pi_comparison.png` | Value Iteration vs Policy Iteration side-by-side |
| `part2_vi_convergence.png` | Max delta-V per iteration (linear + log scale) |
| `part2_learning_curves.png` | MC & Q-Learning reward/length curves + convergence zoom |
| `part2_policy_comparison.png` | VI vs MC vs Q-Learning policies (red = differs from VI) |
| `part2_qtable_heatmaps.png` | Q-value heatmap per action (Q-Learning) |
| `part2_policy_evolution.png` | Policy snapshots at episodes 500/1000/2000/5000/10000 |

---

## Configuration Constants (top of `main.py`)

| Constant | Value | Description |
|----------|-------|-------------|
| `SOURCE` | `"1"` | Start node |
| `TARGET` | `"50"` | Goal node |
| `ENERGY_BUDGET` | `287,932` | Max energy allowed (Parts 2 & 3) |
| `GAMMA` | `0.9` | Discount factor for MDP |
| `EPSILON` | `0.1` | Exploration rate for MC & Q-Learning |
| `ALPHA` | `0.1` | Learning rate for Q-Learning |
| `MC_EPISODES` | `10,000` | Monte Carlo training episodes |
| `QL_EPISODES` | `10,000` | Q-Learning training episodes |
| `THETA` | `1e-6` | Convergence threshold for VI/PI |

---

## Part 1 — NYC Road Network Results

### Problem
Find the shortest path from node **1** to node **50** across a 264,346-node NYC road network, with and without an energy budget constraint of **287,932**.

### Task 1 — Dijkstra (No Energy Constraint)

| Metric | Value |
|--------|-------|
| Shortest Distance | 148,648.64 |
| Total Energy Cost | 294,853 |
| Nodes in Path | 122 |
| Nodes Expanded | 5,304 |
| Time | 0.024s |

**Path:**
```
1->1363->1358->1357->1356->1276->1273->1277->1269->1267->1268->1284->1283->
1282->1255->1253->1260->1259->1249->1246->963->964->962->1002->952->1000->
998->994->995->996->987->988->979->980->969->977->989->990->991->2369->2366->
2340->2338->2339->2333->2334->2329->2029->2027->2019->2022->2000->1996->1997->
1993->1992->1989->1984->2001->1900->1875->1874->1965->1963->1964->1923->1944->
1945->1938->1937->1939->1935->1931->1934->1673->1675->1674->1837->1671->1828->
1825->1817->1815->1634->1814->1813->1632->1631->1742->1741->1740->1739->1591->
1689->1585->1584->1688->1579->1679->1677->104->5680->5418->5431->5425->5424->
5422->5413->5412->5411->66->5392->5391->5388->5291->5278->5289->5290->5283->
5284->5280->50
```
> Energy exceeds budget (294,853 > 287,932) — no constraint applied here.

---

### Task 2 — UCS (Uninformed, Energy Budget = 287,932)

| Metric | Value |
|--------|-------|
| Shortest Distance | 150,784.61 |
| Total Energy Cost | 287,931 |
| Nodes in Path | 123 |
| Nodes Expanded | 5,099 |
| Time | 0.014s |

**Path:** `1->1363->1358->1357->1356 -> ... -> 5282->5283->5284->5280->50`

---

### Task 3 — A* (Informed, Euclidean Heuristic, Energy Budget = 287,932)

| Metric | Value |
|--------|-------|
| Shortest Distance | 150,784.61 |
| Total Energy Cost | 287,931 |
| Nodes in Path | 123 |
| Nodes Expanded | 1,573 |
| Time | 0.006s |

**Path:** `1->1363->1358->1357->1356 -> ... -> 5282->5283->5284->5280->50`

> UCS and A\* find the **identical optimal path** — A\* simply reaches it by expanding 69.2% fewer nodes, demonstrating the efficiency gain from the Euclidean heuristic.

---

### Part 1 Summary

| Algorithm | Distance | Energy | Nodes Expanded | Time |
|-----------|----------|--------|----------------|------|
| Dijkstra (Task 1) | 148,648.64 | 294,853 | 5,304 | 0.052s |
| UCS (Task 2) | 150,784.61 | 287,931 | 5,099 | 0.014s |
| A* (Task 3) | 150,784.61 | 287,931 | 1,573 | 0.006s |

**A\* vs UCS:** distance `0.0%` | energy `0.0%` | nodes expanded `-69.2%`

> A\* and UCS find the **same optimal path** within the energy budget. A\* is faster because the Euclidean heuristic guides it directly toward the goal, expanding 69% fewer nodes with no loss in solution quality.

---

## Part 2 — Grid World MDP & Reinforcement Learning Results

### Environment
- 5×5 grid world, start at `(0,0)`, goal at `(4,4)`
- Roadblocks at `(2,1)` and `(2,3)`
- Stochastic transitions: 80% intended direction, 10% each perpendicular
- Reward: `+10` on reaching goal, `-1` otherwise
- Discount factor γ = 0.9

---

### Task 1A — Value Iteration

- Converged in **28 iterations** (delta_final = 5.75e-07) in **0.005s**

**Value Function:**
```
y=4 |   2.71   4.65   6.93   9.28  GOAL
y=3 |   1.31   2.71  BLOK    7.16   9.28
y=2 |   0.16   1.56   3.22   5.05   6.74
y=1 |  -0.97   0.04  BLOK    3.35   4.57
y=0 |  -1.90  -0.90   0.25   1.68   2.68
```

**Optimal Policy:**
```
y=4 |  →  →  →  →  G
y=3 |  ↑  ↑  X  ↑  ↑
y=2 |  →  →  →  ↑  ↑
y=1 |  ↑  ↑  X  ↑  ↑
y=0 |  →  →  →  ↑  ↑
```

---

### Task 1B — Policy Iteration

- Converged in **5 outer iterations**, 171 total inner sweeps, in **0.011s**
- Value function and policy **identical** to Value Iteration (22/22 states match)
- Max |V_VI − V_PI| = 4.75e-07 (numerical precision only)

---

### Task 2 — First-Visit Monte Carlo Control

- 10,000 episodes, ε = 0.1
- Completed in **0.4s**
- Final avg reward (last 100 eps): **-0.010** | avg episode length: **11.0 steps**
- Converged (avg reward ≥ -5) at episode **153**
- Policy agreement with VI: **14/22 states (64%)**

**Learned Policy:**
```
y=4 |  →  →  →  →  G
y=3 |  ↑  ↑  X  ↑  ↑
y=2 |  ↑  ↑  →  →  ↑
y=1 |  ↑  ↑  X  ↑  ↑
y=0 |  ↑  ↑  ←  →  ←
```

---

### Task 3 — Tabular Q-Learning

- 10,000 episodes, ε = 0.1, α = 0.1
- Completed in **0.4s**
- Final avg reward (last 100 eps): **-0.090** | avg episode length: **11.1 steps**
- Converged (avg reward ≥ -5) at episode **112**
- Policy agreement with VI: **20/22 states (91%)**

**Learned Policy:**
```
y=4 |  →  →  →  →  G
y=3 |  →  ↑  X  ↑  ↑
y=2 |  →  →  →  ↑  ↑
y=1 |  ↑  ↑  X  ↑  ↑
y=0 |  ↑  →  →  ↑  ↑
```

---

### Part 2 Summary

| Method | Type | Time | Agrees with VI | Convergence Episode |
|--------|------|------|----------------|---------------------|
| Value Iteration | Dynamic Programming | 0.005s | 22/22 (100%) | 28 iterations |
| Policy Iteration | Dynamic Programming | 0.011s | 22/22 (100%) | 5 outer iters |
| Monte Carlo | Model-Free RL | 0.4s | 14/22 (64%) | Episode 153 |
| Q-Learning | Model-Free RL | 0.4s | 20/22 (91%) | Episode 112 |

**Q-Learning converges ~41 episodes faster than Monte Carlo** because it updates Q-values at every step (TD bootstrapping), while Monte Carlo must wait for the full episode return.

---

## Project Structure

```
Lab 1/
├── main.py                     # All code
├── G.json                      # Road network graph
├── Coord.json                  # Node coordinates
├── Dist.json                   # Edge distances
├── Cost.json                   # Edge energy costs
├── main_results.txt            # Generated: full text output
├── part1_path_map.png          # Generated: NYC map
├── part1_algorithm_stats.png   # Generated: bar charts
├── part1_path_profiles.png     # Generated: cumulative profiles
├── part2_vi_pi_comparison.png  # Generated: VI vs PI
├── part2_vi_convergence.png    # Generated: convergence curve
├── part2_learning_curves.png   # Generated: RL learning curves
├── part2_policy_comparison.png # Generated: policy diff
├── part2_qtable_heatmaps.png   # Generated: Q-table
└── part2_policy_evolution.png  # Generated: policy snapshots
```
