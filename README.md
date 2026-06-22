# cities-project

A two-patch metapopulation model of population displacement after a natural
disaster, motivated by Hurricane Katrina and the migration of displaced
residents from New Orleans to Atlanta. The model is used to study how a
**rebuilding effort** in the receiving city absorbs the shock — how much
displacement and mortality it prevents, and how best to invest in it.

## Model

State variables: populations `N_K`, `N_A` and carrying capacities `K_K`, `K_A`
for New Orleans (`K`) and Atlanta (`A`). Each city grows logistically toward its
own carrying capacity. For `t >= t_s` (after the shock):

```
dN_K/dt = r_K N_K (1 - N_K/K_K) - F
dN_A/dt = r_A N_A (1 - N_A/K_A) + F
dK_K/dt = 0                              # New Orleans permanently lower
dK_A/dt = u (1 - K_A / K_A_max)          # Atlanta builds capacity (effort u)
```

The net migration flux from K to A, in the default `gradient` mode, is

```
F = m * N_source * (phi_K - phi_A)       # source = the more-crowded city, phi = N/K
```

so people leave whichever city is more crowded, toward the one with room. A
second `occupancy` mode (`F` from each city emitting `m N^2 / K`) is switchable
via `migration_mode`.

The shock at `t = t_s` is an instantaneous reset `K_K -> (1 - delta_K) * K_K0`.

### The two policy levers

Atlanta's recovery effort has two distinct levers:

- **`u`** — the *rate* at which capacity is built (mobilization speed).
- **`K_A_max`** — the *amount* of capacity that can ultimately be built (ceiling).

Raising `u` builds the same capacity *sooner*; raising `K_A_max` builds *more*.

## Key quantities

- **Displacement** — cumulative migration K → A (people who survive by leaving).
- **Crowding deaths** — net population lost while a city is over capacity,
  `integral of (-G_i)+` where `G_i = r_i N_i (1 - phi_i)` is net growth
  (births minus density-dependent deaths). Migration conserves people, so the
  total population only changes through this term.
- **Population dip** — trough depth `T0 - min(total population)`.
- **Lives saved per unit effort** — marginal `-d(deaths)/du` and average.
- **Optimal investment** — for a budget `B = c_u*u + c_K*dK`, the split that
  minimizes deaths (marginal lives per dollar equalized across levers).

## Install

```bash
pip install numpy scipy matplotlib
```

## Run

```bash
python src/katrina_atlanta_metapop.py
```

All figures are written to `figures/` (resolved relative to the script, so it
works from any working directory). Model parameters are set in the `params`
dict at the top of the script.

| figure | contents |
|---|---|
| `katrina_atlanta_dynamics.png` | populations, capacities, occupancy, migration flux |
| `katrina_atlanta_timeparams.png` | time-varying carrying capacities + build rate |
| `katrina_atlanta_absorption.png` | deaths, lives saved per unit effort, fate of the shock |
| `katrina_atlanta_sweep_u.png` | sweep over build effort `u` |
| `katrina_atlanta_sweep_delta.png` | sweep over shock severity `delta_K` |
| `katrina_atlanta_sweep2d.png` | effort × shock interaction |
| `katrina_atlanta_levers.png` | speed (`u`) vs amount (`K_A_max`): which absorbs the shock |
| `katrina_atlanta_optimum.png` | budget-constrained optimal allocation + efficient frontier |
| `katrina_atlanta_opt_vs_severity.png` | how the optimal split changes with shock severity |

## Selected findings

- With gradient migration + logistic growth, the long-run state is "each city at
  its own capacity" (`phi -> 1`); migration governs the *transient*, capacities
  govern the *destination*.
- Build effort `u` has sharp diminishing returns (a knee around `u ~ 3`), because
  crowding deaths are front-loaded right after the shock.
- Speed and amount are **complements**: capacity built too slowly, or speed with
  no room to build into, both fail. Deaths are minimized only when both are
  adequate.
- Under equal costs, the optimal speed/amount split is ~50/50 and is **robust to
  shock severity** — severity sets *how much* to spend and the death toll, not
  *how to split* the budget. The cost ratio is what shifts the split.
