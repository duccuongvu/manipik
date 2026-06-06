# IIWA 14 — Analytic Inverse Kinematics

Closed-form inverse kinematics for the **KUKA IIWA 14**, a 7-DOF redundant arm.
The solver enumerates **every** joint configuration that reaches a target TCP
pose and respects the joint limits — it is **seed-independent** and does not
return a single "nearest" answer. Picking among the solutions (smoothest
motion, collision-free, ...) is left to the caller.

Two equivalent backends are provided:

| Backend | File | Notes |
|---------|------|-------|
| Python / NumPy | [`iiwa_ik.py`](iiwa_ik.py) | Reference implementation, easy to read |
| C++ / Eigen    | [`iiwa_ik.cpp`](iiwa_ik.cpp) | pybind11 module, ~12–70× faster |

Both return the **identical solution set** (verified in `ik_benchmark.py`).

---

## Quick start

```python
import numpy as np
from iiwa_ik import IiwaIK as IK

x_d = np.eye(4)                       # 4x4 SE(3) TCP target, robot base frame
x_d[:3, 3] = [0.45, 0.0, 0.6]

solutions = IK.solve_all(x_d, n_psi=20)   # (N, 7) array — one row per config
print(len(solutions), "solutions")
```

The IK takes exactly two inputs: the **end-effector pose** `x_d` and `n_psi`,
the number of redundant arm-angle samples (see [Tuning](#how-many-solutions-tuning-the-solver)).

The C++ backend exposes the same call:

```python
import IiwaIKCpp
solutions = IiwaIKCpp.solve_all(x_d, 20)   # (N, 7) numpy array
```

---

## Layout

```
iiwa/
├── iiwa_ik.hpp           # analytic IK core (header-only C++ / Eigen)
├── iiwa_ik.cpp           # pybind11 wrapper around iiwa_ik.hpp -> IiwaIKCpp
├── iiwa_ik_example.cpp   # standalone C++ program that solves IK
├── CMakeLists.txt        # builds iiwa_ik_example
├── iiwa_ik.py            # analytic IK (Python), returns all solutions
├── iiwa_fk.py            # MuJoCo forward kinematics (ground truth for tests)
├── build_cpp.py          # compile iiwa_ik.cpp -> IiwaIKCpp module
├── ik_benchmark.py       # timing + accuracy over 10 random poses
├── iiwa_ik_explanation.md# algorithm write-up
└── examples/
    ├── 01_solve_all.py       # list every solution for a pose
    ├── 02_pick_a_solution.py # selection policies on top of the full set
    └── 03_fk_roundtrip.py    # FK ground-truth + Python/C++ agreement
```

---

## Building the C++ module

The build uses the active Python's `pybind11` and the system Eigen
(`/usr/include/eigen3`):

```bash
python build_cpp.py
```

or directly:

```bash
g++ -O2 -shared -fPIC -std=c++17 \
    $(python -m pybind11 --includes) -I/usr/include/eigen3 \
    iiwa_ik.cpp -o IiwaIKCpp$(python3-config --extension-suffix)
```

> The build and benchmarks need the `manipsim` conda env (MuJoCo + pybind11).
> Install pybind11 there once with `pip install pybind11` if missing.

---

## Standalone C++ usage

The solver core is header-only (`iiwa_ik.hpp`), so a pure-C++ program can call
it without Python. `iiwa_ik_example.cpp` shows the minimal usage:

```cpp
#include "iiwa_ik.hpp"
using namespace iiwa_ik;

Mat4d x_d = Mat4d::Identity();
x_d(0,3) = 0.40; x_d(1,3) = -0.20; x_d(2,3) = 0.65;

std::vector<Vec7d> sols = solve_all(x_d, /*n_psi=*/20);   // all feasible configs
```

Build and run it with CMake (needs Eigen3, e.g. `sudo apt install libeigen3-dev`):

```bash
cmake -B build
cmake --build build
./build/iiwa_ik_example
```

---

## Benchmarks

The benchmark runs on **10 random reachable poses** — no arguments. It reports
per-pose #solutions, Python & C++ solve time, speedup, and worst FK residual.

```bash
python build_cpp.py       # build the C++ module (once)
python ik_benchmark.py    # run the benchmark
```

Representative result (10 poses, `n_psi=20`):

```
─────────────────────────────────────────────────────────
Analytic IK benchmark  |  10 random poses

  pose   #sols      Python         C++   speedup      fk_err
  --------------------------------------------------------
     0      72      970µs         4µs    254.7x    1.80e-09
     1      88      973µs         5µs    203.0x    2.03e-09
     2     140     1027µs         5µs    224.6x    2.45e-09
     3     144     1000µs         5µs    216.8x    3.26e-09
     4     124     1040µs         5µs    208.9x    2.28e-09
     5     132     1047µs         5µs    216.8x    2.30e-09
     6     128     1016µs         5µs    207.5x    2.33e-09
     7     117     1033µs         5µs    207.9x    2.56e-09
     8     114     1036µs         4µs    238.0x    3.17e-09
     9     130     1057µs         5µs    205.2x    2.97e-09

  Python : 1020µs / solve
  C++    : 5µs / solve  (217.2x faster)
  fk_err : 3.26e-09 (worst over all poses & solutions)
─────────────────────────────────────────────────────────
```

Every returned solution round-trips through MuJoCo FK to within ~3e-9
(position + orientation residual). The speedup of the C++ backend varies with
the host NumPy build (≈12× on the `manipsim` env, ≈70× here).

---

## API

```
IiwaIK.solve_all(x_d, n_psi=20, param=RobotParams()) -> np.ndarray
    x_d   : 4x4 SE(3) TCP target pose in the robot base frame
    n_psi : number of arm-angle samples; [0, 2π) is split into n_psi intervals
    param : RobotParams (link lengths + joint limits)
    return: (N, 7) array of feasible joint configs [rad]; empty (0,7) if unreachable

IiwaIKCpp.solve_all(x_d, n_psi=20) -> np.ndarray     # same, C++ backend
```

---

## How many solutions? Tuning the solver

### `n_psi` — the only redundancy knob

The IIWA's single redundant degree of freedom is the **arm angle ψ**: the arm
can swivel around the shoulder→wrist axis while the TCP stays fixed. `solve_all`
samples ψ at `n_psi` evenly-spaced points on `[0, 2π)` — i.e. it splits the
circle into `n_psi` intervals. Each sample produces a family of joint solutions,
so the number of returned configurations scales **≈ linearly** with `n_psi`:

```python
IK.solve_all(x_d, n_psi=10)   # coarse  — fewest solutions, fastest
IK.solve_all(x_d, n_psi=20)   # default
IK.solve_all(x_d, n_psi=40)   # dense   — most solutions, slowest
```

| `n_psi` | step | typical #solutions | relative time |
|--------:|-----:|-------------------:|--------------:|
| 10 | 36° | ~60–120  | fastest |
| 20 | 18° | ~120–150 | 1× (default) |
| 40 |  9° | ~240–300 | ~2× |

> ψ is a *continuous* freedom, so there is no finite "complete" set — `n_psi`
> simply trades resolution for speed. **Want fewer solutions? Lower `n_psi`.**

### Joint limits

Limits live in `RobotParams` (`iiwa_ik.py`) and the `q_upper`/`q_lower` arrays
in `iiwa_ik.cpp`. Tightening them rejects more candidates → fewer solutions:

```python
from iiwa_ik import IiwaIK as IK, RobotParams
import numpy as np

p = RobotParams()
p.q_limit_upper = np.radians([120, 90, 120, 90, 120, 90, 120])
p.q_limit_lower = -p.q_limit_upper
solutions = IK.solve_all(x_d, n_psi=20, param=p)   # constrained → fewer solutions
```

### Just want one solution?

`solve_all` intentionally returns them all. To pick one, filter in the caller —
e.g. nearest to the current joint state (see
[`examples/02_pick_a_solution.py`](examples/02_pick_a_solution.py)):

```python
sols = IK.solve_all(x_d)
q = sols[np.argmin(np.linalg.norm(sols - q_current, axis=1))]
```

---

## Theory

The IIWA 14 is a 7-DOF arm solving a 6-DOF task, so it has **one** redundant
degree of freedom. We resolve it with the *arm-angle* parameterization of
Shimizu et al. (2008): the elbow is free to swivel on a circle about the
shoulder→wrist axis, and that swivel angle is the scalar $\psi$.

### Notation

The kinematic chain collapses into four link offsets along $z$:

$$
d_{bs} = d_1 + d_2,\quad
d_{se} = d_3 + d_4,\quad
d_{ew} = d_5 + d_6,\quad
d_{wt} = d_7 + d_8 .
$$

The target TCP pose in the base frame is

$$
T_d=\begin{bmatrix} R_{07} & \mathbf{p} \\ \mathbf{0}^\top & 1 \end{bmatrix}\in SE(3),
\qquad R_{07}\in SO(3),\ \mathbf{p}\in\mathbb{R}^3 .
$$

### 1. Shoulder–wrist vector and the elbow angle

Strip the fixed base and tool offsets to get the vector from shoulder to wrist:

$$
\mathbf{x}_{sw} = \mathbf{p} - \begin{bmatrix}0\\0\\d_{bs}\end{bmatrix}
                              - R_{07}\begin{bmatrix}0\\0\\d_{wt}\end{bmatrix},
\qquad
\hat{\mathbf{u}}_{sw} = \frac{\mathbf{x}_{sw}}{\lVert \mathbf{x}_{sw}\rVert}.
$$

The shoulder–elbow–wrist triangle fixes the elbow joint by the law of cosines:

$$
\cos q_4 = \frac{\lVert \mathbf{x}_{sw}\rVert^{2} - d_{se}^{2} - d_{ew}^{2}}
                {2\,d_{se}\,d_{ew}} .
$$

If $\lvert\cos q_4\rvert > 1$ the target is **unreachable** (triangle inequality
violated) and the solver returns the empty set.

### 2. The redundancy circle (arm angle $\psi$)

With the elbow fixed, the arm can rotate rigidly about $\hat{\mathbf{u}}_{sw}$.
Let $R_{03}^{0}$ be the reference shoulder orientation at $\psi=0$ and
$\left[\hat{\mathbf{u}}_{sw}\right]_\times$ the skew-symmetric (cross-product)
matrix of $\hat{\mathbf{u}}_{sw}$. By Rodrigues' rotation formula the shoulder
orientation as a function of $\psi$ is affine in $(\sin\psi,\cos\psi)$:

$$
R_{03}(\psi) = A_s\sin\psi + B_s\cos\psi + C_s,
$$

$$
A_s = \left[\hat{\mathbf{u}}_{sw}\right]_\times R_{03}^{0},\quad
B_s = -\left[\hat{\mathbf{u}}_{sw}\right]_\times^{2} R_{03}^{0},\quad
C_s = \hat{\mathbf{u}}_{sw}\hat{\mathbf{u}}_{sw}^{\top} R_{03}^{0}.
$$

The wrist matrices follow from the known relative transform $R_{34}$:

$$
X_w = R_{34}^{\top}\,X_s^{\top}\,R_{07},\qquad X\in\{A,B,C\}.
$$

### 3. Closed-form joints for a given $\psi$

Each joint is then read directly from entries of these matrices. The shoulder
joints come from the $A_s,B_s,C_s$ block, e.g.

$$
q_1 = \operatorname{atan2}\!\big(\,-A_{s,11}s_\psi - B_{s,11}c_\psi - C_{s,11},\;
                                  -A_{s,01}s_\psi - B_{s,01}c_\psi - C_{s,01}\big),
$$

$$
\cos q_2 = -A_{s,21}s_\psi - B_{s,21}c_\psi - C_{s,21},
$$

with $s_\psi=\sin\psi,\ c_\psi=\cos\psi$ (0-based matrix indices). The wrist
joints $q_5,q_6,q_7$ come identically from $A_w,B_w,C_w$, and $q_3,q_4$ from the
elbow geometry. A sample is discarded when the $\arccos$ arguments for $q_2$ or
$q_6$ fall outside $[-1,1]$.

### 4. Binary redundancy parameters

Beyond the arm angle $\psi$, the IIWA has three binary redundancy parameters
(Vu et al. 2023, eq. 7 & 13):

- $j_e \in \{-1,+1\}$ — **elbow**: down ($q_4<0$) or up ($q_4>0$)
- $j_s \in \{-1,+1\}$ — **shoulder**: flips the sign of $q_2$; coupled joints $q_1,q_3$ shift by $\pm\pi$
- $j_w \in \{-1,+1\}$ — **wrist**: flips the sign of $q_6$; coupled joints $q_5,q_7$ shift by $\pm\pi$

Each branch uses its own reference shoulder orientation (different $\gamma$ in eq. 11b)
and its own $R_{34}(q_4)$, so all eight branches are enumerated separately.

### 5. Sampling and filtering

$\psi$ is **continuous**, so we sample it uniformly:

$$
\psi_k = \frac{2\pi k}{n_\psi},\qquad k = 0,1,\dots,n_\psi-1 .
$$

Total candidates per pose: $2\,(j_e)\times n_\psi\times 2\,(j_s)\times 2\,(j_w) = 8\,n_\psi$.
Each is wrapped into $(-\pi,\pi]$ and kept only if it satisfies the joint limits

$$
\mathbf{q}_{\min} \le \mathbf{q} \le \mathbf{q}_{\max}.
$$

Duplicates are removed, giving the final $(N,7)$ set of feasible configurations.

> **References:**
> - M. Shimizu et al., *"Analytical Inverse Kinematic Computation for 7-DOF Redundant Manipulators With Joint Limits,"* IEEE Trans. Robotics, 24(5), 2008.
> - M.N. Vu et al., *"Machine learning-based framework for optimally solving the analytical inverse kinematics for redundant manipulators,"* Mechatronics 91, 2023.
