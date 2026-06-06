#!/usr/bin/env python3
#
# Created on Sat Jun 06 2026
#
# Copyright (c) 2026 Duc-Cuong Vu - vdcuong2002@gmail.com
#

"""Example 1 — Solve a single target pose and list every feasible solution.

The analytic solver returns *all* joint configurations that reach the target;
it does not pick one for you. Here we just print them.

Run from the repo root:
    python examples/01_solve_all.py
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iiwa_ik import IiwaIK as IK  # noqa: E402


def rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def main() -> None:
    # 4x4 SE(3) target pose of the TCP in the robot base frame.
    x_d = np.eye(4)
    x_d[:3, :3] = rot_z(math.radians(30.0))
    x_d[:3, 3] = [0.40, -0.20, 0.65]

    # n_psi splits the redundant arm angle [0, 2π) into that many samples;
    # more samples -> more solutions. Try 10 vs 40 to see the count scale.
    for n_psi in (10, 20, 40):
        solutions = IK.solve_all(x_d, n_psi=n_psi)
        print(f"n_psi={n_psi:>2}  ->  {len(solutions)} feasible solutions")

    print(f"\ntarget position: {x_d[:3, 3]}")
    solutions = IK.solve_all(x_d, n_psi=20)
    for i, q in enumerate(solutions):
        print(f"  [{i:3d}]  {np.array2string(np.degrees(q), precision=1, suppress_small=True)} deg")


if __name__ == "__main__":
    main()
