#
# Created on Sat Jun 06 2026
#
# Copyright (c) 2026 Duc-Cuong Vu - vdcuong2002@gmail.com
#

#!/usr/bin/env python3
"""Example 2 — Choosing among the solutions the solver returns.

Because `solve_all` is seed-independent, *selection* is the caller's job. This
shows two common policies built on top of the full solution set:

  * nearest to the current joint state (smooth motion),
  * smallest total joint travel from a "home" posture.

Run from the repo root:
    python examples/02_pick_a_solution.py
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iiwa_ik import IiwaIK as IK  # noqa: E402

Q_HOME = np.array([0.0, -0.785, 0.0, -1.571, 0.0, 0.0, 0.0])


def nearest(solutions: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Solution with the smallest L2 joint distance to q_ref."""
    return solutions[int(np.argmin(np.linalg.norm(solutions - q_ref, axis=1)))]


def main() -> None:
    x_d = np.eye(4)
    x_d[:3, 3] = [0.45, 0.0, 0.6]

    solutions = IK.solve_all(x_d)
    if len(solutions) == 0:
        print("target unreachable")
        return

    print(f"{len(solutions)} feasible solutions\n")

    q_current = np.array([0.2, -0.5, 0.1, -1.4, 0.0, 0.3, 0.0])
    q_smooth = nearest(solutions, q_current)
    q_from_home = nearest(solutions, Q_HOME)

    print("nearest to current state :", np.array2string(np.degrees(q_smooth), precision=1))
    print("  joint travel from current:", round(np.linalg.norm(q_smooth - q_current), 3), "rad")
    print("nearest to home posture  :", np.array2string(np.degrees(q_from_home), precision=1))
    print("  joint travel from home   :", round(np.linalg.norm(q_from_home - Q_HOME), 3), "rad")


if __name__ == "__main__":
    main()
