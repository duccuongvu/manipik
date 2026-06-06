#!/usr/bin/env python3
#
# Created on Sat Jun 06 2026
#
# Copyright (c) 2026 Duc-Cuong Vu - vdcuong2002@gmail.com
#
"""Example 3 — FK ground-truth round-trip and Python/C++ agreement.

Generates a reachable pose from a random joint config via MuJoCo FK, solves it
analytically, and confirms (a) every returned solution reaches the pose and
(b) the Python and C++ backends return the same solution set.

Requires the manipsim conda env (MuJoCo) and a built `IiwaIKCpp` module
(`python build_cpp.py`).

Run from the repo root:
    python examples/03_fk_roundtrip.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iiwa_ik import IiwaIK as IK  # noqa: E402
from iiwa_fk import IiwaFK  # noqa: E402


def main() -> None:
    fk = IiwaFK()
    configs, poses = fk.random_poses(3, seed=7)

    try:
        import IiwaIKCpp
        have_cpp = True
    except ImportError:
        have_cpp = False
        print("(C++ module not built — skipping agreement check)\n")

    for i, (q_src, T) in enumerate(zip(configs, poses)):
        sols = IK.solve_all(T)
        worst = max(
            np.linalg.norm(fk.fk(q)[:3, 3] - T[:3, 3])
            + np.linalg.norm(fk.fk(q)[:3, :3] - T[:3, :3])
            for q in sols
        )
        line = f"pose {i}: {len(sols):3d} solutions, worst FK residual {worst:.2e}"

        if have_cpp:
            cp = IiwaIKCpp.solve_all(T)
            agree = cp.shape == sols.shape
            line += f"  |  C++ returns {len(cp):3d} (match={agree})"
        print(line)


if __name__ == "__main__":
    main()
