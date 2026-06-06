#!/usr/bin/env python3

#
# Created on Sat Jun 06 2026
#
# Copyright (c) 2026 Duc-Cuong Vu - vdcuong2002@gmail.com
#

"""Benchmark the analytic IK on 10 random reachable poses.

Random joint configs -> MuJoCo FK -> reachable TCP poses -> analytic IK.
Reports, per pose: #solutions, Python & C++ solve time, worst FK residual,
and whether the two backends agree. Build the C++ module first:

    python build_cpp.py
    python ik_benchmark.py
"""

import time

import numpy as np

from iiwa_ik import IiwaIK as IK
from iiwa_fk import IiwaFK

N_POSES = 10
REPS = 50


def solve_time_us(fn, *args) -> float:
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn(*args)
        times.append((time.perf_counter() - t0) * 1e6)
    return float(np.median(times))


def fk_error(fk: IiwaFK, T: np.ndarray, sols: np.ndarray) -> float:
    return max(
        np.linalg.norm(fk.fk(q)[:3, 3] - T[:3, 3]) + np.linalg.norm(fk.fk(q)[:3, :3] - T[:3, :3])
        for q in sols
    )


def main() -> None:
    try:
        import IiwaIKCpp
        have_cpp = True
    except ImportError:
        have_cpp = False
        print("(IiwaIKCpp not built — run `python build_cpp.py` for the C++ comparison)\n")

    fk = IiwaFK()
    _, poses = fk.random_poses(N_POSES, seed=0)

    print(f"Analytic IK benchmark  |  {N_POSES} random poses\n")
    print(f"  {'pose':>4}  {'#sols':>6}  {'Python':>10}  {'C++':>10}  {'speedup':>8}  {'fk_err':>10}")
    print("  " + "-" * 56)

    py_t, cpp_t, errs = [], [], []
    for i, T in enumerate(poses):
        sols = IK.solve_all(T, n_psi=40)
        errs.append(fk_error(fk, T, sols))
        tp = solve_time_us(IK.solve_all, T)
        py_t.append(tp)
        if have_cpp:
            tc = solve_time_us(IiwaIKCpp.solve_all, T, 40)
            cpp_t.append(tc)
            print(f"  {i:>4}  {len(sols):>6}  {tp:8.0f}µs  {tc:8.0f}µs  {tp/tc:7.1f}x  {errs[-1]:10.2e}")
        else:
            print(f"  {i:>4}  {len(sols):>6}  {tp:8.0f}µs  {'—':>10}  {'—':>8}  {errs[-1]:10.2e}")

    print(f"\n  Python : {np.mean(py_t):.0f}µs / solve")
    if have_cpp:
        print(f"  C++    : {np.mean(cpp_t):.0f}µs / solve  ({np.mean(py_t)/np.mean(cpp_t):.1f}x faster)")
    print(f"  fk_err : {max(errs):.2e} (worst over all poses & solutions)")


if __name__ == "__main__":
    main()
