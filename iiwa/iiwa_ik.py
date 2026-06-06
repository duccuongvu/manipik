#
# Created on Sat Jun 06 2026
#
# Copyright (c) 2026 Duc-Cuong Vu - vdcuong2002@gmail.com
#


"""Analytic inverse kinematics for the KUKA IIWA 14 (7-DOF, redundant).

Closed-form solver based on the arm-angle parameterization of Shimizu et al.
(2008). The only redundant degree of freedom is the arm angle ψ; the caller
chooses how finely to sample it. The solver evaluates each ψ sample, expands
the result through the IIWA joint symmetries, and filters by joint limits.

Unlike a classic redundancy-resolved IK, this solver does **not** take a seed
configuration and does **not** return a single "closest" answer. It returns
*every* joint configuration that reaches the target and respects the joint
limits. Picking among them (closest to current state, collision-free, ...) is
left to the caller.

    >>> from iiwa_ik import IiwaIK as IK
    >>> sols = IK.solve_all(x_d, n_psi=20)   # sample ψ at 20 points on [0, 2π)
    >>> sols.shape
    (N, 7)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

DEFAULT_N_PSI = 20  # number of arm-angle ψ samples on [0, 2π)
_FLOAT64_EPS = 2.220446049250313e-16  # np.finfo(float).eps, inlined to avoid a numpy warning


@dataclass
class RobotParams:
    """IIWA 14 link lengths (m) and per-joint limits (rad)."""

    d1: float = 0.1575
    d2: float = 0.2025
    d3: float = 0.2045
    d4: float = 0.2155
    d5: float = 0.1845
    d6: float = 0.2155
    d7: float = 0.081
    d8: float = 0.045
    q_limit_upper: np.ndarray = field(
        default_factory=lambda: np.radians(
            [170.0, 120.0, 170.0, 120.0, 170.0, 120.0, 175.0]
        )
    )
    q_limit_lower: np.ndarray = field(
        default_factory=lambda: np.radians(
            [-170.0, -120.0, -170.0, -120.0, -170.0, -120.0, -175.0]
        )
    )


class IiwaIK:
    """Closed-form IK that enumerates *all* feasible IIWA 14 configurations."""

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def wrap2pi(lmbd: float) -> float:
        q = math.fmod(lmbd, 2.0 * math.pi)
        q = q if q >= 0.0 else q + 2.0 * math.pi
        if q == 0.0 and lmbd > 0.0:
            q = 2.0 * math.pi
        return q

    @staticmethod
    def wrap(lmbd: np.ndarray) -> np.ndarray:
        """Wrap each joint angle into (-π, π]."""
        out = np.array(lmbd, dtype=float, copy=True)
        for i in range(out.size):
            if out[i] < -math.pi or out[i] > math.pi:
                out[i] = IiwaIK.wrap2pi(out[i] + math.pi) - math.pi
        return out

    @staticmethod
    def skew(v: np.ndarray) -> np.ndarray:
        s = np.zeros((3, 3), dtype=float)
        s[0, 1], s[0, 2] = -v[2], v[1]
        s[1, 0], s[1, 2] = v[2], -v[0]
        s[2, 0], s[2, 1] = -v[1], v[0]
        return s

    # -------------------------------------------------------------- main solve
    @staticmethod
    def solve_all(
        x_d: np.ndarray,
        n_psi: int = DEFAULT_N_PSI,
        param: RobotParams = RobotParams(),
    ) -> np.ndarray:
        """Return every feasible joint configuration reaching ``x_d``.

        Args:
            x_d:   4×4 SE(3) target pose of the TCP in the robot base frame.
            n_psi: number of arm-angle ψ samples; ``[0, 2π)`` is split into
                   ``n_psi`` equal intervals. More samples → more solutions
                   and slower. Default ``20``.
            param: robot link lengths and joint limits.

        Returns:
            An ``(N, 7)`` array of joint configurations (rad), one row per
            valid solution. Empty ``(0, 7)`` array if the target is unreachable.
        """
        d_bs = param.d1 + param.d2
        d_se = param.d3 + param.d4
        d_ew = param.d5 + param.d6
        d_wt = param.d7 + param.d8

        x_7   = x_d[:3, 3]
        r_0_7 = x_d[:3, :3]

        # Shoulder-to-wrist vector (eq 6)
        x_sw      = x_7 - np.array([0.0, 0.0, d_bs]) - r_0_7 @ np.array([0.0, 0.0, d_wt])
        x_sw_norm = np.linalg.norm(x_sw)
        empty = np.empty((0, 7), dtype=float)
        if x_sw_norm == 0.0:
            return empty
        u_sw = x_sw / x_sw_norm

        # Elbow angle magnitude (eq 7)
        tmp = (float(x_sw @ x_sw) - d_se * d_se - d_ew * d_ew) / (2.0 * d_se * d_ew)
        if abs(tmp) > 1.0 + _FLOAT64_EPS:
            return empty
        q4_abs = math.acos(np.clip(tmp, -1.0, 1.0))

        # Shoulder angle γ for reference arm (eq 11b, law-of-cosines at shoulder)
        tmp_g = (d_se * d_se + x_sw_norm * x_sw_norm - d_ew * d_ew) / (2.0 * d_se * x_sw_norm)
        if abs(tmp_g) > 1.0 + _FLOAT64_EPS:
            return empty
        gamma_abs = math.acos(np.clip(tmp_g, -1.0, 1.0))

        # Reference azimuth and elevation of p_sw (eq 11a/11b)
        q_az  = math.atan2(x_sw[1], x_sw[0])
        rho   = math.sqrt(x_sw[0] ** 2 + x_sw[1] ** 2)
        elev  = math.atan2(rho, x_sw[2])

        psi_list = np.linspace(0.0, 2.0 * math.pi, n_psi + 1)[:-1]
        u_skew = IiwaIK.skew(u_sw)

        feasible: list[np.ndarray] = []

        for j_e in (-1, 1):
            # Eq (7): q4 = j_e * q4_abs
            q4_val = j_e * q4_abs

            # Eq (11b): reference shoulder elevation for this elbow branch
            q_elev_n = elev + j_e * gamma_abs

            c0, s0 = math.cos(q_az),     math.sin(q_az)
            c1, s1 = math.cos(q_elev_n), math.sin(q_elev_n)
            r_0_3_o = np.array([
                [ c0 * c1, -c0 * s1, -s0],
                [ s0 * c1, -s0 * s1,  c0],
                [-s1,      -c1,        0.0],
            ])

            a_s = u_skew @ r_0_3_o
            b_s = -(u_skew @ u_skew) @ r_0_3_o
            c_s = np.outer(u_sw, u_sw) @ r_0_3_o

            # R^4_3(q4): sin sign tracks j_e so both branches work
            cq4 = math.cos(q4_abs)
            sq4 = j_e * math.sin(q4_abs)
            r_3_4 = np.array([
                [ cq4, 0.0, -sq4],
                [-sq4, 0.0, -cq4],
                [ 0.0, 1.0,  0.0],
            ])
            a_w = r_3_4.T @ a_s.T @ r_0_7
            b_w = r_3_4.T @ b_s.T @ r_0_7
            c_w = r_3_4.T @ c_s.T @ r_0_7

            for psi in psi_list:
                spsi, cpsi = math.sin(psi), math.cos(psi)

                q2_raw = -a_s[2, 1] * spsi - b_s[2, 1] * cpsi - c_s[2, 1]
                q6_raw =  a_w[2, 2] * spsi + b_w[2, 2] * cpsi + c_w[2, 2]
                if abs(q2_raw) > 1.0 or abs(q6_raw) > 1.0:
                    continue

                q2_v = math.acos(max(-1.0, min(1.0, q2_raw)))
                q6_v = math.acos(max(-1.0, min(1.0, q6_raw)))

                q1_b = math.atan2(
                    -a_s[1, 1] * spsi - b_s[1, 1] * cpsi - c_s[1, 1],
                    -a_s[0, 1] * spsi - b_s[0, 1] * cpsi - c_s[0, 1],
                )
                q3_b = math.atan2(
                     a_s[2, 2] * spsi + b_s[2, 2] * cpsi + c_s[2, 2],
                    -a_s[2, 0] * spsi - b_s[2, 0] * cpsi - c_s[2, 0],
                )
                q5_b = math.atan2(
                     a_w[1, 2] * spsi + b_w[1, 2] * cpsi + c_w[1, 2],
                     a_w[0, 2] * spsi + b_w[0, 2] * cpsi + c_w[0, 2],
                )
                q7_b = math.atan2(
                     a_w[2, 1] * spsi + b_w[2, 1] * cpsi + c_w[2, 1],
                    -a_w[2, 0] * spsi - b_w[2, 0] * cpsi - c_w[2, 0],
                )

                # j_s: shoulder flip (q2 sign); j_w: wrist flip (q6 sign)
                # When sign flips, adjacent atan2 joints shift by π (eq 13)
                for j_s in (1, -1):
                    d13 = 0.0 if j_s > 0 else math.pi
                    for j_w in (1, -1):
                        d57 = 0.0 if j_w > 0 else math.pi
                        qt = np.array([
                            q1_b + d13, j_s * q2_v, q3_b + d13, q4_val,
                            q5_b + d57, j_w * q6_v, q7_b + d57,
                        ])
                        qt = IiwaIK.wrap(qt)
                        if (np.all(qt >= param.q_limit_lower) and
                                np.all(qt <= param.q_limit_upper)):
                            feasible.append(qt)

        if not feasible:
            return empty

        return np.unique(np.round(np.array(feasible, dtype=float), 9), axis=0)


def _rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


if __name__ == "__main__":
    x_d = np.eye(4)
    x_d[:3, :3] = _rot_z(math.radians(30.0))
    x_d[:3, 3] = np.array([0.40, -0.20, 0.65])

    sols = IiwaIK.solve_all(x_d, n_psi=20)
    print(f"Found {len(sols)} feasible IK solution(s).")
    for i, q in enumerate(sols):
        print(f"  [{i:2d}] q [deg]: {np.array2string(np.degrees(q), precision=2)}")
