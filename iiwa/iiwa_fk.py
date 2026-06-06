#
# Created on Sat Jun 06 2026
#
# Copyright (c) 2026 Duc-Cuong Vu - vdcuong2002@gmail.com
#

from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco

# manipik/iiwa/iiwa_fk.py -> repo root is parents[2]
DEFAULT_XML = (
    Path(__file__).resolve().parents[2]
    / "manipsim" / "assets" / "robots" / "iiwa" / "robot.xml"
)


class IiwaFK:
    """MuJoCo-backed forward kinematics for the IIWA 14 ``tcp`` site.

    The ``tcp`` site (link7 + 0.045 m) matches the analytic IK tool offset
    ``d_wt = d7 + d8 = 0.126 m``, so FK poses round-trip exactly through the IK.
    """

    def __init__(self, xml_path: Path | str = DEFAULT_XML, site: str = "tcp") -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.site_id < 0:
            raise ValueError(f"site {site!r} not found in {xml_path}")
        self.nq = self.model.nq
        # joint limits straight from the model
        self.q_lower = self.model.jnt_range[: self.nq, 0].copy()
        self.q_upper = self.model.jnt_range[: self.nq, 1].copy()

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Return the 4x4 SE(3) pose of the tcp site for joint config ``q``."""
        self.data.qpos[: self.nq] = q
        mujoco.mj_kinematics(self.model, self.data)
        T = np.eye(4, dtype=float)
        T[:3, :3] = self.data.site_xmat[self.site_id].reshape(3, 3)
        T[:3, 3] = self.data.site_xpos[self.site_id]
        return T

    def random_configs(
        self, n: int, *, margin: float = 0.05, seed: int | None = None
    ) -> np.ndarray:
        """Sample ``n`` joint configs uniformly within the joint limits.

        ``margin`` shrinks each limit by that fraction of its span so sampled
        configs stay clear of the boundaries (keeps redundancy-resolved IK
        solutions safely in range).
        """
        rng = np.random.default_rng(seed)
        span = self.q_upper - self.q_lower
        lo = self.q_lower + margin * span
        hi = self.q_upper - margin * span
        return rng.uniform(lo, hi, size=(n, self.nq))

    def random_poses(
        self, n: int, *, margin: float = 0.05, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (configs, poses): ``n`` random configs and their FK poses.

        Each pose is guaranteed reachable (the sampling config itself solves it).
        """
        configs = self.random_configs(n, margin=margin, seed=seed)
        poses = np.array([self.fk(q) for q in configs])
        return configs, poses


if __name__ == "__main__":
    fk = IiwaFK()
    print(f"Loaded IIWA model: nq={fk.nq}, xml={DEFAULT_XML}")
    print("q_lower [deg]:", np.array2string(np.degrees(fk.q_lower), precision=1))
    print("q_upper [deg]:", np.array2string(np.degrees(fk.q_upper), precision=1))
    q0 = np.zeros(fk.nq)
    print("FK(0) tcp position [m]:", np.array2string(fk.fk(q0)[:3, 3], precision=4))
