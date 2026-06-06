#
# Created on Sat Jun 06 2026
#
# Copyright (c) 2026 Duc-Cuong Vu - vdcuong2002@gmail.com
#

import pybind11
import subprocess
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ext = sysconfig.get_config_var("EXT_SUFFIX")
    py_inc = Path(sys.prefix) / "include" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    out = HERE / f"IiwaIKCpp{ext}"

    cmd = [
        "g++", "-O3", "-shared", "-fPIC", "-std=c++17",
        "-march=native", "-mtune=native",
        "-ffast-math", "-funroll-loops", "-fomit-frame-pointer",
        "-DNDEBUG", "-DEIGEN_NO_DEBUG", "-DEIGEN_USE_BLAS",
        f"-I{pybind11.get_include()}", f"-I{py_inc}", "-I/usr/include/eigen3",
        str(HERE / "iiwa_ik.cpp"), "-o", str(out),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print("built:", out.name)


if __name__ == "__main__":
    main()
