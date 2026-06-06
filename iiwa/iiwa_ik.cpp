/*
 * Created on Sat Jun 06 2026
 *
 * Copyright (c) 2026 Duc-Cuong Vu - vdcuong2002@gmail.com
 */

 
/**
* iiwa_ik.cpp — pybind11 wrapper around the header-only solver in iiwa_ik.hpp.
 *
 * Exposes the C++ analytic IK to Python as `IiwaIKCpp.solve_all`. The actual
 * algorithm lives in iiwa_ik.hpp so it can be reused from pure-C++ programs
 * (see iiwa_ik_example.cpp).
 *
 * Build (Python module):
 *   python build_cpp.py
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "iiwa_ik.hpp"

namespace py = pybind11;
using namespace iiwa_ik;

// pybind11 interface: returns an (N, 7) numpy array of all feasible solutions.
py::array_t<double> py_solve_all(py::array_t<double> x_d_np, int n_psi) {
    auto xd = x_d_np.unchecked<2>();
    Mat4d x_d;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            x_d(r, c) = xd(r, c);

    std::vector<Vec7d> sols = solve_all(x_d, n_psi);

    py::array_t<double> out({(py::ssize_t)sols.size(), (py::ssize_t)7});
    auto buf = out.mutable_unchecked<2>();
    for (py::ssize_t r = 0; r < (py::ssize_t)sols.size(); ++r)
        for (py::ssize_t c = 0; c < 7; ++c)
            buf(r, c) = sols[r][c];
    return out;
}

PYBIND11_MODULE(IiwaIKCpp, m) {
    m.doc() = "C++ analytic IK for KUKA IIWA 14 — returns all feasible solutions";
    m.def("solve_all", &py_solve_all, py::arg("x_d"), py::arg("n_psi") = 20,
          "Return an (N, 7) array of all feasible joint configurations that "
          "reach x_d. The arm angle psi is sampled at n_psi points on [0, 2pi).");
}
