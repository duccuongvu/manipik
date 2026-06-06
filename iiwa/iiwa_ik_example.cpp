/*
 * Created on Sat Jun 06 2026
 *
 * Copyright (c) 2026 Duc-Cuong Vu - vdcuong2002@gmail.com
 */

/**
 * iiwa_ik_example.cpp — solve IK in pure C++ using the iiwa_ik.hpp library.
 *
 * Builds a target TCP pose, calls iiwa_ik::solve_all, and prints every feasible
 * joint configuration. No Python involved.
 *
 * Build & run:
 *   cmake -B build && cmake --build build
 *   ./build/iiwa_ik_example
 */

#include "iiwa_ik.hpp"

#include <cstdio>
#include <cmath>

using namespace iiwa_ik;

int main() {
    // Target TCP pose in the robot base frame: 30° yaw about z, at (0.40,-0.20,0.65).
    const double yaw = 30.0 * M_PI / 180.0;
    Mat4d x_d = Mat4d::Identity();
    x_d.block<3,3>(0,0) << std::cos(yaw), -std::sin(yaw), 0,
                           std::sin(yaw),  std::cos(yaw), 0,
                           0,              0,             1;
    x_d(0,3) = 0.40;
    x_d(1,3) = -0.20;
    x_d(2,3) = 0.65;

    const int n_psi = 20;  // arm-angle samples on [0, 2π)
    std::vector<Vec7d> sols = solve_all(x_d, n_psi);

    std::printf("target position : [%.3f, %.3f, %.3f]\n",
                x_d(0,3), x_d(1,3), x_d(2,3));
    std::printf("n_psi           : %d\n", n_psi);
    std::printf("feasible solutions: %zu\n\n", sols.size());

    const double r2d = 180.0 / M_PI;
    for (size_t i = 0; i < sols.size(); ++i) {
        const Vec7d& q = sols[i];
        std::printf("  [%3zu]  ", i);
        for (int j = 0; j < 7; ++j)
            std::printf("%8.2f", q[j] * r2d);
        std::printf("  deg\n");
    }

    return sols.empty() ? 1 : 0;
}
