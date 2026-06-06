/*
 * Created on Sat Jun 06 2026
 *
 * Copyright (c) 2026 Duc-Cuong Vu - vdcuong2002@gmail.com
 */

#ifndef IIWA_IK_HPP
#define IIWA_IK_HPP

#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <limits>
#include <numeric>

namespace iiwa_ik {

using Mat4d = Eigen::Matrix4d;
using Mat3d = Eigen::Matrix3d;
using Vec7d = Eigen::Matrix<double, 7, 1>;
using Vec3d = Eigen::Vector3d;

struct RobotParams {
    double d1 = 0.1575, d2 = 0.2025, d3 = 0.2045, d4 = 0.2155;
    double d5 = 0.1845, d6 = 0.2155, d7 = 0.081,  d8 = 0.045;

    // ±170°/±120° alternating for J1–J6, ±175° for J7 (matches iiwa_ik.py).
    std::array<double, 7> q_upper = {
        2.9670597283903604, 2.0943951023931953, 2.9670597283903604,
        2.0943951023931953, 2.9670597283903604, 2.0943951023931953,
        3.0543261909900767
    };
    std::array<double, 7> q_lower = {
        -2.9670597283903604, -2.0943951023931953, -2.9670597283903604,
        -2.0943951023931953, -2.9670597283903604, -2.0943951023931953,
        -3.0543261909900767
    };
};

inline double wrap2pi(double lmbd) {
    double q = std::fmod(lmbd, 2.0 * M_PI);
    if (q < 0.0) q += 2.0 * M_PI;
    if (q == 0.0 && lmbd > 0.0) q = 2.0 * M_PI;
    return q;
}

inline Vec7d wrap(Vec7d v) {
    for (int i = 0; i < 7; ++i)
        if (v[i] < -M_PI || v[i] > M_PI)
            v[i] = wrap2pi(v[i] + M_PI) - M_PI;
    return v;
}

inline Mat3d skew(const Vec3d& v) {
    Mat3d s;
    s <<     0, -v[2],  v[1],
          v[2],     0, -v[0],
         -v[1],  v[0],     0;
    return s;
}

inline bool within_limits(const Vec7d& q, const RobotParams& p) {
    for (int i = 0; i < 7; ++i)
        if (q[i] < p.q_lower[i] || q[i] > p.q_upper[i]) return false;
    return true;
}

/**
 * Enumerate every feasible configuration that reaches `x_d`.
 *
 * @param x_d    4×4 SE(3) TCP target pose in the robot base frame.
 * @param n_psi  number of arm-angle ψ samples on [0, 2π) (default 20).
 * @param p      robot link lengths and joint limits.
 * @return       vector of 7-DOF joint configs (rad); empty if unreachable.
 */
inline std::vector<Vec7d> solve_all(const Mat4d& x_d, int n_psi = 20,
                                    const RobotParams& p = RobotParams()) {
    std::vector<Vec7d> out;

    const double d_bs = p.d1 + p.d2;
    const double d_se = p.d3 + p.d4;
    const double d_ew = p.d5 + p.d6;
    const double d_wt = p.d7 + p.d8;

    Vec3d x_7   = x_d.block<3,1>(0,3);
    Mat3d r_0_7 = x_d.block<3,3>(0,0);

    // Shoulder-to-wrist vector (eq 6)
    Vec3d x_sw       = x_7 - Vec3d(0,0,d_bs) - r_0_7 * Vec3d(0,0,d_wt);
    double x_sw_norm = x_sw.norm();
    if (x_sw_norm == 0.0) return out;
    Vec3d u_sw = x_sw / x_sw_norm;

    // Elbow angle magnitude (eq 7)
    double tmp = (x_sw.squaredNorm() - d_se*d_se - d_ew*d_ew) / (2.0*d_se*d_ew);
    if (std::abs(tmp) > 1.0 + std::numeric_limits<double>::epsilon()) return out;
    double q4_abs = std::acos(std::clamp(tmp, -1.0, 1.0));

    // Shoulder γ for reference arm (eq 11b, law-of-cosines at shoulder)
    double tmp_g = (d_se*d_se + x_sw_norm*x_sw_norm - d_ew*d_ew) / (2.0*d_se*x_sw_norm);
    if (std::abs(tmp_g) > 1.0 + std::numeric_limits<double>::epsilon()) return out;
    double gamma_abs = std::acos(std::clamp(tmp_g, -1.0, 1.0));

    // Reference azimuth and elevation of p_sw (eq 11a/11b)
    double q_az  = std::atan2(x_sw[1], x_sw[0]);
    double rho   = std::hypot(x_sw[0], x_sw[1]);
    double elev  = std::atan2(rho, x_sw[2]);

    Mat3d u_skew = skew(u_sw);

    out.reserve(n_psi * 8);
    for (int je : {-1, 1}) {
        double q4_val    = je * q4_abs;
        double q_elev_n  = elev + je * gamma_abs;

        double c0 = std::cos(q_az),     s0 = std::sin(q_az);
        double c1 = std::cos(q_elev_n), s1 = std::sin(q_elev_n);
        Mat3d r_0_3_o;
        r_0_3_o << c0*c1, -c0*s1, -s0,
                   s0*c1, -s0*s1,  c0,
                   -s1,   -c1,      0;

        Mat3d a_s = u_skew * r_0_3_o;
        Mat3d b_s = -(u_skew * u_skew) * r_0_3_o;
        Mat3d c_s = u_sw * u_sw.transpose() * r_0_3_o;

        // R^4_3(q4): sin sign tracks je so both elbow branches are correct
        double cq4 = std::cos(q4_abs);
        double sq4 = je * std::sin(q4_abs);
        Mat3d r_3_4;
        r_3_4 <<  cq4, 0, -sq4,
                 -sq4, 0, -cq4,
                    0, 1,    0;

        Mat3d a_w = r_3_4.transpose() * a_s.transpose() * r_0_7;
        Mat3d b_w = r_3_4.transpose() * b_s.transpose() * r_0_7;
        Mat3d c_w = r_3_4.transpose() * c_s.transpose() * r_0_7;

        for (int n = 0; n < n_psi; ++n) {
            double psi  = n * (2.0 * M_PI / n_psi);
            double spsi = std::sin(psi), cpsi = std::cos(psi);

            double q2_raw = -a_s(2,1)*spsi - b_s(2,1)*cpsi - c_s(2,1);
            double q6_raw =  a_w(2,2)*spsi + b_w(2,2)*cpsi + c_w(2,2);
            if (std::abs(q2_raw) > 1.0 || std::abs(q6_raw) > 1.0) continue;

            double q2_v = std::acos(std::clamp(q2_raw, -1.0, 1.0));
            double q6_v = std::acos(std::clamp(q6_raw, -1.0, 1.0));

            double q1_b = std::atan2(-a_s(1,1)*spsi - b_s(1,1)*cpsi - c_s(1,1),
                                      -a_s(0,1)*spsi - b_s(0,1)*cpsi - c_s(0,1));
            double q3_b = std::atan2( a_s(2,2)*spsi + b_s(2,2)*cpsi + c_s(2,2),
                                      -a_s(2,0)*spsi - b_s(2,0)*cpsi - c_s(2,0));
            double q5_b = std::atan2( a_w(1,2)*spsi + b_w(1,2)*cpsi + c_w(1,2),
                                       a_w(0,2)*spsi + b_w(0,2)*cpsi + c_w(0,2));
            double q7_b = std::atan2( a_w(2,1)*spsi + b_w(2,1)*cpsi + c_w(2,1),
                                      -a_w(2,0)*spsi - b_w(2,0)*cpsi - c_w(2,0));

            // j_s: shoulder flip; j_w: wrist flip — adjacent joints shift by π (eq 13)
            for (int js : {1, -1}) {
                double d13 = (js < 0) ? M_PI : 0.0;
                for (int jw : {1, -1}) {
                    double d57 = (jw < 0) ? M_PI : 0.0;
                    Vec7d qt;
                    qt[0] = q1_b + d13;
                    qt[1] = js * q2_v;
                    qt[2] = q3_b + d13;
                    qt[3] = q4_val;
                    qt[4] = q5_b + d57;
                    qt[5] = jw * q6_v;
                    qt[6] = q7_b + d57;
                    qt = wrap(qt);
                    if (within_limits(qt, p)) out.push_back(qt);
                }
            }
        }
    }

    return out;
}

}  // namespace iiwa_ik

#endif  // IIWA_IK_HPP
