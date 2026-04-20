/*
    MIT License

    Copyright (c) 2021 Zhepei Wang (wangzhepei@live.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#ifndef GCOPTER_HPP
#define GCOPTER_HPP

#include "gcopter/igo.hpp"
#include "gcopter/minco.hpp"
#include "gcopter/flatness.hpp"
#include "gcopter/geo_utils.hpp"
#include "gcopter/lbfgs.hpp"

#include <Eigen/Eigen>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gcopter
{

    class GCOPTER_PolytopeSFC
    {
    public:
        typedef Eigen::Matrix3Xd PolyhedronV;
        typedef Eigen::MatrixX4d PolyhedronH;
        typedef std::vector<PolyhedronV> PolyhedraV;
        typedef std::vector<PolyhedronH> PolyhedraH;

        struct TrajectoryViolationMetrics
        {
            double penalty_cost = 0.0;
            double max_corridor_violation = 0.0;
            double max_velocity_violation = 0.0;
            double max_acceleration_violation = 0.0;
            double max_body_rate_violation = 0.0;
            double max_tilt_violation = 0.0;
            double max_thrust_violation = 0.0;
            int sample_count = 0;

            inline double maxViolation() const
            {
                return std::max(std::max(std::max(max_corridor_violation, max_velocity_violation),
                                         std::max(max_acceleration_violation, max_body_rate_violation)),
                                std::max(max_tilt_violation, max_thrust_violation));
            }
        };

        struct LBFGSSolveOptions
        {
            double rel_cost_tol = 1.0e-5;
            int max_iterations = 0;
            int max_evaluations = 0;
            double max_wall_time = 0.0;
            int mem_size = 256;
            int past = 3;
        };

        struct IGOSolveOptions
        {
            int population = 32;
            int max_iterations = 200;
            int max_evaluations = 0;
            double max_wall_time = 0.0;
            unsigned int seed = 0;
            double feasibility_tol = 1.0e-4;
            double tau_box_radius = 2.5;
            double xi_box_bound = 1.5;
            double meta_spatial_logit_bound = 3.0;
            double meta_time_logit_bound = 3.0;
            double meta_gamma_box_radius = 3.0;
            double meta_time_floor_scale = 1.15;
            double meta_initial_time_scale = 1.30;
            double meta_total_slack_min_scale = 0.10;
            double meta_total_slack_max_scale = 3.00;
            int archive_top_k = 6;
            int refine_top_k = 4;
            double outer_budget_ratio = 0.45;
            double refine_rel_cost_tol = 1.0e-5;
            int refine_max_iterations = 60;
            int refine_max_evaluations = 500;
            double refine_max_wall_time = 0.50;
            int refine_mem_size = 64;
            int refine_past = 3;
            double meta_optimizer_time_weight = 4.0;
            double meta_optimizer_length_weight = 0.0;
            double meta_optimizer_energy_weight = 0.0;
            double meta_optimizer_waypoint_smooth_weight = 0.0;
            double meta_optimizer_collision_weight = 30.0;
            double meta_optimizer_velocity_weight = 20.0;
            double meta_optimizer_acceleration_weight = 25.0;
            double meta_optimizer_body_rate_weight = 0.0;
            double meta_optimizer_tilt_weight = 0.0;
            double meta_optimizer_thrust_weight = 0.0;
            double meta_optimizer_max_acceleration = 15.0;
            double meta_optimizer_sample_dt = 0.10;
            int meta_optimizer_midpoints = 3;
            double meta_optimizer_time_lb = 0.1;
            double meta_optimizer_time_ub = 8.0;
            double meta_optimizer_simple_max_velocity = 4.0;
            double meta_optimizer_simple_max_acceleration = 15.0;
            double meta_optimizer_target_velocity_ratio = 0.90;
            bool meta_optimizer_has_workspace_bounds = false;
            Eigen::Vector3d meta_optimizer_workspace_min = Eigen::Vector3d::Zero();
            Eigen::Vector3d meta_optimizer_workspace_max = Eigen::Vector3d::Zero();
            std::function<bool(const Eigen::Vector3d &)> meta_optimizer_collision_checker;
            double min_sigma = 1.0e-3;
            double eta_mean = 0.45;
            double eta_sigma = 0.20;
            double elite_ratio = 0.35;
        };

        struct SolverResult
        {
            bool has_solution = false;
            bool converged = false;
            bool hit_eval_budget = false;
            bool hit_time_budget = false;
            int solver_status = 0;
            int iterations = 0;
            int eval_count = 0;
            double wall_time = 0.0;
            double objective = std::numeric_limits<double>::infinity();
            double total_duration = std::numeric_limits<double>::infinity();
            double trajectory_length = std::numeric_limits<double>::infinity();
            Eigen::VectorXd best_x;
            bool has_meta_direct_solution = false;
            Eigen::Matrix3Xd meta_direct_points;
            Eigen::VectorXd meta_direct_times;
            TrajectoryViolationMetrics violations;
            std::string status;
        };

    private:
        struct BudgetExceededException : public std::runtime_error
        {
            explicit BudgetExceededException(const std::string &msg)
                : std::runtime_error(msg) {}
        };

        struct EvaluatedCandidate
        {
            Eigen::VectorXd x;
            double cost = std::numeric_limits<double>::infinity();
            double max_violation = std::numeric_limits<double>::infinity();
            bool feasible = false;
        };

        struct LBFGSSolveContext
        {
            GCOPTER_PolytopeSFC *solver = nullptr;
            LBFGSSolveOptions options;
            std::chrono::steady_clock::time_point start_time;
            Eigen::VectorXd best_x;
            double best_cost = std::numeric_limits<double>::infinity();
            int eval_count = 0;
            int iterations = 0;
            bool hit_eval_budget = false;
            bool hit_time_budget = false;
        };

        minco::MINCO_S3NU minco;
        flatness::FlatnessMap flatmap;

        double rho;
        Eigen::Matrix3d headPVA;
        Eigen::Matrix3d tailPVA;

        PolyhedraV vPolytopes;
        PolyhedraH hPolytopes;
        Eigen::Matrix3Xd shortPath;

        Eigen::VectorXi pieceIdx;
        Eigen::VectorXi vPolyIdx;
        Eigen::VectorXi hPolyIdx;

        int polyN;
        int pieceN;

        int spatialDim;
        int temporalDim;
        int metaSpatialDim;
        int metaTemporalDim;
        int metaDecisionDim;

        double smoothEps;
        int integralRes;
        Eigen::VectorXd magnitudeBd;
        Eigen::VectorXd penaltyWt;
        Eigen::VectorXd physicalPm;
        double allocSpeed;
        Eigen::VectorXi spatialBlockSizes;
        Eigen::VectorXi spatialBlockOffsets;
        Eigen::VectorXi metaSpatialOffsets;
        Eigen::VectorXd metaTimeFloor;

        lbfgs::lbfgs_parameter_t lbfgs_params;

        Eigen::Matrix3Xd points;
        Eigen::VectorXd times;
        Eigen::Matrix3Xd gradByPoints;
        Eigen::VectorXd gradByTimes;
        Eigen::MatrixX3d partialGradByCoeffs;
        Eigen::VectorXd partialGradByTimes;

    private:
        static inline double positiveEps()
        {
            return 1.0e-12;
        }

        static inline double clampPositive(const double value,
                                           const double eps = 1.0e-12)
        {
            return std::max(value, eps);
        }

        static inline double softplus(const double x)
        {
            if (x >= 0.0)
            {
                return x + std::log1p(std::exp(-x));
            }
            else
            {
                return std::log1p(std::exp(x));
            }
        }

        static inline double inverseSoftplus(const double y)
        {
            const double clamped = clampPositive(y);
            if (clamped > 50.0)
            {
                return clamped;
            }
            return std::log(std::expm1(clamped));
        }

        static inline Eigen::VectorXd softmax(const Eigen::VectorXd &logits)
        {
            Eigen::VectorXd probs(logits.size());
            if (logits.size() <= 0)
            {
                return probs;
            }

            const double max_logit = logits.maxCoeff();
            const Eigen::ArrayXd shifted = (logits.array() - max_logit).exp();
            const double denom = clampPositive(shifted.sum());
            probs = (shifted / denom).matrix();
            return probs;
        }

        static inline bool isBetterCandidate(const EvaluatedCandidate &candidate,
                                             const EvaluatedCandidate &best)
        {
            const bool candidate_finite = std::isfinite(candidate.cost);
            const bool best_finite = std::isfinite(best.cost);
            if (candidate_finite != best_finite)
            {
                return candidate_finite;
            }

            if (candidate.feasible != best.feasible)
            {
                return candidate.feasible;
            }

            if (candidate.feasible && best.feasible)
            {
                if (candidate.cost != best.cost)
                {
                    return candidate.cost < best.cost;
                }

                return candidate.max_violation < best.max_violation;
            }

            if (candidate.max_violation != best.max_violation)
            {
                return candidate.max_violation < best.max_violation;
            }

            return candidate.cost < best.cost;
        }

        static inline bool sameDecisionVector(const Eigen::VectorXd &lhs,
                                              const Eigen::VectorXd &rhs)
        {
            return lhs.size() == rhs.size() && lhs.isApprox(rhs, 1.0e-9);
        }

        static inline void insertArchiveCandidate(std::vector<EvaluatedCandidate> &archive,
                                                  const EvaluatedCandidate &candidate,
                                                  const int top_k)
        {
            if (!std::isfinite(candidate.cost))
            {
                return;
            }

            archive.push_back(candidate);
            std::sort(archive.begin(), archive.end(), &GCOPTER_PolytopeSFC::isBetterCandidate);
            const std::size_t limit = static_cast<std::size_t>(std::max(1, top_k));
            if (archive.size() > limit)
            {
                archive.resize(limit);
            }
        }

        static inline void forwardT(const Eigen::VectorXd &tau,
                                    Eigen::VectorXd &T)
        {
            const int sizeTau = tau.size();
            T.resize(sizeTau);
            for (int i = 0; i < sizeTau; i++)
            {
                T(i) = tau(i) > 0.0
                           ? ((0.5 * tau(i) + 1.0) * tau(i) + 1.0)
                           : 1.0 / ((0.5 * tau(i) - 1.0) * tau(i) + 1.0);
            }
            return;
        }

        template <typename EIGENVEC>
        static inline void backwardT(const Eigen::VectorXd &T,
                                     EIGENVEC &tau)
        {
            const int sizeT = T.size();
            tau.resize(sizeT);
            for (int i = 0; i < sizeT; i++)
            {
                tau(i) = T(i) > 1.0
                             ? (sqrt(2.0 * T(i) - 1.0) - 1.0)
                             : (1.0 - sqrt(2.0 / T(i) - 1.0));
            }

            return;
        }

        template <typename EIGENVEC>
        static inline void backwardGradT(const Eigen::VectorXd &tau,
                                         const Eigen::VectorXd &gradT,
                                         EIGENVEC &gradTau)
        {
            const int sizeTau = tau.size();
            gradTau.resize(sizeTau);
            double denSqrt;
            for (int i = 0; i < sizeTau; i++)
            {
                if (tau(i) > 0)
                {
                    gradTau(i) = gradT(i) * (tau(i) + 1.0);
                }
                else
                {
                    denSqrt = (0.5 * tau(i) - 1.0) * tau(i) + 1.0;
                    gradTau(i) = gradT(i) * (1.0 - tau(i)) / (denSqrt * denSqrt);
                }
            }

            return;
        }

        static inline void forwardP(const Eigen::VectorXd &xi,
                                    const Eigen::VectorXi &vIdx,
                                    const PolyhedraV &vPolys,
                                    Eigen::Matrix3Xd &P)
        {
            const int sizeP = vIdx.size();
            P.resize(3, sizeP);
            Eigen::VectorXd q;
            for (int i = 0, j = 0, k, l; i < sizeP; i++, j += k)
            {
                l = vIdx(i);
                k = vPolys[l].cols();
                q = xi.segment(j, k).normalized().head(k - 1);
                P.col(i) = vPolys[l].rightCols(k - 1) * q.cwiseProduct(q) +
                           vPolys[l].col(0);
            }
            return;
        }

        static inline double costTinyNLS(void *ptr,
                                         const Eigen::VectorXd &xi,
                                         Eigen::VectorXd &gradXi)
        {
            const int n = xi.size();
            const Eigen::Matrix3Xd &ovPoly = *(Eigen::Matrix3Xd *)ptr;

            const double sqrNormXi = xi.squaredNorm();
            const double invNormXi = 1.0 / sqrt(sqrNormXi);
            const Eigen::VectorXd unitXi = xi * invNormXi;
            const Eigen::VectorXd r = unitXi.head(n - 1);
            const Eigen::Vector3d delta = ovPoly.rightCols(n - 1) * r.cwiseProduct(r) +
                                          ovPoly.col(1) - ovPoly.col(0);

            double cost = delta.squaredNorm();
            gradXi.head(n - 1) = (ovPoly.rightCols(n - 1).transpose() * (2 * delta)).array() *
                                 r.array() * 2.0;
            gradXi(n - 1) = 0.0;
            gradXi = (gradXi - unitXi.dot(gradXi) * unitXi).eval() * invNormXi;

            const double sqrNormViolation = sqrNormXi - 1.0;
            if (sqrNormViolation > 0.0)
            {
                double c = sqrNormViolation * sqrNormViolation;
                const double dc = 3.0 * c;
                c *= sqrNormViolation;
                cost += c;
                gradXi += dc * 2.0 * xi;
            }

            return cost;
        }

        template <typename EIGENVEC>
        static inline void backwardP(const Eigen::Matrix3Xd &P,
                                     const Eigen::VectorXi &vIdx,
                                     const PolyhedraV &vPolys,
                                     EIGENVEC &xi)
        {
            const int sizeP = P.cols();

            double minSqrD;
            lbfgs::lbfgs_parameter_t tiny_nls_params;
            tiny_nls_params.past = 0;
            tiny_nls_params.delta = 1.0e-5;
            tiny_nls_params.g_epsilon = FLT_EPSILON;
            tiny_nls_params.max_iterations = 128;

            Eigen::Matrix3Xd ovPoly;
            for (int i = 0, j = 0, k, l; i < sizeP; i++, j += k)
            {
                l = vIdx(i);
                k = vPolys[l].cols();

                ovPoly.resize(3, k + 1);
                ovPoly.col(0) = P.col(i);
                ovPoly.rightCols(k) = vPolys[l];
                Eigen::VectorXd x(k);
                x.setConstant(sqrt(1.0 / k));
                lbfgs::lbfgs_optimize(x,
                                      minSqrD,
                                      &GCOPTER_PolytopeSFC::costTinyNLS,
                                      nullptr,
                                      nullptr,
                                      &ovPoly,
                                      tiny_nls_params);

                xi.segment(j, k) = x;
            }

            return;
        }

        template <typename EIGENVEC>
        static inline void backwardGradP(const Eigen::VectorXd &xi,
                                         const Eigen::VectorXi &vIdx,
                                         const PolyhedraV &vPolys,
                                         const Eigen::Matrix3Xd &gradP,
                                         EIGENVEC &gradXi)
        {
            const int sizeP = vIdx.size();
            gradXi.resize(xi.size());

            double normInv;
            Eigen::VectorXd q, gradQ, unitQ;
            for (int i = 0, j = 0, k, l; i < sizeP; i++, j += k)
            {
                l = vIdx(i);
                k = vPolys[l].cols();
                q = xi.segment(j, k);
                normInv = 1.0 / q.norm();
                unitQ = q * normInv;
                gradQ.resize(k);
                gradQ.head(k - 1) = (vPolys[l].rightCols(k - 1).transpose() * gradP.col(i)).array() *
                                    unitQ.head(k - 1).array() * 2.0;
                gradQ(k - 1) = 0.0;
                gradXi.segment(j, k) = (gradQ - unitQ * unitQ.dot(gradQ)) * normInv;
            }

            return;
        }

        template <typename EIGENVEC>
        static inline void normRetrictionLayer(const Eigen::VectorXd &xi,
                                               const Eigen::VectorXi &vIdx,
                                               const PolyhedraV &vPolys,
                                               double &cost,
                                               EIGENVEC &gradXi)
        {
            const int sizeP = vIdx.size();
            gradXi.resize(xi.size());

            double sqrNormQ, sqrNormViolation, c, dc;
            Eigen::VectorXd q;
            for (int i = 0, j = 0, k; i < sizeP; i++, j += k)
            {
                k = vPolys[vIdx(i)].cols();

                q = xi.segment(j, k);
                sqrNormQ = q.squaredNorm();
                sqrNormViolation = sqrNormQ - 1.0;
                if (sqrNormViolation > 0.0)
                {
                    c = sqrNormViolation * sqrNormViolation;
                    dc = 3.0 * c;
                    c *= sqrNormViolation;
                    cost += c;
                    gradXi.segment(j, k) += dc * 2.0 * q;
                }
            }

            return;
        }

        static inline void normRetrictionLayerCostOnly(const Eigen::VectorXd &xi,
                                                       const Eigen::VectorXi &vIdx,
                                                       const PolyhedraV &vPolys,
                                                       double &cost)
        {
            const int sizeP = vIdx.size();

            double sqrNormQ, sqrNormViolation, c;
            Eigen::VectorXd q;
            for (int i = 0, j = 0, k; i < sizeP; i++, j += k)
            {
                k = vPolys[vIdx(i)].cols();
                q = xi.segment(j, k);
                sqrNormQ = q.squaredNorm();
                sqrNormViolation = sqrNormQ - 1.0;
                if (sqrNormViolation > 0.0)
                {
                    c = sqrNormViolation * sqrNormViolation;
                    c *= sqrNormViolation;
                    cost += c;
                }
            }
        }

        static inline bool smoothedL1(const double &x,
                                      const double &mu,
                                      double &f,
                                      double &df)
        {
            if (x < 0.0)
            {
                return false;
            }
            else if (x > mu)
            {
                f = x - 0.5 * mu;
                df = 1.0;
                return true;
            }
            else
            {
                const double xdmu = x / mu;
                const double sqrxdmu = xdmu * xdmu;
                const double mumxd2 = mu - 0.5 * x;
                f = mumxd2 * sqrxdmu * xdmu;
                df = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);
                return true;
            }
        }

        // magnitudeBounds = [v_max, omg_max, theta_max, thrust_min, thrust_max]^T
        // penaltyWeights = [pos_weight, vel_weight, omg_weight, theta_weight, thrust_weight]^T
        // physicalParams = [vehicle_mass, gravitational_acceleration, horitonral_drag_coeff,
        //                   vertical_drag_coeff, parasitic_drag_coeff, speed_smooth_factor]^T
        static inline void attachPenaltyFunctional(const Eigen::VectorXd &T,
                                                   const Eigen::MatrixX3d &coeffs,
                                                   const Eigen::VectorXi &hIdx,
                                                   const PolyhedraH &hPolys,
                                                   const double &smoothFactor,
                                                   const int &integralResolution,
                                                   const Eigen::VectorXd &magnitudeBounds,
                                                   const Eigen::VectorXd &penaltyWeights,
                                                   flatness::FlatnessMap &flatMap,
                                                   double &cost,
                                                   Eigen::VectorXd &gradT,
                                                   Eigen::MatrixX3d &gradC)
        {
            const double velSqrMax = magnitudeBounds(0) * magnitudeBounds(0);
            const double omgSqrMax = magnitudeBounds(1) * magnitudeBounds(1);
            const double thetaMax = magnitudeBounds(2);
            const double thrustMean = 0.5 * (magnitudeBounds(3) + magnitudeBounds(4));
            const double thrustRadi = 0.5 * fabs(magnitudeBounds(4) - magnitudeBounds(3));
            const double thrustSqrRadi = thrustRadi * thrustRadi;

            const double weightPos = penaltyWeights(0);
            const double weightVel = penaltyWeights(1);
            const double weightOmg = penaltyWeights(2);
            const double weightTheta = penaltyWeights(3);
            const double weightThrust = penaltyWeights(4);

            Eigen::Vector3d pos, vel, acc, jer, sna;
            Eigen::Vector3d totalGradPos, totalGradVel, totalGradAcc, totalGradJer;
            double totalGradPsi, totalGradPsiD;
            double thr, cos_theta;
            Eigen::Vector4d quat;
            Eigen::Vector3d omg;
            double gradThr;
            Eigen::Vector4d gradQuat;
            Eigen::Vector3d gradPos, gradVel, gradOmg;

            double step, alpha;
            double s1, s2, s3, s4, s5;
            Eigen::Matrix<double, 6, 1> beta0, beta1, beta2, beta3, beta4;
            Eigen::Vector3d outerNormal;
            int K, L;
            double violaPos, violaVel, violaOmg, violaTheta, violaThrust;
            double violaPosPenaD, violaVelPenaD, violaOmgPenaD, violaThetaPenaD, violaThrustPenaD;
            double violaPosPena, violaVelPena, violaOmgPena, violaThetaPena, violaThrustPena;
            double node, pena;

            const int pieceNum = T.size();
            const double integralFrac = 1.0 / integralResolution;
            for (int i = 0; i < pieceNum; i++)
            {
                const Eigen::Matrix<double, 6, 3> &c = coeffs.block<6, 3>(i * 6, 0);
                step = T(i) * integralFrac;
                for (int j = 0; j <= integralResolution; j++)
                {
                    s1 = j * step;
                    s2 = s1 * s1;
                    s3 = s2 * s1;
                    s4 = s2 * s2;
                    s5 = s4 * s1;
                    beta0(0) = 1.0, beta0(1) = s1, beta0(2) = s2, beta0(3) = s3, beta0(4) = s4, beta0(5) = s5;
                    beta1(0) = 0.0, beta1(1) = 1.0, beta1(2) = 2.0 * s1, beta1(3) = 3.0 * s2, beta1(4) = 4.0 * s3, beta1(5) = 5.0 * s4;
                    beta2(0) = 0.0, beta2(1) = 0.0, beta2(2) = 2.0, beta2(3) = 6.0 * s1, beta2(4) = 12.0 * s2, beta2(5) = 20.0 * s3;
                    beta3(0) = 0.0, beta3(1) = 0.0, beta3(2) = 0.0, beta3(3) = 6.0, beta3(4) = 24.0 * s1, beta3(5) = 60.0 * s2;
                    beta4(0) = 0.0, beta4(1) = 0.0, beta4(2) = 0.0, beta4(3) = 0.0, beta4(4) = 24.0, beta4(5) = 120.0 * s1;
                    pos = c.transpose() * beta0;
                    vel = c.transpose() * beta1;
                    acc = c.transpose() * beta2;
                    jer = c.transpose() * beta3;
                    sna = c.transpose() * beta4;

                    flatMap.forward(vel, acc, jer, 0.0, 0.0, thr, quat, omg);

                    violaVel = vel.squaredNorm() - velSqrMax;
                    violaOmg = omg.squaredNorm() - omgSqrMax;
                    cos_theta = 1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2));
                    violaTheta = acos(cos_theta) - thetaMax;
                    violaThrust = (thr - thrustMean) * (thr - thrustMean) - thrustSqrRadi;

                    gradThr = 0.0;
                    gradQuat.setZero();
                    gradPos.setZero(), gradVel.setZero(), gradOmg.setZero();
                    pena = 0.0;

                    L = hIdx(i);
                    K = hPolys[L].rows();
                    for (int k = 0; k < K; k++)
                    {
                        outerNormal = hPolys[L].block<1, 3>(k, 0);
                        violaPos = outerNormal.dot(pos) + hPolys[L](k, 3);
                        if (smoothedL1(violaPos, smoothFactor, violaPosPena, violaPosPenaD))
                        {
                            gradPos += weightPos * violaPosPenaD * outerNormal;
                            pena += weightPos * violaPosPena;
                        }
                    }

                    if (smoothedL1(violaVel, smoothFactor, violaVelPena, violaVelPenaD))
                    {
                        gradVel += weightVel * violaVelPenaD * 2.0 * vel;
                        pena += weightVel * violaVelPena;
                    }

                    if (smoothedL1(violaOmg, smoothFactor, violaOmgPena, violaOmgPenaD))
                    {
                        gradOmg += weightOmg * violaOmgPenaD * 2.0 * omg;
                        pena += weightOmg * violaOmgPena;
                    }

                    if (smoothedL1(violaTheta, smoothFactor, violaThetaPena, violaThetaPenaD))
                    {
                        gradQuat += weightTheta * violaThetaPenaD /
                                    sqrt(1.0 - cos_theta * cos_theta) * 4.0 *
                                    Eigen::Vector4d(0.0, quat(1), quat(2), 0.0);
                        pena += weightTheta * violaThetaPena;
                    }

                    if (smoothedL1(violaThrust, smoothFactor, violaThrustPena, violaThrustPenaD))
                    {
                        gradThr += weightThrust * violaThrustPenaD * 2.0 * (thr - thrustMean);
                        pena += weightThrust * violaThrustPena;
                    }

                    flatMap.backward(gradPos, gradVel, gradThr, gradQuat, gradOmg,
                                     totalGradPos, totalGradVel, totalGradAcc, totalGradJer,
                                     totalGradPsi, totalGradPsiD);

                    node = (j == 0 || j == integralResolution) ? 0.5 : 1.0;
                    alpha = j * integralFrac;
                    gradC.block<6, 3>(i * 6, 0) += (beta0 * totalGradPos.transpose() +
                                                    beta1 * totalGradVel.transpose() +
                                                    beta2 * totalGradAcc.transpose() +
                                                    beta3 * totalGradJer.transpose()) *
                                                   node * step;
                    gradT(i) += (totalGradPos.dot(vel) +
                                 totalGradVel.dot(acc) +
                                 totalGradAcc.dot(jer) +
                                 totalGradJer.dot(sna)) *
                                    alpha * node * step +
                                node * integralFrac * pena;
                    cost += node * step * pena;
                }
            }

            return;
        }

        static inline void attachPenaltyFunctionalCostOnly(const Eigen::VectorXd &T,
                                                           const Eigen::MatrixX3d &coeffs,
                                                           const Eigen::VectorXi &hIdx,
                                                           const PolyhedraH &hPolys,
                                                           const double &smoothFactor,
                                                           const int &integralResolution,
                                                           const Eigen::VectorXd &magnitudeBounds,
                                                           const Eigen::VectorXd &penaltyWeights,
                                                           flatness::FlatnessMap &flatMap,
                                                           double &cost,
                                                           TrajectoryViolationMetrics *metrics)
        {
            const double velSqrMax = magnitudeBounds(0) * magnitudeBounds(0);
            const double omgSqrMax = magnitudeBounds(1) * magnitudeBounds(1);
            const double thetaMax = magnitudeBounds(2);
            const double thrustMean = 0.5 * (magnitudeBounds(3) + magnitudeBounds(4));
            const double thrustRadi = 0.5 * fabs(magnitudeBounds(4) - magnitudeBounds(3));
            const double thrustSqrRadi = thrustRadi * thrustRadi;

            const double weightPos = penaltyWeights(0);
            const double weightVel = penaltyWeights(1);
            const double weightOmg = penaltyWeights(2);
            const double weightTheta = penaltyWeights(3);
            const double weightThrust = penaltyWeights(4);

            Eigen::Vector3d pos, vel, acc, jer;
            double thr, cosTheta;
            Eigen::Vector4d quat;
            Eigen::Vector3d omg;

            double step;
            double s1, s2, s3, s4, s5;
            Eigen::Matrix<double, 6, 1> beta0, beta1, beta2, beta3;
            Eigen::Vector3d outerNormal;
            int K, L;
            double violaPos, violaVel, violaOmg, violaTheta, violaThrust;
            double violaPosPena, violaVelPena, violaOmgPena, violaThetaPena, violaThrustPena;
            double pena, dummyGrad, node;

            const int pieceNum = T.size();
            const double integralFrac = 1.0 / integralResolution;
            for (int i = 0; i < pieceNum; i++)
            {
                const Eigen::Matrix<double, 6, 3> &c = coeffs.block<6, 3>(i * 6, 0);
                step = T(i) * integralFrac;
                for (int j = 0; j <= integralResolution; j++)
                {
                    s1 = j * step;
                    s2 = s1 * s1;
                    s3 = s2 * s1;
                    s4 = s2 * s2;
                    s5 = s4 * s1;
                    beta0(0) = 1.0, beta0(1) = s1, beta0(2) = s2, beta0(3) = s3, beta0(4) = s4, beta0(5) = s5;
                    beta1(0) = 0.0, beta1(1) = 1.0, beta1(2) = 2.0 * s1, beta1(3) = 3.0 * s2, beta1(4) = 4.0 * s3, beta1(5) = 5.0 * s4;
                    beta2(0) = 0.0, beta2(1) = 0.0, beta2(2) = 2.0, beta2(3) = 6.0 * s1, beta2(4) = 12.0 * s2, beta2(5) = 20.0 * s3;
                    beta3(0) = 0.0, beta3(1) = 0.0, beta3(2) = 0.0, beta3(3) = 6.0, beta3(4) = 24.0 * s1, beta3(5) = 60.0 * s2;
                    pos = c.transpose() * beta0;
                    vel = c.transpose() * beta1;
                    acc = c.transpose() * beta2;
                    jer = c.transpose() * beta3;

                    flatMap.forward(vel, acc, jer, 0.0, 0.0, thr, quat, omg);

                    violaVel = vel.squaredNorm() - velSqrMax;
                    violaOmg = omg.squaredNorm() - omgSqrMax;
                    cosTheta = 1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2));
                    cosTheta = std::max(-1.0, std::min(1.0, cosTheta));
                    violaTheta = acos(cosTheta) - thetaMax;
                    violaThrust = (thr - thrustMean) * (thr - thrustMean) - thrustSqrRadi;

                    pena = 0.0;

                    L = hIdx(i);
                    K = hPolys[L].rows();
                    for (int k = 0; k < K; k++)
                    {
                        outerNormal = hPolys[L].block<1, 3>(k, 0);
                        violaPos = outerNormal.dot(pos) + hPolys[L](k, 3);
                        if (smoothedL1(violaPos, smoothFactor, violaPosPena, dummyGrad))
                        {
                            pena += weightPos * violaPosPena;
                        }
                        if (metrics)
                        {
                            metrics->max_corridor_violation =
                                std::max(metrics->max_corridor_violation, std::max(0.0, violaPos));
                        }
                    }

                    if (smoothedL1(violaVel, smoothFactor, violaVelPena, dummyGrad))
                    {
                        pena += weightVel * violaVelPena;
                    }
                    if (metrics)
                    {
                        metrics->max_velocity_violation =
                            std::max(metrics->max_velocity_violation,
                                     std::max(0.0, vel.norm() - magnitudeBounds(0)));
                    }

                    if (smoothedL1(violaOmg, smoothFactor, violaOmgPena, dummyGrad))
                    {
                        pena += weightOmg * violaOmgPena;
                    }
                    if (metrics)
                    {
                        metrics->max_body_rate_violation =
                            std::max(metrics->max_body_rate_violation,
                                     std::max(0.0, omg.norm() - magnitudeBounds(1)));
                    }

                    if (smoothedL1(violaTheta, smoothFactor, violaThetaPena, dummyGrad))
                    {
                        pena += weightTheta * violaThetaPena;
                    }
                    if (metrics)
                    {
                        metrics->max_tilt_violation =
                            std::max(metrics->max_tilt_violation, std::max(0.0, violaTheta));
                    }

                    if (smoothedL1(violaThrust, smoothFactor, violaThrustPena, dummyGrad))
                    {
                        pena += weightThrust * violaThrustPena;
                    }
                    if (metrics)
                    {
                        metrics->max_thrust_violation =
                            std::max(metrics->max_thrust_violation,
                                     std::max(std::max(0.0, magnitudeBounds(3) - thr),
                                              std::max(0.0, thr - magnitudeBounds(4))));
                    }

                    node = (j == 0 || j == integralResolution) ? 0.5 : 1.0;
                    cost += node * step * pena;
                    if (metrics)
                    {
                        metrics->penalty_cost += node * step * pena;
                        ++metrics->sample_count;
                    }
                }
            }
        }

        static inline void accumulateViolationIntegrals(const Eigen::VectorXd &T,
                                                        const Eigen::MatrixX3d &coeffs,
                                                        const Eigen::VectorXi &hIdx,
                                                        const PolyhedraH &hPolys,
                                                        const int &integralResolution,
                                                        const Eigen::VectorXd &magnitudeBounds,
                                                        flatness::FlatnessMap &flatMap,
                                                        double &corridor_integral,
                                                        double &velocity_integral,
                                                        double &body_rate_integral,
                                                        double &tilt_integral,
                                                        double &thrust_integral,
                                                        TrajectoryViolationMetrics *metrics)
        {
            corridor_integral = 0.0;
            velocity_integral = 0.0;
            body_rate_integral = 0.0;
            tilt_integral = 0.0;
            thrust_integral = 0.0;

            if (metrics)
            {
                *metrics = TrajectoryViolationMetrics();
            }

            const double velMax = magnitudeBounds(0);
            const double bodyRateMax = magnitudeBounds(1);
            const double tiltMax = magnitudeBounds(2);
            const double thrustMin = magnitudeBounds(3);
            const double thrustMax = magnitudeBounds(4);

            Eigen::Vector3d pos, vel, acc, jer;
            double thr, cosTheta;
            Eigen::Vector4d quat;
            Eigen::Vector3d omg;

            double step, node;
            double s1, s2, s3, s4, s5;
            Eigen::Matrix<double, 6, 1> beta0, beta1, beta2, beta3;
            Eigen::Vector3d outerNormal;
            int K, L;
            double corridor_violation, velocity_violation, body_rate_violation, tilt_violation, thrust_violation;

            const int pieceNum = T.size();
            const double integralFrac = 1.0 / integralResolution;
            for (int i = 0; i < pieceNum; i++)
            {
                const Eigen::Matrix<double, 6, 3> &c = coeffs.block<6, 3>(i * 6, 0);
                step = T(i) * integralFrac;
                for (int j = 0; j <= integralResolution; j++)
                {
                    s1 = j * step;
                    s2 = s1 * s1;
                    s3 = s2 * s1;
                    s4 = s2 * s2;
                    s5 = s4 * s1;
                    beta0(0) = 1.0, beta0(1) = s1, beta0(2) = s2, beta0(3) = s3, beta0(4) = s4, beta0(5) = s5;
                    beta1(0) = 0.0, beta1(1) = 1.0, beta1(2) = 2.0 * s1, beta1(3) = 3.0 * s2, beta1(4) = 4.0 * s3, beta1(5) = 5.0 * s4;
                    beta2(0) = 0.0, beta2(1) = 0.0, beta2(2) = 2.0, beta2(3) = 6.0 * s1, beta2(4) = 12.0 * s2, beta2(5) = 20.0 * s3;
                    beta3(0) = 0.0, beta3(1) = 0.0, beta3(2) = 0.0, beta3(3) = 6.0, beta3(4) = 24.0 * s1, beta3(5) = 60.0 * s2;

                    pos = c.transpose() * beta0;
                    vel = c.transpose() * beta1;
                    acc = c.transpose() * beta2;
                    jer = c.transpose() * beta3;
                    flatMap.forward(vel, acc, jer, 0.0, 0.0, thr, quat, omg);

                    corridor_violation = 0.0;
                    L = hIdx(i);
                    K = hPolys[L].rows();
                    for (int k = 0; k < K; k++)
                    {
                        outerNormal = hPolys[L].block<1, 3>(k, 0);
                        corridor_violation =
                            std::max(corridor_violation,
                                     std::max(0.0, outerNormal.dot(pos) + hPolys[L](k, 3)));
                    }

                    velocity_violation = std::max(0.0, vel.norm() - velMax);
                    body_rate_violation = std::max(0.0, omg.norm() - bodyRateMax);
                    cosTheta = 1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2));
                    cosTheta = std::max(-1.0, std::min(1.0, cosTheta));
                    tilt_violation = std::max(0.0, acos(cosTheta) - tiltMax);
                    thrust_violation = std::max(std::max(0.0, thrustMin - thr),
                                                std::max(0.0, thr - thrustMax));

                    node = (j == 0 || j == integralResolution) ? 0.5 : 1.0;
                    corridor_integral += node * step * corridor_violation;
                    velocity_integral += node * step * velocity_violation;
                    body_rate_integral += node * step * body_rate_violation;
                    tilt_integral += node * step * tilt_violation;
                    thrust_integral += node * step * thrust_violation;

                    if (metrics)
                    {
                        metrics->max_corridor_violation =
                            std::max(metrics->max_corridor_violation, corridor_violation);
                        metrics->max_velocity_violation =
                            std::max(metrics->max_velocity_violation, velocity_violation);
                        metrics->max_body_rate_violation =
                            std::max(metrics->max_body_rate_violation, body_rate_violation);
                        metrics->max_tilt_violation =
                            std::max(metrics->max_tilt_violation, tilt_violation);
                        metrics->max_thrust_violation =
                            std::max(metrics->max_thrust_violation, thrust_violation);
                        ++metrics->sample_count;
                    }
                }
            }
        }

        static inline double approximateTrajectoryLength(const Trajectory<5> &traj,
                                                         const int samplePerPiece)
        {
            if (traj.getPieceNum() <= 0)
            {
                return std::numeric_limits<double>::infinity();
            }

            const int segments = std::max(2, samplePerPiece);
            double length = 0.0;
            for (const Piece<5> &piece : traj)
            {
                Eigen::Vector3d previous = piece.getPos(0.0);
                for (int i = 1; i <= segments; ++i)
                {
                    const double alpha = static_cast<double>(i) / static_cast<double>(segments);
                    const Eigen::Vector3d current = piece.getPos(alpha * piece.getDuration());
                    length += (current - previous).norm();
                    previous = current;
                }
            }

            return length;
        }

        static inline double budgetedCostFunctional(void *ptr,
                                                    const Eigen::VectorXd &x,
                                                    Eigen::VectorXd &g)
        {
            LBFGSSolveContext &ctx = *(LBFGSSolveContext *)ptr;
            if (ctx.options.max_evaluations > 0 && ctx.eval_count >= ctx.options.max_evaluations)
            {
                ctx.hit_eval_budget = true;
                throw BudgetExceededException("evaluation budget reached");
            }

            if (ctx.options.max_wall_time > 0.0)
            {
                const std::chrono::duration<double> elapsed =
                    std::chrono::steady_clock::now() - ctx.start_time;
                if (elapsed.count() >= ctx.options.max_wall_time)
                {
                    ctx.hit_time_budget = true;
                    throw BudgetExceededException("wall-clock budget reached");
                }
            }

            const double cost = costFunctional(ctx.solver, x, g);
            ++ctx.eval_count;
            if (std::isfinite(cost) && cost < ctx.best_cost)
            {
                ctx.best_cost = cost;
                ctx.best_x = x;
            }
            return cost;
        }

        static inline int budgetedLBFGSProgress(void *ptr,
                                                const Eigen::VectorXd &x,
                                                const Eigen::VectorXd &g,
                                                const double fx,
                                                const double step,
                                                const int k,
                                                const int ls)
        {
            (void)x;
            (void)g;
            (void)fx;
            (void)step;
            (void)ls;
            LBFGSSolveContext &ctx = *(LBFGSSolveContext *)ptr;
            ctx.iterations = k;
            return 0;
        }

        static inline double costFunctional(void *ptr,
                                            const Eigen::VectorXd &x,
                                            Eigen::VectorXd &g)
        {
            GCOPTER_PolytopeSFC &obj = *(GCOPTER_PolytopeSFC *)ptr;
            const int dimTau = obj.temporalDim;
            const int dimXi = obj.spatialDim;
            const double weightT = obj.rho;
            Eigen::Map<const Eigen::VectorXd> tau(x.data(), dimTau);
            Eigen::Map<const Eigen::VectorXd> xi(x.data() + dimTau, dimXi);
            Eigen::Map<Eigen::VectorXd> gradTau(g.data(), dimTau);
            Eigen::Map<Eigen::VectorXd> gradXi(g.data() + dimTau, dimXi);

            forwardT(tau, obj.times);
            forwardP(xi, obj.vPolyIdx, obj.vPolytopes, obj.points);

            double cost;
            obj.minco.setParameters(obj.points, obj.times);
            obj.minco.getEnergy(cost);
            obj.minco.getEnergyPartialGradByCoeffs(obj.partialGradByCoeffs);
            obj.minco.getEnergyPartialGradByTimes(obj.partialGradByTimes);

            attachPenaltyFunctional(obj.times, obj.minco.getCoeffs(),
                                    obj.hPolyIdx, obj.hPolytopes,
                                    obj.smoothEps, obj.integralRes,
                                    obj.magnitudeBd, obj.penaltyWt, obj.flatmap,
                                    cost, obj.partialGradByTimes, obj.partialGradByCoeffs);

            obj.minco.propogateGrad(obj.partialGradByCoeffs, obj.partialGradByTimes,
                                    obj.gradByPoints, obj.gradByTimes);

            cost += weightT * obj.times.sum();
            obj.gradByTimes.array() += weightT;

            backwardGradT(tau, obj.gradByTimes, gradTau);
            backwardGradP(xi, obj.vPolyIdx, obj.vPolytopes, obj.gradByPoints, gradXi);
            normRetrictionLayer(xi, obj.vPolyIdx, obj.vPolytopes, cost, gradXi);

            return cost;
        }

        static inline double costDistance(void *ptr,
                                          const Eigen::VectorXd &xi,
                                          Eigen::VectorXd &gradXi)
        {
            void **dataPtrs = (void **)ptr;
            const double &dEps = *((const double *)(dataPtrs[0]));
            const Eigen::Vector3d &ini = *((const Eigen::Vector3d *)(dataPtrs[1]));
            const Eigen::Vector3d &fin = *((const Eigen::Vector3d *)(dataPtrs[2]));
            const PolyhedraV &vPolys = *((PolyhedraV *)(dataPtrs[3]));

            double cost = 0.0;
            const int overlaps = vPolys.size() / 2;

            Eigen::Matrix3Xd gradP = Eigen::Matrix3Xd::Zero(3, overlaps);
            Eigen::Vector3d a, b, d;
            Eigen::VectorXd r;
            double smoothedDistance;
            for (int i = 0, j = 0, k = 0; i <= overlaps; i++, j += k)
            {
                a = i == 0 ? ini : b;
                if (i < overlaps)
                {
                    k = vPolys[2 * i + 1].cols();
                    Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);
                    r = q.normalized().head(k - 1);
                    b = vPolys[2 * i + 1].rightCols(k - 1) * r.cwiseProduct(r) +
                        vPolys[2 * i + 1].col(0);
                }
                else
                {
                    b = fin;
                }

                d = b - a;
                smoothedDistance = sqrt(d.squaredNorm() + dEps);
                cost += smoothedDistance;

                if (i < overlaps)
                {
                    gradP.col(i) += d / smoothedDistance;
                }
                if (i > 0)
                {
                    gradP.col(i - 1) -= d / smoothedDistance;
                }
            }

            Eigen::VectorXd unitQ;
            double sqrNormQ, invNormQ, sqrNormViolation, c, dc;
            for (int i = 0, j = 0, k; i < overlaps; i++, j += k)
            {
                k = vPolys[2 * i + 1].cols();
                Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);
                Eigen::Map<Eigen::VectorXd> gradQ(gradXi.data() + j, k);
                sqrNormQ = q.squaredNorm();
                invNormQ = 1.0 / sqrt(sqrNormQ);
                unitQ = q * invNormQ;
                gradQ.head(k - 1) = (vPolys[2 * i + 1].rightCols(k - 1).transpose() * gradP.col(i)).array() *
                                    unitQ.head(k - 1).array() * 2.0;
                gradQ(k - 1) = 0.0;
                gradQ = (gradQ - unitQ * unitQ.dot(gradQ)).eval() * invNormQ;

                sqrNormViolation = sqrNormQ - 1.0;
                if (sqrNormViolation > 0.0)
                {
                    c = sqrNormViolation * sqrNormViolation;
                    dc = 3.0 * c;
                    c *= sqrNormViolation;
                    cost += c;
                    gradQ += dc * 2.0 * q;
                }
            }

            return cost;
        }

        static inline void getShortestPath(const Eigen::Vector3d &ini,
                                           const Eigen::Vector3d &fin,
                                           const PolyhedraV &vPolys,
                                           const double &smoothD,
                                           Eigen::Matrix3Xd &path)
        {
            const int overlaps = vPolys.size() / 2;
            Eigen::VectorXi vSizes(overlaps);
            for (int i = 0; i < overlaps; i++)
            {
                vSizes(i) = vPolys[2 * i + 1].cols();
            }
            Eigen::VectorXd xi(vSizes.sum());
            for (int i = 0, j = 0; i < overlaps; i++)
            {
                xi.segment(j, vSizes(i)).setConstant(sqrt(1.0 / vSizes(i)));
                j += vSizes(i);
            }

            double minDistance;
            void *dataPtrs[4];
            dataPtrs[0] = (void *)(&smoothD);
            dataPtrs[1] = (void *)(&ini);
            dataPtrs[2] = (void *)(&fin);
            dataPtrs[3] = (void *)(&vPolys);
            lbfgs::lbfgs_parameter_t shortest_path_params;
            shortest_path_params.past = 3;
            shortest_path_params.delta = 1.0e-3;
            shortest_path_params.g_epsilon = 1.0e-5;

            lbfgs::lbfgs_optimize(xi,
                                  minDistance,
                                  &GCOPTER_PolytopeSFC::costDistance,
                                  nullptr,
                                  nullptr,
                                  dataPtrs,
                                  shortest_path_params);

            path.resize(3, overlaps + 2);
            path.leftCols<1>() = ini;
            path.rightCols<1>() = fin;
            Eigen::VectorXd r;
            for (int i = 0, j = 0, k; i < overlaps; i++, j += k)
            {
                k = vPolys[2 * i + 1].cols();
                Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);
                r = q.normalized().head(k - 1);
                path.col(i + 1) = vPolys[2 * i + 1].rightCols(k - 1) * r.cwiseProduct(r) +
                                  vPolys[2 * i + 1].col(0);
            }

            return;
        }

        static inline bool processCorridor(const PolyhedraH &hPs,
                                           PolyhedraV &vPs)
        {
            const int sizeCorridor = hPs.size() - 1;

            vPs.clear();
            vPs.reserve(2 * sizeCorridor + 1);

            int nv;
            PolyhedronH curIH;
            PolyhedronV curIV, curIOB;
            for (int i = 0; i < sizeCorridor; i++)
            {
                if (!geo_utils::enumerateVs(hPs[i], curIV))
                {
                    return false;
                }
                nv = curIV.cols();
                curIOB.resize(3, nv);
                curIOB.col(0) = curIV.col(0);
                curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
                vPs.push_back(curIOB);

                curIH.resize(hPs[i].rows() + hPs[i + 1].rows(), 4);
                curIH.topRows(hPs[i].rows()) = hPs[i];
                curIH.bottomRows(hPs[i + 1].rows()) = hPs[i + 1];
                if (!geo_utils::enumerateVs(curIH, curIV))
                {
                    return false;
                }
                nv = curIV.cols();
                curIOB.resize(3, nv);
                curIOB.col(0) = curIV.col(0);
                curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
                vPs.push_back(curIOB);
            }

            if (!geo_utils::enumerateVs(hPs.back(), curIV))
            {
                return false;
            }
            nv = curIV.cols();
            curIOB.resize(3, nv);
            curIOB.col(0) = curIV.col(0);
            curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
            vPs.push_back(curIOB);

            return true;
        }

        static inline void setInitial(const Eigen::Matrix3Xd &path,
                                      const double &speed,
                                      const Eigen::VectorXi &intervalNs,
                                      Eigen::Matrix3Xd &innerPoints,
                                      Eigen::VectorXd &timeAlloc)
        {
            const int sizeM = intervalNs.size();
            const int sizeN = intervalNs.sum();
            innerPoints.resize(3, sizeN - 1);
            timeAlloc.resize(sizeN);

            Eigen::Vector3d a, b, c;
            for (int i = 0, j = 0, k = 0, l; i < sizeM; i++)
            {
                l = intervalNs(i);
                a = path.col(i);
                b = path.col(i + 1);
                c = (b - a) / l;
                timeAlloc.segment(j, l).setConstant(c.norm() / speed);
                j += l;
                for (int m = 0; m < l; m++)
                {
                    if (i > 0 || m > 0)
                    {
                        innerPoints.col(k++) = a + c * m;
                    }
                }
            }
        }

        inline int getMetaBetaDim() const
        {
            return std::max(0, temporalDim - 1);
        }

        inline int getMetaGammaIndex() const
        {
            return getMetaBetaDim();
        }

        inline int getMetaDecisionDim() const
        {
            return metaDecisionDim;
        }

        inline void computeMetaLayout()
        {
            metaTemporalDim = temporalDim;
            const int block_count = std::max(0, pieceN - 1);
            spatialBlockSizes.resize(block_count);
            spatialBlockOffsets.resize(block_count);
            metaSpatialOffsets.resize(block_count);

            metaSpatialDim = 0;
            for (int i = 0, raw_offset = 0, meta_offset = 0; i < block_count; ++i)
            {
                const int block_size = vPolytopes[vPolyIdx(i)].cols();
                spatialBlockSizes(i) = block_size;
                spatialBlockOffsets(i) = raw_offset;
                metaSpatialOffsets(i) = meta_offset;
                raw_offset += block_size;
                meta_offset += std::max(0, block_size - 1);
                metaSpatialDim = meta_offset;
            }

            metaDecisionDim = metaTemporalDim + metaSpatialDim;
        }

        inline void computeMetaTimeFloor(const double scale)
        {
            metaTimeFloor.resize(temporalDim);
            if (temporalDim <= 0)
            {
                return;
            }

            Eigen::Matrix3Xd dummy_points;
            const double speed = std::max(positiveEps(), magnitudeBd(0));
            setInitial(shortPath, speed, pieceIdx, dummy_points, metaTimeFloor);
            metaTimeFloor.array() *= std::max(scale, positiveEps());
            metaTimeFloor.array() = metaTimeFloor.array().max(positiveEps());
        }

        inline Eigen::VectorXd getMetaInitialGuessFromX(const Eigen::VectorXd &x0,
                                                        const IGOSolveOptions &options) const
        {
            Eigen::VectorXd z = Eigen::VectorXd::Zero(getMetaDecisionDim());
            const double eps = positiveEps();
            const bool x0_valid =
                x0.size() == getDecisionDim() && x0.allFinite();
            const Eigen::VectorXd x_buffer =
                x0_valid ? x0 : Eigen::VectorXd::Zero(getDecisionDim());
            const Eigen::Map<const Eigen::VectorXd> tau(x_buffer.data(), temporalDim);
            const Eigen::Map<const Eigen::VectorXd> xi(x_buffer.data() + temporalDim, spatialDim);

            if (temporalDim > 0)
            {
                Eigen::VectorXd floor =
                    metaTimeFloor.size() == temporalDim
                        ? metaTimeFloor
                        : Eigen::VectorXd::Constant(temporalDim, eps);
                const double init_scale =
                    std::max(1.0 + 1.0e-3, options.meta_initial_time_scale);
                const Eigen::VectorXd base_seed =
                    (floor.array() * init_scale).max(floor.array() + eps).matrix();

                Eigen::VectorXd TSeed = base_seed;
                if (x0_valid)
                {
                    Eigen::VectorXd T0;
                    forwardT(tau, T0);
                    if (T0.size() == temporalDim && T0.allFinite())
                    {
                        TSeed = T0.cwiseMax(base_seed);
                    }
                }

                const Eigen::VectorXd slack =
                    (TSeed - floor).array().max(eps).matrix();
                const double total_slack = clampPositive(slack.sum(), eps);
                const Eigen::VectorXd ratio = slack / total_slack;
                const double base_ratio =
                    clampPositive(ratio(temporalDim - 1), eps);

                for (int i = 0; i < temporalDim - 1; ++i)
                {
                    z(i) = std::log(clampPositive(ratio(i), eps)) -
                           std::log(base_ratio);
                }
                z(getMetaGammaIndex()) =
                    inverseSoftplus(clampPositive(total_slack - eps, eps));
            }

            for (int i = 0; i < spatialBlockSizes.size(); ++i)
            {
                const int block_size = spatialBlockSizes(i);
                if (block_size <= 1)
                {
                    continue;
                }

                const int raw_offset = spatialBlockOffsets(i);
                const int meta_offset = metaSpatialOffsets(i);
                Eigen::VectorXd q(block_size);
                if (x0_valid)
                {
                    q = xi.segment(raw_offset, block_size);
                    const double norm_q = q.norm();
                    if (!(norm_q > eps) || !q.allFinite())
                    {
                        q.setConstant(std::sqrt(1.0 / static_cast<double>(block_size)));
                    }
                    else
                    {
                        q /= norm_q;
                    }
                }
                else
                {
                    q.setConstant(std::sqrt(1.0 / static_cast<double>(block_size)));
                }

                Eigen::ArrayXd weights = q.array().square();
                weights /= clampPositive(weights.sum(), eps);
                const double base_weight =
                    clampPositive(weights(block_size - 1), eps);
                for (int j = 0; j < block_size - 1; ++j)
                {
                    z(metaTemporalDim + meta_offset + j) =
                        std::log(clampPositive(weights(j), eps)) -
                        std::log(base_weight);
                }
            }

            return z;
        }

        inline void liftMetaToX(const Eigen::VectorXd &z,
                                Eigen::VectorXd &x) const
        {
            x.resize(getDecisionDim());
            if (z.size() != getMetaDecisionDim())
            {
                x.setZero();
                return;
            }

            const double eps = positiveEps();
            Eigen::Map<Eigen::VectorXd> tau(x.data(), temporalDim);
            Eigen::Map<Eigen::VectorXd> xi(x.data() + temporalDim, spatialDim);

            if (temporalDim > 0)
            {
                Eigen::VectorXd floor =
                    metaTimeFloor.size() == temporalDim
                        ? metaTimeFloor
                        : Eigen::VectorXd::Constant(temporalDim, eps);
                Eigen::VectorXd beta_logits = Eigen::VectorXd::Zero(temporalDim);
                if (temporalDim > 1)
                {
                    beta_logits.head(temporalDim - 1) =
                        z.head(temporalDim - 1);
                }
                const Eigen::VectorXd ratio = softmax(beta_logits);
                const double total_slack =
                    softplus(z(getMetaGammaIndex())) + eps;
                const Eigen::VectorXd T =
                    floor + ratio * total_slack;
                backwardT(T.array().max(eps).matrix(), tau);
            }

            for (int i = 0; i < spatialBlockSizes.size(); ++i)
            {
                const int block_size = spatialBlockSizes(i);
                const int raw_offset = spatialBlockOffsets(i);
                if (block_size <= 0)
                {
                    continue;
                }

                if (block_size == 1)
                {
                    xi(raw_offset) = 1.0;
                    continue;
                }

                const int meta_offset = metaSpatialOffsets(i);
                Eigen::VectorXd logits = Eigen::VectorXd::Zero(block_size);
                logits.head(block_size - 1) =
                    z.segment(metaTemporalDim + meta_offset, block_size - 1);
                const Eigen::VectorXd weights = softmax(logits);
                xi.segment(raw_offset, block_size) =
                    weights.array().max(eps).sqrt().matrix();
            }
        }

        inline double evaluateMetaObjectiveOnly(const Eigen::VectorXd &z,
                                                TrajectoryViolationMetrics *metrics = nullptr,
                                                Trajectory<5> *traj = nullptr)
        {
            Eigen::VectorXd x;
            liftMetaToX(z, x);
            return evaluateObjectiveOnly(x, metrics, traj);
        }

        inline void getIGOXSpaceBounds(const Eigen::VectorXd &initialX,
                                       const IGOSolveOptions &options,
                                       Eigen::VectorXd &lower,
                                       Eigen::VectorXd &upper) const
        {
            lower.resize(getDecisionDim());
            upper.resize(getDecisionDim());

            const double tauRadius = std::max(1.0e-3, options.tau_box_radius);
            for (int i = 0; i < temporalDim; ++i)
            {
                const double span = tauRadius * std::max(1.0, std::abs(initialX(i)));
                lower(i) = initialX(i) - span;
                upper(i) = initialX(i) + span;
            }

            double xiBound = std::max(1.0e-3, options.xi_box_bound);
            if (spatialDim > 0)
            {
                xiBound = std::max(xiBound, initialX.tail(spatialDim).cwiseAbs().maxCoeff() + 0.1);
            }
            lower.tail(spatialDim).setConstant(-xiBound);
            upper.tail(spatialDim).setConstant(xiBound);
        }

        inline void getIGOMetaBounds(const Eigen::VectorXd &z0,
                                     const IGOSolveOptions &options,
                                     Eigen::VectorXd &lower,
                                     Eigen::VectorXd &upper) const
        {
            const double eps = positiveEps();
            lower.resize(getMetaDecisionDim());
            upper.resize(getMetaDecisionDim());
            if (getMetaDecisionDim() <= 0)
            {
                return;
            }

            const bool has_valid_center =
                z0.size() == getMetaDecisionDim() && z0.allFinite();
            const double global_logit_limit = 8.0;

            const int beta_dim = getMetaBetaDim();
            if (beta_dim > 0)
            {
                const double time_bound =
                    std::max(1.0e-3, options.meta_time_logit_bound);
                for (int i = 0; i < beta_dim; ++i)
                {
                    const double raw_center = has_valid_center ? z0(i) : 0.0;
                    const double center =
                        std::max(-global_logit_limit,
                                 std::min(global_logit_limit, raw_center));
                    lower(i) = std::max(-global_logit_limit, center - time_bound);
                    upper(i) = std::min(global_logit_limit, center + time_bound);
                }
            }

            const int gamma_index = getMetaGammaIndex();
            const double floor_sum =
                metaTimeFloor.size() == temporalDim
                    ? clampPositive(metaTimeFloor.sum(), eps)
                    : clampPositive(static_cast<double>(temporalDim), eps);
            const double slack_lb =
                std::max(eps, options.meta_total_slack_min_scale * floor_sum);
            const double slack_ub =
                std::max(slack_lb * 1.01,
                         std::max(eps, options.meta_total_slack_max_scale * floor_sum));
            lower(gamma_index) = inverseSoftplus(std::max(eps, slack_lb - eps));
            upper(gamma_index) = inverseSoftplus(std::max(eps, slack_ub - eps));

            if (metaSpatialDim > 0)
            {
                const double spatial_bound =
                    std::max(1.0e-3, options.meta_spatial_logit_bound);
                for (int i = 0; i < metaSpatialDim; ++i)
                {
                    const int idx = metaTemporalDim + i;
                    const double raw_center = has_valid_center ? z0(idx) : 0.0;
                    const double center =
                        std::max(-global_logit_limit,
                                 std::min(global_logit_limit, raw_center));
                    lower(idx) = std::max(-global_logit_limit, center - spatial_bound);
                    upper(idx) = std::min(global_logit_limit, center + spatial_bound);
                }
            }
        }

        inline bool evaluateCandidate(const Eigen::VectorXd &x,
                                      const double feasibility_tol,
                                      EvaluatedCandidate &candidate)
        {
            candidate.x = x;
            TrajectoryViolationMetrics metrics;
            candidate.cost = evaluateObjectiveOnly(x, &metrics, nullptr);
            candidate.max_violation = metrics.maxViolation();
            candidate.feasible =
                std::isfinite(candidate.cost) &&
                candidate.max_violation <= feasibility_tol;
            return std::isfinite(candidate.cost);
        }

        inline double evaluateIGOSearchFitness(const Eigen::VectorXd &x,
                                               TrajectoryViolationMetrics *metrics = nullptr,
                                               Trajectory<5> *traj = nullptr)
        {
            IGOSolveOptions default_options;
            return evaluateMetaPlannerFitness(x, default_options, metrics, traj);
        }

        inline double evaluateMetaPlannerFitness(const Eigen::VectorXd &x,
                                                 const IGOSolveOptions &options,
                                                 TrajectoryViolationMetrics *metrics = nullptr,
                                                 Trajectory<5> *traj = nullptr);

        inline double evaluateWaypointTimeFitness(const Eigen::Matrix3Xd &candidate_points,
                                                  const Eigen::VectorXd &candidate_times,
                                                  const IGOSolveOptions &options,
                                                  TrajectoryViolationMetrics *metrics = nullptr,
                                                  Trajectory<5> *traj = nullptr);

        inline LBFGSSolveOptions makeIGORefineLBFGSOptions(const IGOSolveOptions &options) const
        {
            LBFGSSolveOptions refine_options;
            refine_options.rel_cost_tol = options.refine_rel_cost_tol;
            refine_options.max_iterations = options.refine_max_iterations;
            refine_options.max_evaluations = options.refine_max_evaluations;
            refine_options.max_wall_time = options.refine_max_wall_time;
            refine_options.mem_size = options.refine_mem_size;
            refine_options.past = options.refine_past;
            return refine_options;
        }

    public:
        // magnitudeBounds = [v_max, omg_max, theta_max, thrust_min, thrust_max]^T
        // penaltyWeights = [pos_weight, vel_weight, omg_weight, theta_weight, thrust_weight]^T
        // physicalParams = [vehicle_mass, gravitational_acceleration, horitonral_drag_coeff,
        //                   vertical_drag_coeff, parasitic_drag_coeff, speed_smooth_factor]^T
        inline bool setup(const double &timeWeight,
                          const Eigen::Matrix3d &initialPVA,
                          const Eigen::Matrix3d &terminalPVA,
                          const PolyhedraH &safeCorridor,
                          const double &lengthPerPiece,
                          const double &smoothingFactor,
                          const int &integralResolution,
                          const Eigen::VectorXd &magnitudeBounds,
                          const Eigen::VectorXd &penaltyWeights,
                          const Eigen::VectorXd &physicalParams)
        {
            rho = timeWeight;
            headPVA = initialPVA;
            tailPVA = terminalPVA;

            hPolytopes = safeCorridor;
            for (size_t i = 0; i < hPolytopes.size(); i++)
            {
                const Eigen::ArrayXd norms =
                    hPolytopes[i].leftCols<3>().rowwise().norm();
                hPolytopes[i].array().colwise() /= norms;
            }
            if (!processCorridor(hPolytopes, vPolytopes))
            {
                return false;
            }

            polyN = hPolytopes.size();
            smoothEps = smoothingFactor;
            integralRes = integralResolution;
            magnitudeBd = magnitudeBounds;
            penaltyWt = penaltyWeights;
            physicalPm = physicalParams;
            allocSpeed = magnitudeBd(0) * 3.0;

            getShortestPath(headPVA.col(0), tailPVA.col(0),
                            vPolytopes, smoothEps, shortPath);
            const Eigen::Matrix3Xd deltas = shortPath.rightCols(polyN) - shortPath.leftCols(polyN);
            pieceIdx = (deltas.colwise().norm() / lengthPerPiece).cast<int>().transpose();
            pieceIdx.array() += 1;
            pieceN = pieceIdx.sum();

            temporalDim = pieceN;
            spatialDim = 0;
            vPolyIdx.resize(pieceN - 1);
            hPolyIdx.resize(pieceN);
            for (int i = 0, j = 0, k; i < polyN; i++)
            {
                k = pieceIdx(i);
                for (int l = 0; l < k; l++, j++)
                {
                    if (l < k - 1)
                    {
                        vPolyIdx(j) = 2 * i;
                        spatialDim += vPolytopes[2 * i].cols();
                    }
                    else if (i < polyN - 1)
                    {
                        vPolyIdx(j) = 2 * i + 1;
                        spatialDim += vPolytopes[2 * i + 1].cols();
                    }
                    hPolyIdx(j) = i;
                }
            }

            // Setup for MINCO_S3NU, FlatnessMap, and L-BFGS solver
            minco.setConditions(headPVA, tailPVA, pieceN);
            flatmap.reset(physicalPm(0), physicalPm(1), physicalPm(2),
                          physicalPm(3), physicalPm(4), physicalPm(5));

            // Allocate temp variables
            points.resize(3, pieceN - 1);
            times.resize(pieceN);
            gradByPoints.resize(3, pieceN - 1);
            gradByTimes.resize(pieceN);
            partialGradByCoeffs.resize(6 * pieceN, 3);
            partialGradByTimes.resize(pieceN);

            computeMetaLayout();
            computeMetaTimeFloor(IGOSolveOptions().meta_time_floor_scale);

            return true;
        }

        inline int getTemporalDim() const
        {
            return temporalDim;
        }

        inline int getSpatialDim() const
        {
            return spatialDim;
        }

        inline int getDecisionDim() const
        {
            return temporalDim + spatialDim;
        }

        inline Eigen::VectorXd getCommonInitialGuess()
        {
            Eigen::VectorXd x(getDecisionDim());
            Eigen::Map<Eigen::VectorXd> tau(x.data(), temporalDim);
            Eigen::Map<Eigen::VectorXd> xi(x.data() + temporalDim, spatialDim);

            setInitial(shortPath, allocSpeed, pieceIdx, points, times);
            backwardT(times, tau);
            backwardP(points, vPolyIdx, vPolytopes, xi);

            return x;
        }

        inline double evaluateObjectiveOnly(const Eigen::VectorXd &x,
                                            TrajectoryViolationMetrics *metrics = nullptr,
                                            Trajectory<5> *traj = nullptr)
        {
            if (x.size() != getDecisionDim())
            {
                return std::numeric_limits<double>::infinity();
            }

            if (metrics)
            {
                *metrics = TrajectoryViolationMetrics();
            }

            Eigen::Map<const Eigen::VectorXd> tau(x.data(), temporalDim);
            Eigen::Map<const Eigen::VectorXd> xi(x.data() + temporalDim, spatialDim);

            forwardT(tau, times);
            forwardP(xi, vPolyIdx, vPolytopes, points);

            double cost = 0.0;
            minco.setParameters(points, times);
            minco.getEnergy(cost);

            attachPenaltyFunctionalCostOnly(times, minco.getCoeffs(),
                                            hPolyIdx, hPolytopes,
                                            smoothEps, integralRes,
                                            magnitudeBd, penaltyWt, flatmap,
                                            cost, metrics);

            cost += rho * times.sum();
            normRetrictionLayerCostOnly(xi, vPolyIdx, vPolytopes, cost);

            if (traj)
            {
                minco.getTrajectory(*traj);
            }

            return cost;
        }

        inline bool buildJerkOpt(const Eigen::VectorXd &x,
                                 minco::MINCO_S3NU &jerkOpt) const
        {
            if (x.size() != getDecisionDim() || pieceN <= 0)
            {
                return false;
            }

            Eigen::Map<const Eigen::VectorXd> tau(x.data(), temporalDim);
            Eigen::Map<const Eigen::VectorXd> xi(x.data() + temporalDim, spatialDim);

            Eigen::VectorXd candidate_times;
            Eigen::Matrix3Xd candidate_points;
            forwardT(tau, candidate_times);
            forwardP(xi, vPolyIdx, vPolytopes, candidate_points);

            if (candidate_times.size() != pieceN ||
                candidate_points.cols() != pieceN - 1 ||
                !candidate_times.allFinite() ||
                !candidate_points.allFinite() ||
                (candidate_times.array() <= positiveEps()).any())
            {
                return false;
            }

            jerkOpt.setConditions(headPVA, tailPVA, pieceN);
            jerkOpt.setParameters(candidate_points, candidate_times);
            return true;
        }

        inline bool buildMetaDirectJerkOpt(const SolverResult &result,
                                           minco::MINCO_S3NU &jerkOpt) const
        {
            if (!result.has_meta_direct_solution ||
                result.meta_direct_times.size() <= 0 ||
                result.meta_direct_points.cols() != result.meta_direct_times.size() - 1 ||
                !result.meta_direct_times.allFinite() ||
                !result.meta_direct_points.allFinite() ||
                (result.meta_direct_times.array() <= positiveEps()).any())
            {
                return false;
            }

            jerkOpt.setConditions(headPVA, tailPVA,
                                  static_cast<int>(result.meta_direct_times.size()));
            jerkOpt.setParameters(result.meta_direct_points,
                                  result.meta_direct_times);
            return true;
        }

        inline SolverResult solveLBFGS(const Eigen::VectorXd &initialX,
                                       const LBFGSSolveOptions &options)
        {
            SolverResult result;
            Eigen::VectorXd x;
            if (initialX.size() == getDecisionDim())
            {
                x = initialX;
            }
            else
            {
                x = getCommonInitialGuess();
            }

            LBFGSSolveContext ctx;
            ctx.solver = this;
            ctx.options = options;
            ctx.start_time = std::chrono::steady_clock::now();
            ctx.best_x = x;

            double minCostFunctional = std::numeric_limits<double>::infinity();
            lbfgs_params.mem_size = options.mem_size;
            lbfgs_params.past = options.past;
            lbfgs_params.min_step = 1.0e-32;
            lbfgs_params.g_epsilon = 0.0;
            lbfgs_params.delta = options.rel_cost_tol;
            lbfgs_params.max_iterations = options.max_iterations;

            int ret = lbfgs::LBFGS_CANCELED;
            try
            {
                ret = lbfgs::lbfgs_optimize(x,
                                            minCostFunctional,
                                            &GCOPTER_PolytopeSFC::budgetedCostFunctional,
                                            nullptr,
                                            &GCOPTER_PolytopeSFC::budgetedLBFGSProgress,
                                            &ctx,
                                            lbfgs_params);
            }
            catch (const BudgetExceededException &)
            {
                ret = lbfgs::LBFGS_CANCELED;
            }

            result.wall_time =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - ctx.start_time).count();
            result.solver_status = ret;
            result.iterations = ctx.iterations;
            result.eval_count = ctx.eval_count;
            result.hit_eval_budget = ctx.hit_eval_budget;
            result.hit_time_budget = ctx.hit_time_budget;
            result.converged = (ret == lbfgs::LBFGS_CONVERGENCE || ret == lbfgs::LBFGS_STOP);

            if (ctx.hit_time_budget)
            {
                result.status = "wall-clock budget reached";
            }
            else if (ctx.hit_eval_budget)
            {
                result.status = "evaluation budget reached";
            }
            else
            {
                result.status = lbfgs::lbfgs_strerror(ret);
            }

            if (ctx.best_x.size() == getDecisionDim())
            {
                result.best_x = ctx.best_x;
            }
            else
            {
                result.best_x = x;
            }

            Trajectory<5> traj;
            result.objective = evaluateObjectiveOnly(result.best_x, &result.violations, &traj);
            if (std::isfinite(result.objective) && traj.getPieceNum() > 0)
            {
                result.has_solution = true;
                result.total_duration = traj.getTotalDuration();
                result.trajectory_length = approximateTrajectoryLength(traj, std::max(8, 4 * integralRes));
            }

            return result;
        }

        inline SolverResult solveLBFGSOriginal(const Eigen::VectorXd &initialX,
                                               const double relCostTol)
        {
            LBFGSSolveOptions options;
            options.rel_cost_tol = relCostTol;
            return solveLBFGS(initialX, options);
        }

        inline SolverResult solveLBFGSOriginal(const double relCostTol)
        {
            return solveLBFGSOriginal(getCommonInitialGuess(), relCostTol);
        }

        inline SolverResult solveIGOXSpaceBenchmark(const Eigen::VectorXd &initialX,
                                                    const IGOSolveOptions &options)
        {
            SolverResult result;
            Eigen::VectorXd x0;
            if (initialX.size() == getDecisionDim())
            {
                x0 = initialX;
            }
            else
            {
                x0 = getCommonInitialGuess();
            }

            Eigen::VectorXd lower, upper;
            getIGOXSpaceBounds(x0, options, lower, upper);

            IGO igo;
            IGO::Options igoOptions;
            igoOptions.population = options.population;
            igoOptions.max_iterations = options.max_iterations;
            igoOptions.min_sigma = options.min_sigma;
            igoOptions.eta_mean = options.eta_mean;
            igoOptions.eta_sigma = options.eta_sigma;
            igoOptions.elite_ratio = options.elite_ratio;
            igoOptions.seed = options.seed;

            IGO::Budget budget;
            budget.max_evaluations = options.max_evaluations;
            budget.max_wall_time = options.max_wall_time;

            const int archive_limit =
                std::max(std::max(1, options.archive_top_k), std::max(1, options.refine_top_k));
            std::vector<EvaluatedCandidate> archive;
            const std::chrono::steady_clock::time_point start_time =
                std::chrono::steady_clock::now();
            const IGO::Result igoResult = igo.optimize(
                lower, upper, x0,
                [this, &archive, &options, archive_limit](const Eigen::VectorXd &candidate) -> std::pair<double, bool>
                {
                    EvaluatedCandidate evaluated;
                    const bool is_finite =
                        evaluateCandidate(candidate, options.feasibility_tol, evaluated);
                    insertArchiveCandidate(archive, evaluated, archive_limit);
                    return std::make_pair(evaluated.cost, is_finite);
                },
                igoOptions,
                budget);

            result.wall_time =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
            result.iterations = igoResult.iterations;
            result.eval_count = igoResult.eval_count;
            result.hit_eval_budget = igoResult.hit_eval_budget;
            result.hit_time_budget = igoResult.hit_time_budget;
            result.converged = igoResult.converged;
            result.solver_status = igoResult.converged ? 0 : 1;

            if (igoResult.hit_time_budget)
            {
                result.status = "IGO x-space benchmark (wall-clock budget reached)";
            }
            else if (igoResult.hit_eval_budget)
            {
                result.status = "IGO x-space benchmark (evaluation budget reached)";
            }
            else if (igoResult.converged)
            {
                result.status = "IGO x-space benchmark (converged)";
            }
            else
            {
                result.status = "IGO x-space benchmark";
            }

            if (!archive.empty())
            {
                result.best_x = archive.front().x;
            }
            else
            {
                result.best_x = x0;
            }

            Trajectory<5> traj;
            result.objective = evaluateObjectiveOnly(result.best_x, &result.violations, &traj);
            if (std::isfinite(result.objective) && traj.getPieceNum() > 0)
            {
                result.has_solution = true;
                result.total_duration = traj.getTotalDuration();
                result.trajectory_length = approximateTrajectoryLength(traj, std::max(8, 4 * integralRes));
            }

            return result;
        }

        inline SolverResult solveIGOMetaObjective(const Eigen::VectorXd &initialX,
                                                  const IGOSolveOptions &options)
        {
            SolverResult result;
            Eigen::VectorXd x0;
            if (initialX.size() == getDecisionDim())
            {
                x0 = initialX;
            }
            else
            {
                x0 = getCommonInitialGuess();
            }

            computeMetaTimeFloor(options.meta_time_floor_scale);
            const Eigen::VectorXd z0 = getMetaInitialGuessFromX(x0, options);

            Eigen::VectorXd lower, upper;
            getIGOMetaBounds(z0, options, lower, upper);

            IGO igo;
            IGO::Options igoOptions;
            igoOptions.population = options.population;
            igoOptions.max_iterations = options.max_iterations;
            igoOptions.min_sigma = options.min_sigma;
            igoOptions.eta_mean = options.eta_mean;
            igoOptions.eta_sigma = options.eta_sigma;
            igoOptions.elite_ratio = options.elite_ratio;
            igoOptions.seed = options.seed;

            IGO::Budget budget;
            budget.max_evaluations = options.max_evaluations;
            budget.max_wall_time = options.max_wall_time;

            const int archive_limit =
                std::max(1, options.archive_top_k);
            std::vector<EvaluatedCandidate> archive;
            const std::chrono::steady_clock::time_point start_time =
                std::chrono::steady_clock::now();
            const IGO::Result igoResult = igo.optimize(
                lower, upper, z0,
                [this, &archive, &options, archive_limit](const Eigen::VectorXd &candidate_z) -> std::pair<double, bool>
                {
                    Eigen::VectorXd candidate_x;
                    liftMetaToX(candidate_z, candidate_x);

                    EvaluatedCandidate evaluated;
                    const bool is_finite =
                        evaluateCandidate(candidate_x, options.feasibility_tol, evaluated);
                    insertArchiveCandidate(archive, evaluated, archive_limit);
                    return std::make_pair(evaluated.cost, is_finite);
                },
                igoOptions,
                budget);

            EvaluatedCandidate initial_candidate;
            if (evaluateCandidate(x0, options.feasibility_tol, initial_candidate))
            {
                insertArchiveCandidate(archive, initial_candidate, archive_limit);
            }

            result.wall_time =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
            result.iterations = igoResult.iterations;
            result.eval_count = igoResult.eval_count;
            result.hit_eval_budget = igoResult.hit_eval_budget;
            result.hit_time_budget = igoResult.hit_time_budget;
            result.converged = igoResult.converged;
            result.solver_status = igoResult.converged ? 0 : 1;

            if (igoResult.hit_time_budget)
            {
                result.status = "IGO meta-space corridor objective (wall-clock budget reached)";
            }
            else if (igoResult.hit_eval_budget)
            {
                result.status = "IGO meta-space corridor objective (evaluation budget reached)";
            }
            else if (igoResult.converged)
            {
                result.status = "IGO meta-space corridor objective (converged)";
            }
            else
            {
                result.status = "IGO meta-space corridor objective";
            }

            if (!archive.empty())
            {
                result.best_x = archive.front().x;
            }
            else
            {
                result.best_x = x0;
            }

            Trajectory<5> traj;
            result.objective = evaluateObjectiveOnly(result.best_x, &result.violations, &traj);
            if (std::isfinite(result.objective) && traj.getPieceNum() > 0)
            {
                result.has_solution = true;
                result.total_duration = traj.getTotalDuration();
                result.trajectory_length = approximateTrajectoryLength(traj, std::max(8, 4 * integralRes));
            }

            return result;
        }

        inline SolverResult solveIGOHeuristicPlanner(const Eigen::VectorXd &initialX,
                                                     const IGOSolveOptions &options)
        {
            SolverResult result;
            Eigen::VectorXd x0;
            if (initialX.size() == getDecisionDim())
            {
                x0 = initialX;
            }
            else
            {
                x0 = getCommonInitialGuess();
            }

            computeMetaTimeFloor(options.meta_time_floor_scale);
            const Eigen::VectorXd z0 = getMetaInitialGuessFromX(x0, options);

            Eigen::VectorXd lower, upper;
            getIGOMetaBounds(z0, options, lower, upper);

            IGO igo;
            IGO::Options igoOptions;
            igoOptions.population = options.population;
            igoOptions.max_iterations = options.max_iterations;
            igoOptions.min_sigma = options.min_sigma;
            igoOptions.eta_mean = options.eta_mean;
            igoOptions.eta_sigma = options.eta_sigma;
            igoOptions.elite_ratio = options.elite_ratio;
            igoOptions.seed = options.seed;

            IGO::Budget budget;
            budget.max_evaluations = options.max_evaluations;
            budget.max_wall_time = options.max_wall_time;

            const int archive_limit =
                std::max(std::max(1, options.archive_top_k), std::max(1, options.refine_top_k));
            std::vector<EvaluatedCandidate> archive;
            const std::chrono::steady_clock::time_point start_time =
                std::chrono::steady_clock::now();
            const IGO::Result igoResult = igo.optimize(
                lower, upper, z0,
                [this, &archive, &options, archive_limit](const Eigen::VectorXd &candidate_z) -> std::pair<double, bool>
                {
                    Eigen::VectorXd candidate_x;
                    liftMetaToX(candidate_z, candidate_x);

                    EvaluatedCandidate evaluated;
                    TrajectoryViolationMetrics metrics;
                    evaluated.x = candidate_x;
                    evaluated.cost = evaluateMetaPlannerFitness(candidate_x, options, &metrics, nullptr);
                    evaluated.max_violation = metrics.maxViolation();
                    evaluated.feasible =
                        std::isfinite(evaluated.cost) &&
                        evaluated.max_violation <= options.feasibility_tol;
                    insertArchiveCandidate(archive, evaluated, archive_limit);
                    return std::make_pair(evaluated.cost, std::isfinite(evaluated.cost));
                },
                igoOptions,
                budget);

            result.wall_time =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
            result.iterations = igoResult.iterations;
            result.eval_count = igoResult.eval_count;
            result.hit_eval_budget = igoResult.hit_eval_budget;
            result.hit_time_budget = igoResult.hit_time_budget;
            result.converged = igoResult.converged;
            result.solver_status = igoResult.converged ? 0 : 1;

            if (!archive.empty())
            {
                result.best_x = archive.front().x;
            }
            else
            {
                result.best_x = x0;
            }

            if (result.hit_time_budget)
            {
                result.status = "IGO heuristic planner (wall-clock budget reached)";
            }
            else if (result.hit_eval_budget)
            {
                result.status = "IGO heuristic planner (evaluation budget reached)";
            }
            else if (igoResult.converged)
            {
                result.status = "IGO heuristic planner (converged)";
            }
            else
            {
                result.status = "IGO heuristic planner";
            }

            Trajectory<5> traj;
            result.objective = evaluateObjectiveOnly(result.best_x, &result.violations, &traj);
            if (std::isfinite(result.objective) && traj.getPieceNum() > 0)
            {
                result.has_solution = true;
                result.total_duration = traj.getTotalDuration();
                result.trajectory_length = approximateTrajectoryLength(traj, std::max(8, 4 * integralRes));
            }

            return result;
        }

        inline SolverResult solveIGOMetaPlanner(const Eigen::VectorXd &initialX,
                                                const IGOSolveOptions &options)
        {
            return solveIGOHeuristicPlanner(initialX, options);
        }

        inline SolverResult solveMetaOptimizer(const Eigen::VectorXd &initialX,
                                               const IGOSolveOptions &options);

        inline SolverResult solveIGOSeededLBFGS(const Eigen::VectorXd &initialX,
                                                const IGOSolveOptions &options)
        {
            SolverResult result;
            Eigen::VectorXd x0;
            if (initialX.size() == getDecisionDim())
            {
                x0 = initialX;
            }
            else
            {
                x0 = getCommonInitialGuess();
            }

            computeMetaTimeFloor(options.meta_time_floor_scale);
            const Eigen::VectorXd z0 = getMetaInitialGuessFromX(x0, options);

            Eigen::VectorXd lower, upper;
            getIGOMetaBounds(z0, options, lower, upper);

            IGO igo;
            IGO::Options igoOptions;
            igoOptions.population = options.population;
            igoOptions.max_iterations = options.max_iterations;
            igoOptions.min_sigma = options.min_sigma;
            igoOptions.eta_mean = options.eta_mean;
            igoOptions.eta_sigma = options.eta_sigma;
            igoOptions.elite_ratio = options.elite_ratio;
            igoOptions.seed = options.seed;

            IGO::Budget budget;
            const double outer_budget_ratio =
                std::max(1.0e-3, std::min(1.0, options.outer_budget_ratio));
            if (options.max_evaluations > 0)
            {
                budget.max_evaluations = std::min(
                    options.max_evaluations,
                    std::max(1, static_cast<int>(
                                    std::floor(options.max_evaluations * outer_budget_ratio))));
            }
            if (options.max_wall_time > 0.0)
            {
                budget.max_wall_time =
                    std::min(options.max_wall_time,
                             std::max(1.0e-6, options.max_wall_time * outer_budget_ratio));
            }

            const int archive_limit =
                std::max(std::max(1, options.archive_top_k), std::max(1, options.refine_top_k));
            std::vector<EvaluatedCandidate> archive;
            const std::chrono::steady_clock::time_point start_time =
                std::chrono::steady_clock::now();
            const IGO::Result igoResult = igo.optimize(
                lower, upper, z0,
                [this, &archive, &options, archive_limit](const Eigen::VectorXd &candidate_z) -> std::pair<double, bool>
                {
                    Eigen::VectorXd candidate_x;
                    liftMetaToX(candidate_z, candidate_x);

                    EvaluatedCandidate evaluated;
                    const bool is_finite =
                        evaluateCandidate(candidate_x, options.feasibility_tol, evaluated);
                    insertArchiveCandidate(archive, evaluated, archive_limit);
                    return std::make_pair(evaluated.cost, is_finite);
                },
                igoOptions,
                budget);

            result.wall_time =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
            result.iterations = igoResult.iterations;
            result.eval_count = igoResult.eval_count;
            result.hit_eval_budget = false;
            result.hit_time_budget = false;

            Eigen::VectorXd fallback_seed;
            liftMetaToX(z0, fallback_seed);
            if (fallback_seed.size() != getDecisionDim() || !fallback_seed.allFinite())
            {
                fallback_seed = x0;
            }
            if (!archive.empty())
            {
                fallback_seed = archive.front().x;
            }

            std::vector<Eigen::VectorXd> refine_seeds;
            const int refine_limit = std::max(1, options.refine_top_k);
            auto append_unique_seed =
                [this, &refine_seeds](const Eigen::VectorXd &seed) -> void
            {
                if (seed.size() != getDecisionDim() || !seed.allFinite())
                {
                    return;
                }
                for (const Eigen::VectorXd &existing : refine_seeds)
                {
                    if (sameDecisionVector(existing, seed))
                    {
                        return;
                    }
                }
                refine_seeds.push_back(seed);
            };

            // Always give the GCOPTER initial guess and deterministic fallback
            // a chance to smooth the result before spending budget on sampled seeds.
            append_unique_seed(x0);
            append_unique_seed(fallback_seed);
            for (int i = 0; i < static_cast<int>(archive.size()) && i < refine_limit; ++i)
            {
                append_unique_seed(archive[i].x);
            }
            if (refine_seeds.empty())
            {
                append_unique_seed(fallback_seed);
            }

            std::vector<EvaluatedCandidate> final_candidates;
            auto append_final_candidate =
                [this, &final_candidates](const EvaluatedCandidate &candidate) -> void
            {
                if (candidate.x.size() != getDecisionDim() ||
                    !candidate.x.allFinite() ||
                    !std::isfinite(candidate.cost))
                {
                    return;
                }

                for (std::size_t i = 0; i < final_candidates.size(); ++i)
                {
                    if (sameDecisionVector(final_candidates[i].x, candidate.x))
                    {
                        if (isBetterCandidate(candidate, final_candidates[i]))
                        {
                            final_candidates[i] = candidate;
                        }
                        return;
                    }
                }

                final_candidates.push_back(candidate);
            };
            for (int i = 0; i < static_cast<int>(archive.size()); ++i)
            {
                append_final_candidate(archive[i]);
            }

            EvaluatedCandidate fallback_candidate;
            if (evaluateCandidate(fallback_seed, options.feasibility_tol, fallback_candidate))
            {
                append_final_candidate(fallback_candidate);
            }

            EvaluatedCandidate initial_candidate;
            if (evaluateCandidate(x0, options.feasibility_tol, initial_candidate))
            {
                append_final_candidate(initial_candidate);
            }

            bool any_refine_converged = false;
            const auto remaining_eval_budget =
                [&result, &options]() -> int
            {
                if (options.max_evaluations <= 0)
                {
                    return 0;
                }
                return std::max(0, options.max_evaluations - result.eval_count);
            };
            const auto remaining_wall_time =
                [&start_time, &options]() -> double
            {
                if (options.max_wall_time <= 0.0)
                {
                    return 0.0;
                }
                const double elapsed =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
                return std::max(0.0, options.max_wall_time - elapsed);
            };

            for (const Eigen::VectorXd &seed : refine_seeds)
            {
                if (options.max_evaluations > 0 && remaining_eval_budget() <= 0)
                {
                    result.hit_eval_budget = true;
                    break;
                }
                if (options.max_wall_time > 0.0 && remaining_wall_time() <= 0.0)
                {
                    result.hit_time_budget = true;
                    break;
                }

                LBFGSSolveOptions refine_options =
                    makeIGORefineLBFGSOptions(options);
                if (options.max_evaluations > 0)
                {
                    const int remaining_eval = remaining_eval_budget();
                    if (remaining_eval <= 0)
                    {
                        result.hit_eval_budget = true;
                        break;
                    }
                    refine_options.max_evaluations =
                        refine_options.max_evaluations > 0
                            ? std::min(refine_options.max_evaluations, remaining_eval)
                            : remaining_eval;
                }
                if (options.max_wall_time > 0.0)
                {
                    const double remaining_time = remaining_wall_time();
                    if (remaining_time <= 0.0)
                    {
                        result.hit_time_budget = true;
                        break;
                    }
                    refine_options.max_wall_time =
                        refine_options.max_wall_time > 0.0
                            ? std::min(refine_options.max_wall_time, remaining_time)
                            : remaining_time;
                }

                const SolverResult refine_result = solveLBFGS(seed, refine_options);
                result.iterations += refine_result.iterations;
                result.eval_count += refine_result.eval_count;
                any_refine_converged = any_refine_converged || refine_result.converged;

                if (refine_result.best_x.size() == getDecisionDim())
                {
                    EvaluatedCandidate refined_candidate;
                    refined_candidate.x = refine_result.best_x;
                    refined_candidate.cost = refine_result.objective;
                    refined_candidate.max_violation =
                        refine_result.violations.maxViolation();
                    refined_candidate.feasible =
                        std::isfinite(refined_candidate.cost) &&
                        refined_candidate.max_violation <= options.feasibility_tol;
                    append_final_candidate(refined_candidate);
                }
            }

            if (final_candidates.empty())
            {
                result.best_x = fallback_seed;
            }
            else
            {
                std::sort(final_candidates.begin(), final_candidates.end(),
                          &GCOPTER_PolytopeSFC::isBetterCandidate);
                result.best_x = final_candidates.front().x;
            }

            result.converged = igoResult.converged || any_refine_converged;
            result.solver_status = result.converged ? 0 : 1;
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
            if (options.max_wall_time > 0.0 && elapsed >= options.max_wall_time)
            {
                result.hit_time_budget = true;
            }
            if (options.max_evaluations > 0 && result.eval_count >= options.max_evaluations)
            {
                result.hit_eval_budget = true;
            }

            if (result.hit_time_budget)
            {
                result.status = "IGO meta search + LBFGS refine (wall-clock budget reached)";
            }
            else if (result.hit_eval_budget)
            {
                result.status = "IGO meta search + LBFGS refine (evaluation budget reached)";
            }
            else if (igoResult.converged)
            {
                result.status = "IGO meta search + LBFGS refine (outer IGO converged)";
            }
            else
            {
                result.status = "IGO meta search + LBFGS refine";
            }

            result.wall_time =
                elapsed;

            Trajectory<5> traj;
            result.objective = evaluateObjectiveOnly(result.best_x, &result.violations, &traj);
            if (std::isfinite(result.objective) && traj.getPieceNum() > 0)
            {
                result.has_solution = true;
                result.total_duration = traj.getTotalDuration();
                result.trajectory_length = approximateTrajectoryLength(traj, std::max(8, 4 * integralRes));
            }

            return result;
        }

        inline SolverResult solveIGO(const Eigen::VectorXd &initialX,
                                     const IGOSolveOptions &options);

        inline double optimize(Trajectory<5> &traj,
                               const double &relCostTol)
        {
            const SolverResult result =
                solveLBFGSOriginal(getCommonInitialGuess(), relCostTol);

            if (result.has_solution)
            {
                evaluateObjectiveOnly(result.best_x, nullptr, &traj);
            }
            else
            {
                traj.clear();
                std::cout << "Optimization Failed: "
                          << result.status
                          << std::endl;
                return INFINITY;
            }

            return result.objective;
        }
    };

}

#endif
