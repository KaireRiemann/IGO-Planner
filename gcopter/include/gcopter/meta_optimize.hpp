#ifndef GCOPTER_META_OPTIMIZE_HPP
#define GCOPTER_META_OPTIMIZE_HPP

#include "gcopter/gcopter.hpp"
#include "gcopter/trajectory.hpp"

namespace gcopter
{
    inline double GCOPTER_PolytopeSFC::evaluateMetaPlannerFitness(
        const Eigen::VectorXd &x,
        const IGOSolveOptions &options,
        TrajectoryViolationMetrics *metrics,
        Trajectory<5> *traj)
    {
        if (x.size() != getDecisionDim())
        {
            if (metrics)
            {
                *metrics = TrajectoryViolationMetrics();
            }
            return std::numeric_limits<double>::infinity();
        }

        Eigen::Map<const Eigen::VectorXd> tau(x.data(), temporalDim);
        Eigen::Map<const Eigen::VectorXd> xi(x.data() + temporalDim, spatialDim);

        Eigen::VectorXd candidate_times;
        Eigen::Matrix3Xd candidate_points;
        forwardT(tau, candidate_times);
        forwardP(xi, vPolyIdx, vPolytopes, candidate_points);

        return evaluateWaypointTimeFitness(candidate_points,
                                           candidate_times,
                                           options,
                                           metrics,
                                           traj);
    }

    inline double GCOPTER_PolytopeSFC::evaluateWaypointTimeFitness(
        const Eigen::Matrix3Xd &candidate_points,
        const Eigen::VectorXd &candidate_times,
        const IGOSolveOptions &options,
        TrajectoryViolationMetrics *metrics,
        Trajectory<5> *traj)
    {
        const int point_count = std::max(0, pieceN - 1);
        if (candidate_points.cols() != point_count ||
            candidate_times.size() != pieceN ||
            !candidate_points.allFinite() ||
            !candidate_times.allFinite() ||
            (candidate_times.array() <= positiveEps()).any())
        {
            if (metrics)
            {
                *metrics = TrajectoryViolationMetrics();
            }
            return std::numeric_limits<double>::infinity();
        }

        minco.setParameters(candidate_points, candidate_times);
        double trajectory_energy = 0.0;
        minco.getEnergy(trajectory_energy);
        if (!std::isfinite(trajectory_energy))
        {
            if (metrics)
            {
                *metrics = TrajectoryViolationMetrics();
            }
            return std::numeric_limits<double>::infinity();
        }

        TrajectoryViolationMetrics local_metrics;

        Trajectory<5> local_traj;
        minco.getTrajectory(local_traj);
        if (local_traj.getPieceNum() <= 0)
        {
            return std::numeric_limits<double>::infinity();
        }

        const double sample_dt =
            std::max(1.0e-3, options.meta_optimizer_sample_dt);
        const double max_acceleration =
            std::max(positiveEps(), options.meta_optimizer_max_acceleration);
        const double vel_max = magnitudeBd(0);
        const double body_rate_max = magnitudeBd(1);
        const double tilt_max = magnitudeBd(2);
        const double thrust_min = magnitudeBd(3);
        const double thrust_max = magnitudeBd(4);

        double trajectory_length = 0.0;
        double corridor_integral = 0.0;
        double velocity_integral = 0.0;
        double acceleration_integral = 0.0;
        double body_rate_integral = 0.0;
        double tilt_integral = 0.0;
        double thrust_integral = 0.0;
        Eigen::Vector3d previous_pos;
        bool has_previous_pos = false;

        for (int piece_idx = 0; piece_idx < local_traj.getPieceNum(); ++piece_idx)
        {
            const Piece<5> &piece = local_traj[piece_idx];
            const double duration = std::max(positiveEps(), piece.getDuration());
            const int samples = std::max(1, static_cast<int>(std::ceil(duration / sample_dt)));
            const double step = duration / static_cast<double>(samples);

            for (int sample_idx = 0; sample_idx <= samples; ++sample_idx)
            {
                const double local_t =
                    std::min(duration, step * static_cast<double>(sample_idx));
                const double node =
                    (sample_idx == 0 || sample_idx == samples) ? 0.5 : 1.0;

                const Eigen::Vector3d pos = piece.getPos(local_t);
                const Eigen::Vector3d vel = piece.getVel(local_t);
                const Eigen::Vector3d acc = piece.getAcc(local_t);
                const Eigen::Vector3d jerk = piece.getJer(local_t);

                if (has_previous_pos)
                {
                    trajectory_length += (pos - previous_pos).norm();
                }
                previous_pos = pos;
                has_previous_pos = true;

                double corridor_violation = 0.0;
                if (piece_idx < hPolyIdx.size())
                {
                    const int h_idx = hPolyIdx(piece_idx);
                    if (h_idx >= 0 && h_idx < static_cast<int>(hPolytopes.size()))
                    {
                        const PolyhedronH &h_poly = hPolytopes[h_idx];
                        for (int row = 0; row < h_poly.rows(); ++row)
                        {
                            corridor_violation =
                                std::max(corridor_violation,
                                         std::max(0.0,
                                                  h_poly.block<1, 3>(row, 0).dot(pos) +
                                                      h_poly(row, 3)));
                        }
                    }
                }

                double thrust = 0.0;
                Eigen::Vector4d quat;
                Eigen::Vector3d body_rate;
                flatmap.forward(vel, acc, jerk, 0.0, 0.0,
                                thrust, quat, body_rate);

                const double velocity_violation =
                    std::max(0.0, vel.norm() - vel_max);
                const double acceleration_violation =
                    std::max(0.0, acc.norm() - max_acceleration);
                const double body_rate_violation =
                    std::max(0.0, body_rate.norm() - body_rate_max);
                const double cos_tilt =
                    std::max(-1.0, std::min(1.0,
                                            1.0 - 2.0 * (quat(1) * quat(1) +
                                                         quat(2) * quat(2))));
                const double tilt_violation =
                    std::max(0.0, std::acos(cos_tilt) - tilt_max);
                const double thrust_violation =
                    std::max(std::max(0.0, thrust_min - thrust),
                             std::max(0.0, thrust - thrust_max));

                corridor_integral += node * step * corridor_violation;
                velocity_integral += node * step * velocity_violation;
                acceleration_integral += node * step * acceleration_violation;
                body_rate_integral += node * step * body_rate_violation;
                tilt_integral += node * step * tilt_violation;
                thrust_integral += node * step * thrust_violation;

                local_metrics.max_corridor_violation =
                    std::max(local_metrics.max_corridor_violation, corridor_violation);
                local_metrics.max_velocity_violation =
                    std::max(local_metrics.max_velocity_violation, velocity_violation);
                local_metrics.max_acceleration_violation =
                    std::max(local_metrics.max_acceleration_violation, acceleration_violation);
                local_metrics.max_body_rate_violation =
                    std::max(local_metrics.max_body_rate_violation, body_rate_violation);
                local_metrics.max_tilt_violation =
                    std::max(local_metrics.max_tilt_violation, tilt_violation);
                local_metrics.max_thrust_violation =
                    std::max(local_metrics.max_thrust_violation, thrust_violation);
                ++local_metrics.sample_count;
            }
        }

        const double inv_sample_dt = 1.0 / sample_dt;
        const double penalty_cost =
            std::max(0.0, options.meta_optimizer_collision_weight) * corridor_integral * inv_sample_dt +
            std::max(0.0, options.meta_optimizer_velocity_weight) * velocity_integral * inv_sample_dt +
            std::max(0.0, options.meta_optimizer_acceleration_weight) * acceleration_integral * inv_sample_dt +
            std::max(0.0, options.meta_optimizer_body_rate_weight) * body_rate_integral * inv_sample_dt +
            std::max(0.0, options.meta_optimizer_tilt_weight) * tilt_integral * inv_sample_dt +
            std::max(0.0, options.meta_optimizer_thrust_weight) * thrust_integral * inv_sample_dt;
        const double peak_penalty_cost =
            std::max(0.0, options.meta_optimizer_collision_weight) *
                local_metrics.max_corridor_violation * local_metrics.max_corridor_violation +
            std::max(0.0, options.meta_optimizer_velocity_weight) *
                local_metrics.max_velocity_violation * local_metrics.max_velocity_violation +
            std::max(0.0, options.meta_optimizer_acceleration_weight) *
                local_metrics.max_acceleration_violation * local_metrics.max_acceleration_violation +
            std::max(0.0, options.meta_optimizer_body_rate_weight) *
                local_metrics.max_body_rate_violation * local_metrics.max_body_rate_violation +
            std::max(0.0, options.meta_optimizer_tilt_weight) *
                local_metrics.max_tilt_violation * local_metrics.max_tilt_violation +
            std::max(0.0, options.meta_optimizer_thrust_weight) *
                local_metrics.max_thrust_violation * local_metrics.max_thrust_violation;
        const double fitness =
            std::max(0.0, options.meta_optimizer_time_weight) * candidate_times.sum() +
            std::max(0.0, options.meta_optimizer_length_weight) * trajectory_length +
            std::max(0.0, options.meta_optimizer_energy_weight) * trajectory_energy +
            penalty_cost +
            peak_penalty_cost;

        local_metrics.penalty_cost = penalty_cost + peak_penalty_cost;

        if (metrics)
        {
            *metrics = local_metrics;
        }
        if (traj)
        {
            *traj = local_traj;
        }

        return fitness;
    }

    inline GCOPTER_PolytopeSFC::SolverResult
    GCOPTER_PolytopeSFC::solveMetaOptimizer(const Eigen::VectorXd &initialX,
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

        const int point_count = std::max(0, pieceN - 1);
        const int time_offset = 3 * point_count;
        const int direct_dim = time_offset + temporalDim;
        if (direct_dim <= 0)
        {
            result.best_x = x0;
            Trajectory<5> local_traj;
            result.objective =
                evaluateObjectiveOnly(result.best_x, &result.violations, &local_traj);
            if (std::isfinite(result.objective) && local_traj.getPieceNum() > 0)
            {
                result.has_solution = true;
                result.total_duration = local_traj.getTotalDuration();
                result.trajectory_length =
                    approximateTrajectoryLength(local_traj, std::max(8, 4 * integralRes));
            }
            result.status = "MetaPlanner-style waypoint-time IGO optimizer (empty decision space)";
            return result;
        }

        auto unpack_direct =
            [this, point_count, time_offset, direct_dim](const Eigen::VectorXd &y,
                                                        Eigen::Matrix3Xd &candidate_points,
                                                        Eigen::VectorXd &candidate_times) -> bool
        {
            if (y.size() != direct_dim || !y.allFinite())
            {
                return false;
            }

            candidate_points.resize(3, point_count);
            for (int i = 0; i < point_count; ++i)
            {
                candidate_points.col(i) = y.segment<3>(3 * i);
            }
            candidate_times = y.segment(time_offset, temporalDim);

            return candidate_times.size() == temporalDim &&
                   candidate_times.allFinite() &&
                   (candidate_times.array() > positiveEps()).all();
        };

        auto pack_x_to_direct =
            [this, point_count, time_offset, direct_dim](const Eigen::VectorXd &x) -> Eigen::VectorXd
        {
            Eigen::VectorXd y = Eigen::VectorXd::Zero(direct_dim);
            if (x.size() == getDecisionDim() && x.allFinite())
            {
                Eigen::Map<const Eigen::VectorXd> tau(x.data(), temporalDim);
                Eigen::Map<const Eigen::VectorXd> xi(x.data() + temporalDim, spatialDim);

                Eigen::VectorXd packed_times;
                Eigen::Matrix3Xd packed_points;
                forwardT(tau, packed_times);
                forwardP(xi, vPolyIdx, vPolytopes, packed_points);

                for (int i = 0; i < point_count; ++i)
                {
                    y.segment<3>(3 * i) = packed_points.col(i);
                }
                y.segment(time_offset, temporalDim) = packed_times;
            }
            return y;
        };

        auto direct_to_x =
            [this, &unpack_direct](const Eigen::VectorXd &y,
                                   Eigen::VectorXd &x) -> bool
        {
            Eigen::Matrix3Xd candidate_points;
            Eigen::VectorXd candidate_times;
            if (!unpack_direct(y, candidate_points, candidate_times))
            {
                return false;
            }

            x.resize(getDecisionDim());
            Eigen::Map<Eigen::VectorXd> tau(x.data(), temporalDim);
            Eigen::Map<Eigen::VectorXd> xi(x.data() + temporalDim, spatialDim);
            backwardT(candidate_times, tau);
            backwardP(candidate_points, vPolyIdx, vPolytopes, xi);
            return x.allFinite();
        };

        Eigen::VectorXd y0 = pack_x_to_direct(x0);

        Eigen::VectorXd lower(direct_dim);
        Eigen::VectorXd upper(direct_dim);
        const double eps = positiveEps();
        const double fallback_radius = std::max(1.0, options.xi_box_bound);
        for (int i = 0; i < point_count; ++i)
        {
            Eigen::Vector3d point_min =
                y0.segment<3>(3 * i) - Eigen::Vector3d::Constant(fallback_radius);
            Eigen::Vector3d point_max =
                y0.segment<3>(3 * i) + Eigen::Vector3d::Constant(fallback_radius);

            if (i < vPolyIdx.size())
            {
                const int poly_idx = vPolyIdx(i);
                if (poly_idx >= 0 &&
                    poly_idx < static_cast<int>(vPolytopes.size()) &&
                    vPolytopes[poly_idx].cols() > 0)
                {
                    const PolyhedronV &v_poly = vPolytopes[poly_idx];
                    point_min.setConstant(std::numeric_limits<double>::infinity());
                    point_max.setConstant(-std::numeric_limits<double>::infinity());
                    for (int col = 0; col < v_poly.cols(); ++col)
                    {
                        Eigen::Vector3d vertex = v_poly.col(0);
                        if (col > 0)
                        {
                            vertex += v_poly.col(col);
                        }
                        point_min = point_min.cwiseMin(vertex);
                        point_max = point_max.cwiseMax(vertex);
                    }
                }
            }

            for (int axis = 0; axis < 3; ++axis)
            {
                if (!std::isfinite(point_min(axis)) ||
                    !std::isfinite(point_max(axis)) ||
                    point_max(axis) <= point_min(axis))
                {
                    point_min(axis) = y0(3 * i + axis) - fallback_radius;
                    point_max(axis) = y0(3 * i + axis) + fallback_radius;
                }
            }

            lower.segment<3>(3 * i) =
                point_min - Eigen::Vector3d::Constant(1.0e-3);
            upper.segment<3>(3 * i) =
                point_max + Eigen::Vector3d::Constant(1.0e-3);
        }

        const Eigen::VectorXd floor_times =
            metaTimeFloor.size() == temporalDim
                ? metaTimeFloor
                : Eigen::VectorXd::Constant(temporalDim, eps);
        const double upper_scale =
            std::max(1.01, 1.0 + std::max(0.0, options.meta_total_slack_max_scale));
        const double seed_scale =
            std::max(1.01, options.meta_initial_time_scale);
        for (int i = 0; i < temporalDim; ++i)
        {
            const double floor_time = std::max(eps, floor_times(i));
            const double seed_time =
                std::isfinite(y0(time_offset + i))
                    ? std::max(floor_time, y0(time_offset + i))
                    : floor_time * seed_scale;
            lower(time_offset + i) = floor_time;
            upper(time_offset + i) =
                std::max(lower(time_offset + i) + eps,
                         std::max(floor_time * upper_scale,
                                  seed_time * seed_scale));
        }
        y0 = y0.cwiseMax(lower).cwiseMin(upper);

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

        const int archive_limit = std::max(1, options.archive_top_k);
        struct DirectCandidate
        {
            Eigen::VectorXd y;
            double cost = std::numeric_limits<double>::infinity();
            double max_violation = std::numeric_limits<double>::infinity();
            bool feasible = false;
        };

        std::vector<DirectCandidate> archive;
        auto insert_meta_candidate =
            [&archive, archive_limit](const DirectCandidate &candidate) -> void
        {
            if (!std::isfinite(candidate.cost))
            {
                return;
            }

            archive.push_back(candidate);
            std::sort(archive.begin(), archive.end(),
                      [](const DirectCandidate &lhs,
                         const DirectCandidate &rhs) -> bool
                      {
                          if (lhs.feasible != rhs.feasible)
                          {
                              return lhs.feasible;
                          }
                          const bool lhs_finite = std::isfinite(lhs.cost);
                          const bool rhs_finite = std::isfinite(rhs.cost);
                          if (lhs_finite != rhs_finite)
                          {
                              return lhs_finite;
                          }
                          if (!lhs.feasible && lhs.max_violation != rhs.max_violation)
                          {
                              return lhs.max_violation < rhs.max_violation;
                          }
                          if (lhs.cost != rhs.cost)
                          {
                              return lhs.cost < rhs.cost;
                          }
                          return lhs.max_violation < rhs.max_violation;
                      });

            const std::size_t limit =
                static_cast<std::size_t>(std::max(1, archive_limit));
            if (archive.size() > limit)
            {
                archive.resize(limit);
            }
        };

        auto evaluate_direct_candidate =
            [this, &options, &unpack_direct](const Eigen::VectorXd &candidate_y,
                                             DirectCandidate &evaluated) -> bool
        {
            Eigen::Matrix3Xd candidate_points;
            Eigen::VectorXd candidate_times;
            if (!unpack_direct(candidate_y, candidate_points, candidate_times))
            {
                evaluated.y = candidate_y;
                evaluated.cost = std::numeric_limits<double>::infinity();
                evaluated.max_violation = std::numeric_limits<double>::infinity();
                evaluated.feasible = false;
                return false;
            }

            TrajectoryViolationMetrics local_metrics;
            evaluated.y = candidate_y;
            evaluated.cost =
                evaluateWaypointTimeFitness(candidate_points,
                                            candidate_times,
                                            options,
                                            &local_metrics,
                                            nullptr);
            evaluated.max_violation = local_metrics.maxViolation();
            evaluated.feasible =
                std::isfinite(evaluated.cost) &&
                evaluated.max_violation <= options.feasibility_tol;
            return std::isfinite(evaluated.cost);
        };

        const std::chrono::steady_clock::time_point start_time =
            std::chrono::steady_clock::now();
        const IGO::Result igoResult = igo.optimize(
            lower, upper, y0,
            [&insert_meta_candidate, &evaluate_direct_candidate](const Eigen::VectorXd &candidate_y) -> std::pair<double, bool>
            {
                DirectCandidate evaluated;
                const bool is_finite =
                    evaluate_direct_candidate(candidate_y, evaluated);
                insert_meta_candidate(evaluated);
                return std::make_pair(evaluated.cost, is_finite);
            },
            igoOptions,
            budget);

        DirectCandidate initial_candidate;
        if (evaluate_direct_candidate(y0, initial_candidate))
        {
            insert_meta_candidate(initial_candidate);
        }

        if (!archive.empty())
        {
            if (!direct_to_x(archive.front().y, result.best_x))
            {
                result.best_x = x0;
            }
        }
        else
        {
            result.best_x = x0;
        }

        result.wall_time =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        result.iterations = igoResult.iterations;
        result.eval_count = igoResult.eval_count;
        result.hit_eval_budget = igoResult.hit_eval_budget;
        result.hit_time_budget = igoResult.hit_time_budget;
        result.converged = igoResult.converged;
        result.solver_status = igoResult.converged ? 0 : 1;

        if (result.hit_time_budget)
        {
            result.status = "MetaPlanner-style waypoint-time IGO optimizer (wall-clock budget reached)";
        }
        else if (result.hit_eval_budget)
        {
            result.status = "MetaPlanner-style waypoint-time IGO optimizer (evaluation budget reached)";
        }
        else if (igoResult.converged)
        {
            result.status = "MetaPlanner-style waypoint-time IGO optimizer (converged)";
        }
        else
        {
            result.status = "MetaPlanner-style waypoint-time IGO optimizer";
        }

        Trajectory<5> local_traj;
        result.objective =
            evaluateMetaPlannerFitness(result.best_x, options, &result.violations, &local_traj);
        if (std::isfinite(result.objective) && local_traj.getPieceNum() > 0)
        {
            result.has_solution = true;
            result.total_duration = local_traj.getTotalDuration();
            result.trajectory_length =
                approximateTrajectoryLength(local_traj, std::max(8, 4 * integralRes));
        }
        return result;
    }

    class MetaTrajectoryPlanner
    {
    public:
        typedef GCOPTER_PolytopeSFC::SolverResult SolverResult;

        struct Result
        {
            SolverResult summary;
            Trajectory<5> trajectory;
        };

    public:
        inline Result solve(GCOPTER_PolytopeSFC &solver,
                            const Eigen::VectorXd &initial_x,
                            const GCOPTER_PolytopeSFC::IGOSolveOptions &options) const
        {
            Result result;
            result.summary = solver.solveMetaOptimizer(initial_x, options);
            if (result.summary.has_solution)
            {
                solver.evaluateObjectiveOnly(result.summary.best_x,
                                             nullptr,
                                             &result.trajectory);
            }
            return result;
        }
    };
}

#endif
