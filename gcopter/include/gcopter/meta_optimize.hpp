#ifndef GCOPTER_META_OPTIMIZE_HPP
#define GCOPTER_META_OPTIMIZE_HPP

#include "gcopter/gcopter.hpp"
#include "gcopter/minco_blackbox_optimizer.hpp"
#include "gcopter/trajectory.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

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
        const double thrust_mean = 0.5 * (thrust_min + thrust_max);
        const double thrust_radius =
            std::max(positiveEps(), 0.5 * std::abs(thrust_max - thrust_min));
        const double total_time = candidate_times.sum();

        auto waypoint_at =
            [this, &candidate_points, point_count](const int idx) -> Eigen::Vector3d
        {
            if (idx <= 0)
            {
                return headPVA.col(0);
            }
            if (idx >= point_count + 1)
            {
                return tailPVA.col(0);
            }
            return candidate_points.col(idx - 1);
        };

        double polyline_length = 0.0;
        double waypoint_smoothness = 0.0;
        for (int i = 0; i <= point_count; ++i)
        {
            polyline_length += (waypoint_at(i + 1) - waypoint_at(i)).norm();
        }
        for (int i = 1; i <= point_count; ++i)
        {
            const Eigen::Vector3d second_diff =
                waypoint_at(i + 1) - 2.0 * waypoint_at(i) + waypoint_at(i - 1);
            waypoint_smoothness += second_diff.squaredNorm();
        }
        const double length_scale = std::max(1.0e-3, polyline_length);
        const double normalized_waypoint_smoothness =
            waypoint_smoothness / std::max(positiveEps(), length_scale * length_scale);

        double trajectory_length = 0.0;
        double corridor_cost_integral = 0.0;
        double velocity_cost_integral = 0.0;
        double acceleration_cost_integral = 0.0;
        double body_rate_cost_integral = 0.0;
        double tilt_cost_integral = 0.0;
        double thrust_cost_integral = 0.0;
        double max_corridor_cost = 0.0;
        double max_velocity_cost = 0.0;
        double max_acceleration_cost = 0.0;
        double max_body_rate_cost = 0.0;
        double max_tilt_cost = 0.0;
        double max_thrust_cost = 0.0;
        Eigen::Vector3d previous_pos;
        bool has_previous_pos = false;

        const double corridor_margin = 0.10;
        const double soft_margin_ratio = 0.85;
        auto normalizedLimitCost =
            [soft_margin_ratio](const double usage) -> double
        {
            const double violation = std::max(0.0, usage - 1.0);
            const double margin =
                std::max(0.0, usage - soft_margin_ratio) /
                std::max(1.0e-6, 1.0 - soft_margin_ratio);
            return violation * violation + 0.05 * margin * margin;
        };

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
                double corridor_clearance =
                    std::numeric_limits<double>::infinity();
                if (piece_idx < hPolyIdx.size())
                {
                    const int h_idx = hPolyIdx(piece_idx);
                    if (h_idx >= 0 && h_idx < static_cast<int>(hPolytopes.size()))
                    {
                        const PolyhedronH &h_poly = hPolytopes[h_idx];
                        for (int row = 0; row < h_poly.rows(); ++row)
                        {
                            const double signed_distance =
                                h_poly.block<1, 3>(row, 0).dot(pos) +
                                h_poly(row, 3);
                            corridor_violation =
                                std::max(corridor_violation,
                                         std::max(0.0, signed_distance));
                            corridor_clearance =
                                std::min(corridor_clearance, -signed_distance);
                        }
                    }
                }
                if (!std::isfinite(corridor_clearance))
                {
                    corridor_clearance = corridor_margin;
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
                const double tilt_angle = std::acos(cos_tilt);
                const double tilt_violation =
                    std::max(0.0, tilt_angle - tilt_max);
                const double thrust_violation =
                    std::max(std::max(0.0, thrust_min - thrust),
                             std::max(0.0, thrust - thrust_max));

                const double corridor_margin_deficit =
                    std::max(0.0, corridor_margin - corridor_clearance) /
                    std::max(positiveEps(), corridor_margin);
                const double corridor_cost =
                    corridor_margin_deficit * corridor_margin_deficit;
                const double velocity_cost =
                    normalizedLimitCost(vel.norm() / std::max(positiveEps(), vel_max));
                const double acceleration_cost =
                    normalizedLimitCost(acc.norm() / std::max(positiveEps(), max_acceleration));
                const double body_rate_cost =
                    normalizedLimitCost(body_rate.norm() / std::max(positiveEps(), body_rate_max));
                const double tilt_cost =
                    normalizedLimitCost(tilt_angle / std::max(positiveEps(), tilt_max));
                const double thrust_cost =
                    normalizedLimitCost(std::abs(thrust - thrust_mean) / thrust_radius);

                corridor_cost_integral += node * step * corridor_cost;
                velocity_cost_integral += node * step * velocity_cost;
                acceleration_cost_integral += node * step * acceleration_cost;
                body_rate_cost_integral += node * step * body_rate_cost;
                tilt_cost_integral += node * step * tilt_cost;
                thrust_cost_integral += node * step * thrust_cost;

                max_corridor_cost = std::max(max_corridor_cost, corridor_cost);
                max_velocity_cost = std::max(max_velocity_cost, velocity_cost);
                max_acceleration_cost = std::max(max_acceleration_cost, acceleration_cost);
                max_body_rate_cost = std::max(max_body_rate_cost, body_rate_cost);
                max_tilt_cost = std::max(max_tilt_cost, tilt_cost);
                max_thrust_cost = std::max(max_thrust_cost, thrust_cost);

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

        const double inv_total_time =
            1.0 / std::max(sample_dt, std::max(positiveEps(), total_time));
        const double penalty_cost =
            std::max(0.0, options.meta_optimizer_collision_weight) * corridor_cost_integral * inv_total_time +
            std::max(0.0, options.meta_optimizer_velocity_weight) * velocity_cost_integral * inv_total_time +
            std::max(0.0, options.meta_optimizer_acceleration_weight) * acceleration_cost_integral * inv_total_time +
            std::max(0.0, options.meta_optimizer_body_rate_weight) * body_rate_cost_integral * inv_total_time +
            std::max(0.0, options.meta_optimizer_tilt_weight) * tilt_cost_integral * inv_total_time +
            std::max(0.0, options.meta_optimizer_thrust_weight) * thrust_cost_integral * inv_total_time;
        const double peak_penalty_cost =
            std::max(0.0, options.meta_optimizer_collision_weight) * max_corridor_cost +
            std::max(0.0, options.meta_optimizer_velocity_weight) * max_velocity_cost +
            std::max(0.0, options.meta_optimizer_acceleration_weight) * max_acceleration_cost +
            std::max(0.0, options.meta_optimizer_body_rate_weight) * max_body_rate_cost +
            std::max(0.0, options.meta_optimizer_tilt_weight) * max_tilt_cost +
            std::max(0.0, options.meta_optimizer_thrust_weight) * max_thrust_cost;
        const double normalized_energy =
            trajectory_energy * std::pow(std::max(positiveEps(), total_time), 5.0) /
            std::max(positiveEps(), length_scale * length_scale);
        const double fitness =
            std::max(0.0, options.meta_optimizer_time_weight) * total_time +
            std::max(0.0, options.meta_optimizer_length_weight) * trajectory_length +
            std::max(0.0, options.meta_optimizer_energy_weight) * normalized_energy +
            std::max(0.0, options.meta_optimizer_waypoint_smooth_weight) * normalized_waypoint_smoothness +
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
        const bool use_time_profile = options.meta_optimizer_use_time_profile;
        const bool use_nubs_direct = options.meta_optimizer_use_nubs_direct;
        const int requested_point_count =
            std::max(0, options.meta_optimizer_midpoints);
        const int point_count =
            use_nubs_direct
                ? std::max(requested_point_count, std::max(0, pieceN - 1))
                : requested_point_count;
        const int direct_piece_num = point_count + 1;
        const int time_param_offset = 3 * point_count;
        const int gamma_offset = time_param_offset + direct_piece_num;
        const int direct_dim =
            use_time_profile ? (gamma_offset + 1)
                             : (time_param_offset + direct_piece_num);
        const std::string direct_mode_name =
            use_time_profile ? "fixed P,beta,gamma"
                             : "fixed P,T";
        const std::string direct_family_name =
            use_nubs_direct ? "NUBS" : "MINCO";
        if (direct_piece_num <= 0 || pieceN <= 0 || direct_dim <= 0)
        {
            result.best_x =
                initialX.size() == getDecisionDim() ? initialX : getCommonInitialGuess();
            Trajectory<5> local_traj;
            result.objective =
                evaluateMetaPlannerFitness(result.best_x, options,
                                           &result.violations, &local_traj);
            if (std::isfinite(result.objective) && local_traj.getPieceNum() > 0)
            {
                result.has_solution = true;
                result.total_duration = local_traj.getTotalDuration();
                result.trajectory_length =
                    approximateTrajectoryLength(local_traj, std::max(8, 4 * integralRes));
            }
            result.status = "MetaPlanner-style " + direct_family_name +
                            " " + direct_mode_name +
                            " IGO optimizer (empty decision space)";
            return result;
        }

        const double eps = positiveEps();
        const double sample_dt =
            std::max(1.0e-3, options.meta_optimizer_sample_dt);
        const double time_lb =
            std::max(eps, options.meta_optimizer_time_lb);
        const double time_ub =
            std::max(time_lb + eps, options.meta_optimizer_time_ub);
        const double physical_vel_max = std::max(eps, magnitudeBd(0));
        const double configured_vel_max =
            std::max(eps, options.meta_optimizer_simple_max_velocity);
        const double simple_vel_max =
            std::min(physical_vel_max, configured_vel_max);
        const double simple_acc_max =
            std::max(eps, options.meta_optimizer_simple_max_acceleration);
        const double time_floor_scale =
            std::max(0.0, options.meta_time_floor_scale);
        const double target_velocity_ratio =
            std::max(0.20,
                     std::min(1.0, options.meta_optimizer_target_velocity_ratio));
        const double target_vel =
            std::max(eps, target_velocity_ratio * simple_vel_max);
        const double time_upper_scale =
            std::max(1.05,
                     1.0 + std::max(0.0, options.meta_total_slack_max_scale));
        const double slack_min_scale =
            std::max(0.0, options.meta_total_slack_min_scale);
        const double slack_max_scale =
            std::max(slack_min_scale, options.meta_total_slack_max_scale);
        const double beta_bound =
            std::max(0.1, options.meta_time_logit_bound);
        const double gamma_bound =
            std::max(0.1, options.meta_gamma_box_radius);
        const double local_box_radius =
            std::max(0.0, options.meta_optimizer_local_box_radius);

        Eigen::Vector3d workspace_min =
            headPVA.col(0).cwiseMin(tailPVA.col(0));
        Eigen::Vector3d workspace_max =
            headPVA.col(0).cwiseMax(tailPVA.col(0));
        const bool use_option_workspace =
            options.meta_optimizer_has_workspace_bounds &&
            options.meta_optimizer_workspace_min.allFinite() &&
            options.meta_optimizer_workspace_max.allFinite() &&
            (options.meta_optimizer_workspace_max.array() >
             options.meta_optimizer_workspace_min.array()).all();
        if (use_option_workspace)
        {
            workspace_min = options.meta_optimizer_workspace_min;
            workspace_max = options.meta_optimizer_workspace_max;
        }
        else
        {
            for (const PolyhedronV &v_poly : vPolytopes)
            {
                for (int col = 0; col < v_poly.cols(); ++col)
                {
                    Eigen::Vector3d vertex = v_poly.col(0);
                    if (col > 0)
                    {
                        vertex += v_poly.col(col);
                    }
                    workspace_min = workspace_min.cwiseMin(vertex);
                    workspace_max = workspace_max.cwiseMax(vertex);
                }
            }
            const double workspace_margin =
                std::max(2.0, 0.25 * (tailPVA.col(0) - headPVA.col(0)).norm());
            workspace_min.array() -= workspace_margin;
            workspace_max.array() += workspace_margin;
        }

        auto point_at =
            [this, point_count](const Eigen::Matrix3Xd &points,
                                const int idx) -> Eigen::Vector3d
        {
            if (idx <= 0)
            {
                return headPVA.col(0);
            }
            if (idx >= point_count + 1)
            {
                return tailPVA.col(0);
            }
            return points.col(idx - 1);
        };

        auto sigmoid =
            [](const double x) -> double
        {
            if (x >= 0.0)
            {
                const double z = std::exp(-x);
                return 1.0 / (1.0 + z);
            }
            const double z = std::exp(x);
            return z / (1.0 + z);
        };

        auto clamp_times_to_geometry =
            [direct_piece_num, time_lb, time_ub, simple_vel_max,
             target_vel, time_upper_scale, &point_at](
                const Eigen::Matrix3Xd &points,
                const Eigen::VectorXd &raw_times) -> Eigen::VectorXd
        {
            Eigen::VectorXd times(direct_piece_num);
            for (int i = 0; i < direct_piece_num; ++i)
            {
                const double segment_length =
                    (point_at(points, i + 1) - point_at(points, i)).norm();
                const double min_time =
                    std::max(time_lb, segment_length / simple_vel_max);
                const double nominal_time =
                    std::max(min_time, segment_length / target_vel);
                const double max_time =
                    std::max(min_time,
                             std::min(time_ub,
                                      std::max(min_time + 1.0e-6,
                                               nominal_time * time_upper_scale)));
                times(i) = std::max(min_time,
                                    std::min(max_time, raw_times(i)));
            }
            return times;
        };

        auto recover_profile_times =
            [direct_piece_num, time_lb, simple_vel_max, time_floor_scale,
             slack_min_scale, slack_max_scale, &point_at, &sigmoid](
                const Eigen::Matrix3Xd &points,
                const Eigen::VectorXd &beta,
                const double gamma) -> Eigen::VectorXd
        {
            Eigen::VectorXd floor_times(direct_piece_num);
            for (int i = 0; i < direct_piece_num; ++i)
            {
                const double segment_length =
                    (point_at(points, i + 1) - point_at(points, i)).norm();
                floor_times(i) =
                    std::max(time_lb,
                             time_floor_scale * segment_length / simple_vel_max);
            }

            const double floor_sum =
                std::max(GCOPTER_PolytopeSFC::positiveEps(), floor_times.sum());
            const double slack_min = slack_min_scale * floor_sum;
            const double slack_max = std::max(slack_min, slack_max_scale * floor_sum);
            const double slack_total =
                slack_min + sigmoid(gamma) * (slack_max - slack_min);
            const Eigen::VectorXd ratio = GCOPTER_PolytopeSFC::softmax(beta);
            Eigen::VectorXd times = floor_times + ratio * slack_total;
            return times;
        };

        auto unpack_direct =
            [point_count, time_param_offset, gamma_offset, direct_piece_num,
             direct_dim, use_time_profile, &recover_profile_times,
             &clamp_times_to_geometry](
                const Eigen::VectorXd &y,
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
            if (use_time_profile)
            {
                const Eigen::VectorXd beta =
                    y.segment(time_param_offset, direct_piece_num);
                candidate_times =
                    recover_profile_times(candidate_points, beta, y(gamma_offset));
            }
            else
            {
                candidate_times =
                    clamp_times_to_geometry(
                        candidate_points,
                        y.segment(time_param_offset, direct_piece_num));
            }
            return candidate_times.allFinite() &&
                   (candidate_times.array() > 0.0).all();
        };

        struct CorridorClearance
        {
            double violation = 0.0;
            double clearance = std::numeric_limits<double>::infinity();
        };

        auto corridor_clearance =
            [this](const Eigen::Vector3d &position) -> CorridorClearance
        {
            if (hPolytopes.empty())
            {
                return CorridorClearance{};
            }

            double best_violation = std::numeric_limits<double>::infinity();
            double best_clearance = -std::numeric_limits<double>::infinity();
            double best_outside_clearance = -std::numeric_limits<double>::infinity();
            bool has_inside_poly = false;

            for (const PolyhedronH &h_poly : hPolytopes)
            {
                double poly_violation = -std::numeric_limits<double>::infinity();
                double poly_clearance = std::numeric_limits<double>::infinity();
                for (int row = 0; row < h_poly.rows(); ++row)
                {
                    const double normal_norm =
                        std::max(positiveEps(), h_poly.block<1, 3>(row, 0).norm());
                    const double signed_distance =
                        (h_poly.block<1, 3>(row, 0).dot(position) +
                         h_poly(row, 3)) /
                        normal_norm;
                    poly_violation =
                        std::max(poly_violation, signed_distance);
                    poly_clearance =
                        std::min(poly_clearance, -signed_distance);
                }

                if (poly_violation <= 0.0)
                {
                    has_inside_poly = true;
                    best_clearance = std::max(best_clearance, poly_clearance);
                }
                else if (poly_violation < best_violation)
                {
                    best_violation = poly_violation;
                    best_outside_clearance = poly_clearance;
                }
            }

            CorridorClearance result;
            if (has_inside_poly)
            {
                result.violation = 0.0;
                result.clearance = std::max(0.0, best_clearance);
                return result;
            }

            result.violation = std::max(0.0, best_violation);
            result.clearance =
                std::isfinite(best_outside_clearance)
                    ? best_outside_clearance
                    : -result.violation;
            return result;
        };

        auto build_direct_trajectory =
            [this, direct_piece_num](const Eigen::Matrix3Xd &candidate_points,
                                     const Eigen::VectorXd &candidate_times,
                                     minco::MINCO_S3NU &direct_minco,
                                     Trajectory<5> &trajectory,
                                     double &energy) -> bool
        {
            if (candidate_points.cols() != direct_piece_num - 1 ||
                candidate_times.size() != direct_piece_num ||
                !candidate_points.allFinite() ||
                !candidate_times.allFinite() ||
                (candidate_times.array() <= positiveEps()).any())
            {
                return false;
            }

            direct_minco.setConditions(headPVA, tailPVA, direct_piece_num);
            direct_minco.setParameters(candidate_points, candidate_times);
            direct_minco.getEnergy(energy);
            direct_minco.getTrajectory(trajectory);
            return std::isfinite(energy) && trajectory.getPieceNum() > 0;
        };

        auto build_direct_nubs =
            [this, direct_piece_num](const Eigen::Matrix3Xd &candidate_points,
                                     const Eigen::VectorXd &candidate_times,
                                     bsplinetrajectory::NUBSTrajectory<3> &trajectory,
                                     double &energy) -> bool
        {
            if (candidate_points.cols() != direct_piece_num - 1 ||
                candidate_times.size() != direct_piece_num ||
                !candidate_points.allFinite() ||
                !candidate_times.allFinite() ||
                (candidate_times.array() <= positiveEps()).any())
            {
                return false;
            }

            Eigen::MatrixXd control_points;
            const Eigen::MatrixXd inner_points_rows =
                candidate_points.cols() > 0 ? candidate_points.transpose()
                                            : Eigen::MatrixXd(0, 3);
            try
            {
                trajectory.generate(inner_points_rows, headPVA, tailPVA,
                                    candidate_times, control_points);
                energy = trajectory.getEnergy();
            }
            catch (const std::exception &)
            {
                energy = std::numeric_limits<double>::infinity();
                return false;
            }
            return std::isfinite(energy) && trajectory.getPieceNum() > 0;
        };

        auto evaluate_simple_sampled_candidate =
            [this, &options, &corridor_clearance,
             direct_piece_num, sample_dt, simple_vel_max, simple_acc_max, eps](
                const Eigen::Matrix3Xd &candidate_points,
                const Eigen::VectorXd &candidate_times,
                const double candidate_energy,
                const int trajectory_piece_num,
                const auto &position_at,
                const auto &velocity_at,
                const auto &acceleration_at,
                TrajectoryViolationMetrics *metrics) -> double
        {
            if (candidate_points.cols() != direct_piece_num - 1 ||
                candidate_times.size() != direct_piece_num ||
                !candidate_points.allFinite() ||
                !candidate_times.allFinite() ||
                (candidate_times.array() <= eps).any() ||
                !std::isfinite(candidate_energy) ||
                trajectory_piece_num <= 0)
            {
                if (metrics)
                {
                    *metrics = TrajectoryViolationMetrics();
                }
                return std::numeric_limits<double>::infinity();
            }

            TrajectoryViolationMetrics local_metrics;
            const double total_time = candidate_times.sum();
            const int sample_count =
                std::max(1, static_cast<int>(std::ceil(total_time / sample_dt)));
            const double step = total_time / static_cast<double>(sample_count);

            const double clearance_margin = 0.25;
            const bool use_collision_length_cost =
                options.meta_optimizer_use_collision_length_cost &&
                static_cast<bool>(options.meta_optimizer_collision_checker);
            double collision_length = 0.0;
            double clearance_margin_integral = 0.0;
            double outside_violation_integral = 0.0;
            double peak_corridor_violation = 0.0;
            double occupancy_integral = 0.0;
            double occupancy_peak = 0.0;
            double velocity_exceed_integral = 0.0;
            double acceleration_exceed_integral = 0.0;
            Eigen::Vector3d previous_position = position_at(0.0);
            bool previous_occupied = false;

            if (!previous_position.allFinite())
            {
                if (metrics)
                {
                    *metrics = TrajectoryViolationMetrics();
                }
                return std::numeric_limits<double>::infinity();
            }

            for (int i = 0; i <= sample_count; ++i)
            {
                const double t =
                    std::min(total_time, step * static_cast<double>(i));
                const double node =
                    (i == 0 || i == sample_count) ? 0.5 : 1.0;
                const Eigen::Vector3d position = position_at(t);
                const Eigen::Vector3d velocity = velocity_at(t);
                const Eigen::Vector3d acceleration = acceleration_at(t);

                if (!position.allFinite() ||
                    !velocity.allFinite() ||
                    !acceleration.allFinite())
                {
                    if (metrics)
                    {
                        *metrics = TrajectoryViolationMetrics();
                    }
                    return std::numeric_limits<double>::infinity();
                }

                const double segment_length =
                    i > 0 ? (position - previous_position).norm() : 0.0;
                const bool occupied =
                    options.meta_optimizer_collision_checker &&
                    options.meta_optimizer_collision_checker(position);

                if (i > 0 && previous_occupied)
                {
                    collision_length += segment_length;
                }

                if (!use_collision_length_cost)
                {
                    const CorridorClearance clearance =
                        corridor_clearance(position);
                    const double clearance_deficit =
                        std::max(0.0, clearance_margin - clearance.clearance) /
                        std::max(eps, clearance_margin);
                    clearance_margin_integral +=
                        node * step * clearance_deficit * clearance_deficit;
                    outside_violation_integral +=
                        node * step * clearance.violation * clearance.violation;
                    peak_corridor_violation =
                        std::max(peak_corridor_violation, clearance.violation);
                    local_metrics.max_corridor_violation =
                        std::max(local_metrics.max_corridor_violation,
                                 clearance.violation);
                }

                if (occupied)
                {
                    occupancy_integral += node * step;
                    occupancy_peak = 1.0;
                }

                const double velocity_violation =
                    std::max(0.0, velocity.norm() - simple_vel_max);
                const double acceleration_violation =
                    std::max(0.0, acceleration.norm() - simple_acc_max);

                velocity_exceed_integral += node * step * velocity_violation;
                acceleration_exceed_integral += node * step * acceleration_violation;

                local_metrics.max_velocity_violation =
                    std::max(local_metrics.max_velocity_violation,
                             velocity_violation);
                local_metrics.max_acceleration_violation =
                    std::max(local_metrics.max_acceleration_violation,
                             acceleration_violation);
                ++local_metrics.sample_count;
                previous_occupied = occupied;
                previous_position = position;
            }

            const double inv_total_time =
                1.0 / std::max(sample_dt, std::max(eps, total_time));
            const double clearance_collision_cost =
                clearance_margin_integral * inv_total_time +
                outside_violation_integral * inv_total_time +
                peak_corridor_violation * peak_corridor_violation +
                10.0 * occupancy_integral * inv_total_time +
                10.0 * occupancy_peak;
            const double collision_cost =
                use_collision_length_cost ? collision_length
                                          : clearance_collision_cost;
            if (use_collision_length_cost)
            {
                local_metrics.max_corridor_violation =
                    std::max(local_metrics.max_corridor_violation,
                             collision_length);
            }
            const double fitness =
                std::max(0.0, options.meta_optimizer_time_weight) * total_time +
                std::max(0.0, options.meta_optimizer_collision_weight) * collision_cost +
                std::max(0.0, options.meta_optimizer_velocity_weight) *
                    velocity_exceed_integral / sample_dt +
                std::max(0.0, options.meta_optimizer_acceleration_weight) *
                    acceleration_exceed_integral / sample_dt;

            local_metrics.penalty_cost =
                std::max(0.0, options.meta_optimizer_collision_weight) * collision_cost +
                std::max(0.0, options.meta_optimizer_velocity_weight) *
                    velocity_exceed_integral / sample_dt +
                std::max(0.0, options.meta_optimizer_acceleration_weight) *
                    acceleration_exceed_integral / sample_dt;

            if (metrics)
            {
                *metrics = local_metrics;
            }
            return fitness;
        };

        auto evaluate_simple_candidate =
            [&evaluate_simple_sampled_candidate](
                const MINCOBlackboxOptimizerS3::Candidate &candidate,
                TrajectoryViolationMetrics *metrics,
                Trajectory<5> *traj) -> double
        {
            const Eigen::VectorXd &candidate_times = candidate.times;
            const Trajectory<5> &direct_traj = candidate.trajectory;
            const double fitness =
                evaluate_simple_sampled_candidate(
                    candidate.points,
                    candidate_times,
                    candidate.energy,
                    direct_traj.getPieceNum(),
                    [&direct_traj](const double t) -> Eigen::Vector3d
                    {
                        return direct_traj.getPos(t);
                    },
                    [&direct_traj](const double t) -> Eigen::Vector3d
                    {
                        return direct_traj.getVel(t);
                    },
                    [&direct_traj](const double t) -> Eigen::Vector3d
                    {
                        return direct_traj.getAcc(t);
                    },
                    metrics);
            if (traj && std::isfinite(fitness))
            {
                *traj = direct_traj;
            }
            return fitness;
        };

        struct DirectNUBSCandidate
        {
            Eigen::VectorXd y;
            Eigen::Matrix3Xd points;
            Eigen::VectorXd times;
            bsplinetrajectory::NUBSTrajectory<3> trajectory;
            double energy = std::numeric_limits<double>::infinity();
        };

        auto evaluate_simple_nubs_candidate =
            [&evaluate_simple_sampled_candidate](
                const DirectNUBSCandidate &candidate,
                TrajectoryViolationMetrics *metrics) -> double
        {
            const bsplinetrajectory::NUBSTrajectory<3> &direct_traj =
                candidate.trajectory;
            return evaluate_simple_sampled_candidate(
                candidate.points,
                candidate.times,
                candidate.energy,
                direct_traj.getPieceNum(),
                [&direct_traj](const double t) -> Eigen::Vector3d
                {
                    return direct_traj.evaluate(t, 0);
                },
                [&direct_traj](const double t) -> Eigen::Vector3d
                {
                    return direct_traj.evaluate(t, 1);
                },
                [&direct_traj](const double t) -> Eigen::Vector3d
                {
                    return direct_traj.evaluate(t, 2);
                },
                metrics);
        };

        auto evaluate_control_point_nubs_candidate =
            [this, &options, direct_piece_num,
             simple_vel_max, simple_acc_max, eps](
                const DirectNUBSCandidate &candidate,
                TrajectoryViolationMetrics *metrics) -> double
        {
            typedef Eigen::Matrix<double, Eigen::Dynamic, 3> MatrixX3d;

            if (candidate.points.cols() != direct_piece_num - 1 ||
                candidate.times.size() != direct_piece_num ||
                !candidate.points.allFinite() ||
                !candidate.times.allFinite() ||
                (candidate.times.array() <= eps).any() ||
                !std::isfinite(candidate.energy) ||
                candidate.trajectory.getPieceNum() <= 0)
            {
                if (metrics)
                {
                    *metrics = TrajectoryViolationMetrics();
                }
                return std::numeric_limits<double>::infinity();
            }

            const MatrixX3d &control_points =
                candidate.trajectory.getControlPoints();
            const Eigen::VectorXd &knots = candidate.trajectory.getKnots();
            const int degree = candidate.trajectory.getP();
            if (control_points.rows() <= 0 ||
                knots.size() <= degree + 1 ||
                !control_points.allFinite() ||
                !knots.allFinite())
            {
                if (metrics)
                {
                    *metrics = TrajectoryViolationMetrics();
                }
                return std::numeric_limits<double>::infinity();
            }

            TrajectoryViolationMetrics local_metrics;
            const double total_time = candidate.times.sum();
            if (!std::isfinite(total_time) || total_time <= eps)
            {
                if (metrics)
                {
                    *metrics = TrajectoryViolationMetrics();
                }
                return std::numeric_limits<double>::infinity();
            }

            auto waypoint_at =
                [this, &candidate, direct_piece_num](const int idx) -> Eigen::Vector3d
            {
                if (idx <= 0)
                {
                    return headPVA.col(0);
                }
                if (idx >= direct_piece_num)
                {
                    return tailPVA.col(0);
                }
                return candidate.points.col(idx - 1);
            };

            double polyline_length = 0.0;
            for (int i = 0; i < direct_piece_num; ++i)
            {
                polyline_length += (waypoint_at(i + 1) - waypoint_at(i)).norm();
            }
            const double length_scale = std::max(eps, polyline_length);
            const double normalized_energy =
                candidate.energy * std::pow(std::max(eps, total_time), 5.0) /
                std::max(eps, length_scale * length_scale);

            double collision_accum = 0.0;
            double collision_peak = 0.0;
            int collision_span_count = 0;
            if (!hPolytopes.empty())
            {
                for (int span = degree; span < knots.size() - degree - 1; ++span)
                {
                    if (knots(span + 1) - knots(span) <= eps)
                    {
                        continue;
                    }

                    const int first_ctrl = std::max(0, span - degree);
                    const int last_ctrl =
                        std::min<int>(control_points.rows() - 1, span);
                    if (first_ctrl > last_ctrl)
                    {
                        continue;
                    }

                    std::vector<int> nearby_poly_indices;
                    auto add_poly_index =
                        [&nearby_poly_indices](const int poly_idx) -> void
                    {
                        if (poly_idx < 0)
                        {
                            return;
                        }
                        if (std::find(nearby_poly_indices.begin(),
                                      nearby_poly_indices.end(),
                                      poly_idx) == nearby_poly_indices.end())
                        {
                            nearby_poly_indices.push_back(poly_idx);
                        }
                    };
                    const int piece_idx = span - degree;
                    if (hPolyIdx.size() == pieceN &&
                        direct_piece_num == pieceN)
                    {
                        for (int local_piece = piece_idx - 1;
                             local_piece <= piece_idx + 1;
                             ++local_piece)
                        {
                            if (0 <= local_piece && local_piece < hPolyIdx.size())
                            {
                                add_poly_index(hPolyIdx(local_piece));
                            }
                        }
                    }
                    else
                    {
                        const double ratio =
                            (static_cast<double>(piece_idx) + 0.5) /
                            static_cast<double>(std::max(1, direct_piece_num));
                        const int center_poly =
                            static_cast<int>(std::floor(
                                ratio * static_cast<double>(hPolytopes.size())));
                        for (int poly_idx = center_poly - 1;
                             poly_idx <= center_poly + 1;
                             ++poly_idx)
                        {
                            add_poly_index(poly_idx);
                        }
                    }

                    double best_poly_cost =
                        std::numeric_limits<double>::infinity();
                    double best_poly_peak = 0.0;
                    int best_term_count = 1;
                    for (const int poly_idx : nearby_poly_indices)
                    {
                        if (poly_idx < 0 ||
                            poly_idx >= static_cast<int>(hPolytopes.size()))
                        {
                            continue;
                        }
                        const PolyhedronH &h_poly = hPolytopes[poly_idx];
                        double poly_cost = 0.0;
                        double poly_peak = 0.0;
                        int term_count = 0;
                        for (int ctrl_idx = first_ctrl;
                             ctrl_idx <= last_ctrl;
                             ++ctrl_idx)
                        {
                            const Eigen::Vector3d cp =
                                control_points.row(ctrl_idx).transpose();
                            for (int row = 0; row < h_poly.rows(); ++row)
                            {
                                const double normal_norm =
                                    std::max(eps,
                                             h_poly.block<1, 3>(row, 0).norm());
                                const double signed_distance =
                                    (h_poly.block<1, 3>(row, 0).dot(cp) +
                                     h_poly(row, 3)) /
                                    normal_norm;
                                const double violation =
                                    std::max(0.0, signed_distance);
                                poly_cost += violation * violation;
                                poly_peak = std::max(poly_peak, violation);
                                ++term_count;
                            }
                        }

                        if (poly_cost < best_poly_cost)
                        {
                            best_poly_cost = poly_cost;
                            best_poly_peak = poly_peak;
                            best_term_count = std::max(1, term_count);
                        }
                    }

                    if (std::isfinite(best_poly_cost))
                    {
                        collision_accum +=
                            best_poly_cost /
                            static_cast<double>(std::max(1, best_term_count));
                        collision_peak = std::max(collision_peak, best_poly_peak);
                        local_metrics.max_corridor_violation =
                            std::max(local_metrics.max_corridor_violation,
                                     best_poly_peak);
                        ++collision_span_count;
                    }
                }
            }
            else if (options.meta_optimizer_collision_checker)
            {
                for (int i = 0; i < control_points.rows(); ++i)
                {
                    const Eigen::Vector3d cp = control_points.row(i).transpose();
                    if (options.meta_optimizer_collision_checker(cp))
                    {
                        collision_accum += 1.0;
                        collision_peak = 1.0;
                    }
                    ++collision_span_count;
                }
                local_metrics.max_corridor_violation = collision_peak;
            }

            const double collision_cost =
                collision_span_count > 0
                    ? collision_accum /
                              static_cast<double>(collision_span_count) +
                          collision_peak * collision_peak
                    : 0.0;

            auto derivative_control_points =
                [eps](const bsplinetrajectory::NUBSTrajectory<3> &trajectory,
                      const int derivative_order,
                      MatrixX3d &derivative_points) -> bool
            {
                derivative_points = trajectory.getControlPoints();
                const Eigen::VectorXd &local_knots = trajectory.getKnots();
                const int local_degree = trajectory.getP();
                if (derivative_order <= 0)
                {
                    return derivative_points.allFinite();
                }
                if (derivative_order > local_degree)
                {
                    return false;
                }

                for (int derivative = 1;
                     derivative <= derivative_order;
                     ++derivative)
                {
                    const int rows = derivative_points.rows();
                    if (rows <= 1)
                    {
                        return false;
                    }

                    MatrixX3d next(rows - 1, 3);
                    const double scale =
                        static_cast<double>(local_degree - derivative + 1);
                    for (int i = 0; i < rows - 1; ++i)
                    {
                        const int knot_lo = i + derivative;
                        const int knot_hi = i + local_degree + 1;
                        if (knot_lo < 0 ||
                            knot_hi >= local_knots.size())
                        {
                            return false;
                        }
                        const double denominator =
                            local_knots(knot_hi) - local_knots(knot_lo);
                        if (!std::isfinite(denominator) ||
                            denominator <= eps)
                        {
                            return false;
                        }
                        next.row(i) =
                            scale *
                            (derivative_points.row(i + 1) -
                             derivative_points.row(i)) /
                            denominator;
                    }
                    derivative_points.swap(next);
                }

                return derivative_points.allFinite();
            };

            auto derivative_limit_cost =
                [eps](const MatrixX3d &derivative_points,
                      const double limit,
                      double &peak_violation) -> double
            {
                peak_violation = 0.0;
                if (derivative_points.rows() <= 0)
                {
                    return 0.0;
                }

                double violation_sum = 0.0;
                for (int i = 0; i < derivative_points.rows(); ++i)
                {
                    const double violation =
                        std::max(0.0,
                                 derivative_points.row(i).norm() -
                                     std::max(eps, limit));
                    violation_sum += violation * violation;
                    peak_violation = std::max(peak_violation, violation);
                }
                return violation_sum /
                           static_cast<double>(derivative_points.rows()) +
                       peak_violation * peak_violation;
            };

            MatrixX3d velocity_points;
            MatrixX3d acceleration_points;
            MatrixX3d jerk_points;
            if (!derivative_control_points(candidate.trajectory, 1,
                                           velocity_points) ||
                !derivative_control_points(candidate.trajectory, 2,
                                           acceleration_points))
            {
                if (metrics)
                {
                    *metrics = TrajectoryViolationMetrics();
                }
                return std::numeric_limits<double>::infinity();
            }

            const double velocity_cost =
                derivative_limit_cost(velocity_points, simple_vel_max,
                                      local_metrics.max_velocity_violation);
            const double acceleration_cost =
                derivative_limit_cost(acceleration_points, simple_acc_max,
                                      local_metrics.max_acceleration_violation);

            double jerk_cost = 0.0;
            const double jerk_weight =
                std::max(0.0, options.meta_optimizer_jerk_weight);
            if (jerk_weight > 0.0)
            {
                if (!derivative_control_points(candidate.trajectory, 3,
                                               jerk_points))
                {
                    if (metrics)
                    {
                        *metrics = TrajectoryViolationMetrics();
                    }
                    return std::numeric_limits<double>::infinity();
                }
                jerk_cost =
                    derivative_limit_cost(
                        jerk_points,
                        std::max(eps, options.meta_optimizer_simple_max_jerk),
                        local_metrics.max_jerk_violation);
            }

            const double time_cost =
                std::max(0.0, options.meta_optimizer_time_weight) *
                total_time;
            const double energy_cost =
                std::max(0.0, options.meta_optimizer_energy_weight) *
                normalized_energy;
            const double collision_penalty =
                std::max(0.0, options.meta_optimizer_collision_weight) *
                collision_cost;
            const double velocity_penalty =
                std::max(0.0, options.meta_optimizer_velocity_weight) *
                velocity_cost;
            const double acceleration_penalty =
                std::max(0.0, options.meta_optimizer_acceleration_weight) *
                acceleration_cost;
            const double jerk_penalty = jerk_weight * jerk_cost;

            local_metrics.penalty_cost =
                collision_penalty + velocity_penalty +
                acceleration_penalty + jerk_penalty;
            local_metrics.sample_count =
                control_points.rows() + velocity_points.rows() +
                acceleration_points.rows() + jerk_points.rows();

            if (metrics)
            {
                *metrics = local_metrics;
            }

            return time_cost + energy_cost + local_metrics.penalty_cost;
        };

        auto direct_to_x =
            [this, use_nubs_direct, &unpack_direct,
             &build_direct_trajectory, &build_direct_nubs](
                const Eigen::VectorXd &y,
                Eigen::VectorXd &x) -> bool
        {
            Eigen::Matrix3Xd candidate_points;
            Eigen::VectorXd candidate_times;
            if (!unpack_direct(y, candidate_points, candidate_times))
            {
                return false;
            }

            double trajectory_energy = 0.0;
            Trajectory<5> direct_traj;
            bsplinetrajectory::NUBSTrajectory<3> direct_nubs;
            if (use_nubs_direct)
            {
                if (!build_direct_nubs(candidate_points, candidate_times,
                                       direct_nubs, trajectory_energy))
                {
                    return false;
                }
            }
            else
            {
                minco::MINCO_S3NU direct_minco;
                if (!build_direct_trajectory(candidate_points, candidate_times,
                                             direct_minco, direct_traj,
                                             trajectory_energy))
                {
                    return false;
                }
            }

            const double total_time =
                std::max(positiveEps(), candidate_times.sum());
            Eigen::VectorXd lifted_times =
                Eigen::VectorXd::Constant(pieceN, total_time / static_cast<double>(pieceN));
            Eigen::Matrix3Xd lifted_points(3, std::max(0, pieceN - 1));
            for (int i = 0; i < pieceN - 1; ++i)
            {
                const double t =
                    total_time * static_cast<double>(i + 1) /
                    static_cast<double>(pieceN);
                lifted_points.col(i) =
                    use_nubs_direct ? direct_nubs.evaluate(t, 0)
                                    : direct_traj.getPos(t);
            }

            x.resize(getDecisionDim());
            Eigen::Map<Eigen::VectorXd> tau(x.data(), temporalDim);
            Eigen::Map<Eigen::VectorXd> xi(x.data() + temporalDim, spatialDim);
            backwardT(lifted_times, tau);
            backwardP(lifted_points, vPolyIdx, vPolytopes, xi);
            return x.allFinite();
        };

        const Eigen::VectorXd fallback_x =
            initialX.size() == getDecisionDim() && initialX.allFinite()
                ? initialX
                : getCommonInitialGuess();

        auto build_anchor_points =
            [this, point_count, &workspace_min, &workspace_max](
                const Eigen::VectorXd &x,
                Eigen::Matrix3Xd &anchors) -> bool
        {
            if (x.size() != getDecisionDim() || point_count <= 0)
            {
                return false;
            }

            minco::MINCO_S3NU seed_minco;
            if (!buildJerkOpt(x, seed_minco))
            {
                return false;
            }

            Trajectory<5> seed_traj;
            seed_minco.getTrajectory(seed_traj);
            const double total_time = seed_traj.getTotalDuration();
            if (seed_traj.getPieceNum() <= 0 ||
                !std::isfinite(total_time) ||
                total_time <= 0.0)
            {
                return false;
            }

            anchors.resize(3, point_count);
            for (int i = 0; i < point_count; ++i)
            {
                const double ratio =
                    static_cast<double>(i + 1) /
                    static_cast<double>(point_count + 1);
                anchors.col(i) =
                    seed_traj.getPos(ratio * total_time)
                        .cwiseMax(workspace_min)
                        .cwiseMin(workspace_max);
            }
            return anchors.allFinite();
        };

        Eigen::VectorXd y0 = Eigen::VectorXd::Zero(direct_dim);
        Eigen::Matrix3Xd anchor_points(3, point_count);
        const Eigen::Vector3d start = headPVA.col(0);
        const Eigen::Vector3d goal = tailPVA.col(0);
        if (!build_anchor_points(fallback_x, anchor_points))
        {
            for (int i = 0; i < point_count; ++i)
            {
                const double ratio =
                    static_cast<double>(i + 1) /
                    static_cast<double>(point_count + 1);
                anchor_points.col(i) = (1.0 - ratio) * start + ratio * goal;
            }
        }
        for (int i = 0; i < point_count; ++i)
        {
            y0.segment<3>(3 * i) =
                anchor_points.col(i)
                    .cwiseMax(workspace_min)
                    .cwiseMin(workspace_max);
        }
        if (use_time_profile)
        {
            y0.segment(time_param_offset, direct_piece_num).setZero();
            y0(gamma_offset) = 0.0;
        }
        else
        {
            Eigen::VectorXd y0_times(direct_piece_num);
            const double initial_time_scale =
                std::max(1.0, options.meta_initial_time_scale);
            for (int i = 0; i < direct_piece_num; ++i)
            {
                const double segment_length =
                    (point_at(anchor_points, i + 1) -
                     point_at(anchor_points, i))
                        .norm();
                y0_times(i) =
                    std::max(time_lb,
                             initial_time_scale * segment_length / target_vel);
            }
            y0.segment(time_param_offset, direct_piece_num) =
                clamp_times_to_geometry(anchor_points, y0_times);
        }

        Eigen::VectorXd lower(direct_dim);
        Eigen::VectorXd upper(direct_dim);
        for (int i = 0; i < point_count; ++i)
        {
            const Eigen::Vector3d anchor =
                anchor_points.col(i)
                    .cwiseMax(workspace_min)
                    .cwiseMin(workspace_max);
            Eigen::Vector3d local_lower =
                (anchor.array() - local_box_radius).matrix().cwiseMax(workspace_min);
            Eigen::Vector3d local_upper =
                (anchor.array() + local_box_radius).matrix().cwiseMin(workspace_max);
            local_lower = local_lower.cwiseMin(anchor);
            local_upper = local_upper.cwiseMax(anchor);
            lower.segment<3>(3 * i) = local_lower;
            upper.segment<3>(3 * i) = local_upper;
        }
        if (use_time_profile)
        {
            lower.segment(time_param_offset, direct_piece_num).setConstant(-beta_bound);
            upper.segment(time_param_offset, direct_piece_num).setConstant(beta_bound);
            lower(gamma_offset) = -gamma_bound;
            upper(gamma_offset) = gamma_bound;
        }
        else
        {
            lower.segment(time_param_offset, direct_piece_num).setConstant(time_lb);
            upper.segment(time_param_offset, direct_piece_num).setConstant(time_ub);
        }
        y0 = y0.cwiseMax(lower).cwiseMin(upper);

        MINCOBlackboxOptimizerS3 direct_optimizer;
        MINCOBlackboxOptimizerS3::Options direct_optimizer_options;
        direct_optimizer_options.time_mode =
            use_time_profile ? MINCOBlackboxOptimizerS3::TimeMode::Profile
                             : MINCOBlackboxOptimizerS3::TimeMode::Direct;
        direct_optimizer_options.time_lb = time_lb;
        direct_optimizer_options.time_ub = time_ub;
        direct_optimizer_options.simple_vel_max = simple_vel_max;
        direct_optimizer_options.target_vel = target_vel;
        direct_optimizer_options.time_floor_scale = time_floor_scale;
        direct_optimizer_options.initial_time_scale =
            std::max(1.0, options.meta_initial_time_scale);
        direct_optimizer_options.total_slack_min_scale = slack_min_scale;
        direct_optimizer_options.total_slack_max_scale = slack_max_scale;
        direct_optimizer_options.beta_bound = beta_bound;
        direct_optimizer_options.gamma_bound = gamma_bound;
        direct_optimizer_options.local_box_radius = local_box_radius;
        direct_optimizer.setWarmStartDecision(y0);
        const bool direct_optimizer_ready =
            !use_nubs_direct &&
            direct_optimizer.setup(headPVA, tailPVA,
                                   anchor_points,
                                   workspace_min, workspace_max,
                                   direct_optimizer_options);

        IGO::Options igoOptions;
        igoOptions.population = options.population;
        igoOptions.max_iterations = options.max_iterations;
        igoOptions.min_sigma = options.min_sigma;
        igoOptions.eta_mean = options.eta_mean;
        igoOptions.eta_sigma = options.eta_sigma;
        igoOptions.elite_ratio = options.elite_ratio;
        igoOptions.seed = options.seed;
        igoOptions.stall_iterations = 5;
        igoOptions.abs_cost_tol = 1.0e-3;
        igoOptions.rel_cost_tol = 0.0;
        igoOptions.sigma_convergence_ratio = 1.0e12;
        igoOptions.initial_sigma_scale = 0.25;

        IGO::Budget budget;
        budget.max_evaluations =
            options.max_evaluations > 0
                ? options.max_evaluations
                : std::max(300, 10 * std::max(2, options.population));
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
                          const bool lhs_finite = std::isfinite(lhs.cost);
                          const bool rhs_finite = std::isfinite(rhs.cost);
                          if (lhs_finite != rhs_finite)
                          {
                              return lhs_finite;
                          }
                          if (lhs.feasible != rhs.feasible)
                          {
                              return lhs.feasible;
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

        auto evaluate_built_direct_candidate =
            [&evaluate_simple_candidate, &options](
                const MINCOBlackboxOptimizerS3::Candidate &candidate,
                DirectCandidate &evaluated) -> bool
        {
            TrajectoryViolationMetrics local_metrics;
            evaluated.y = candidate.y;
            evaluated.cost =
                evaluate_simple_candidate(candidate, &local_metrics, nullptr);
            evaluated.max_violation = local_metrics.maxViolation();
            evaluated.feasible =
                std::isfinite(evaluated.cost) &&
                evaluated.max_violation <= std::max(0.0, options.feasibility_tol);
            return std::isfinite(evaluated.cost);
        };

        auto evaluate_direct_candidate =
            [use_nubs_direct, &unpack_direct,
             &build_direct_trajectory, &build_direct_nubs,
             &evaluate_built_direct_candidate,
             &evaluate_simple_nubs_candidate,
             &evaluate_control_point_nubs_candidate, &options](
                const Eigen::VectorXd &candidate_y,
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

            if (use_nubs_direct)
            {
                DirectNUBSCandidate candidate;
                candidate.y = candidate_y;
                candidate.points = candidate_points;
                candidate.times = candidate_times;
                if (!build_direct_nubs(candidate.points, candidate.times,
                                       candidate.trajectory,
                                       candidate.energy))
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
                    options.meta_optimizer_use_control_point_objective
                        ? evaluate_control_point_nubs_candidate(candidate,
                                                                &local_metrics)
                        : evaluate_simple_nubs_candidate(candidate,
                                                         &local_metrics);
                evaluated.max_violation = local_metrics.maxViolation();
                evaluated.feasible =
                    std::isfinite(evaluated.cost) &&
                    evaluated.max_violation <= std::max(0.0, options.feasibility_tol);
                return std::isfinite(evaluated.cost);
            }

            minco::MINCO_S3NU direct_minco;
            MINCOBlackboxOptimizerS3::Candidate candidate;
            candidate.y = candidate_y;
            candidate.points = candidate_points;
            candidate.times = candidate_times;
            if (!build_direct_trajectory(candidate.points, candidate.times,
                                         direct_minco, candidate.trajectory,
                                         candidate.energy))
            {
                evaluated.y = candidate_y;
                evaluated.cost = std::numeric_limits<double>::infinity();
                evaluated.max_violation = std::numeric_limits<double>::infinity();
                evaluated.feasible = false;
                return false;
            }

            return evaluate_built_direct_candidate(candidate, evaluated);
        };

        const std::chrono::steady_clock::time_point start_time =
            std::chrono::steady_clock::now();
        MINCOBlackboxOptimizerS3::Result blackbox_result;
        IGO::Result igoResult;
        if (direct_optimizer_ready)
        {
            blackbox_result = direct_optimizer.optimize(
                [&insert_meta_candidate, &evaluate_built_direct_candidate](
                    const MINCOBlackboxOptimizerS3::Candidate &candidate) -> std::pair<double, bool>
                {
                    DirectCandidate evaluated;
                    evaluate_built_direct_candidate(candidate, evaluated);
                    insert_meta_candidate(evaluated);
                    return std::make_pair(evaluated.cost, evaluated.feasible);
                },
                igoOptions,
                budget);
            igoResult = blackbox_result.igo_result;
        }
        else
        {
            IGO igo;
            igoResult = igo.optimize(
                lower, upper, y0,
                [&insert_meta_candidate, &evaluate_direct_candidate](const Eigen::VectorXd &candidate_y) -> std::pair<double, bool>
                {
                    DirectCandidate evaluated;
                    evaluate_direct_candidate(candidate_y, evaluated);
                    insert_meta_candidate(evaluated);
                    return std::make_pair(evaluated.cost, evaluated.feasible);
                },
                igoOptions,
                budget);
        }

        DirectCandidate initial_candidate;
        const Eigen::VectorXd initial_y =
            direct_optimizer_ready ? direct_optimizer.initialGuess() : y0;
        if (evaluate_direct_candidate(initial_y, initial_candidate))
        {
            insert_meta_candidate(initial_candidate);
        }

        const DirectCandidate *best_direct = nullptr;
        if (!archive.empty())
        {
            best_direct = &archive.front();
            if (!direct_to_x(best_direct->y, result.best_x))
            {
                result.best_x = fallback_x;
            }
        }
        else
        {
            result.best_x = fallback_x;
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
            result.status = "MetaPlanner-style " + direct_family_name +
                            " " + direct_mode_name +
                            " IGO optimizer (wall-clock budget reached)";
        }
        else if (result.hit_eval_budget)
        {
            result.status = "MetaPlanner-style " + direct_family_name +
                            " " + direct_mode_name +
                            " IGO optimizer (evaluation budget reached)";
        }
        else if (igoResult.converged)
        {
            result.status = "MetaPlanner-style " + direct_family_name +
                            " " + direct_mode_name +
                            " IGO optimizer (converged)";
        }
        else
        {
            result.status = "MetaPlanner-style " + direct_family_name +
                            " " + direct_mode_name +
                            " IGO optimizer";
        }

        if (best_direct != nullptr)
        {
            MINCOBlackboxOptimizerS3::Candidate final_candidate;
            if (direct_optimizer_ready &&
                direct_optimizer.buildCandidate(best_direct->y, final_candidate))
            {
                result.objective =
                    evaluate_simple_candidate(final_candidate,
                                              &result.violations,
                                              nullptr);

                if (std::isfinite(result.objective) &&
                    final_candidate.trajectory.getPieceNum() > 0)
                {
                    result.has_solution = true;
                    result.total_duration =
                        final_candidate.trajectory.getTotalDuration();
                    result.trajectory_length =
                        approximateTrajectoryLength(final_candidate.trajectory,
                                                    std::max(8, 4 * integralRes));
                    result.has_meta_direct_solution = true;
                    result.meta_direct_points = final_candidate.points;
                    result.meta_direct_times = final_candidate.times;
                }
            }
            else
            {
                Eigen::Matrix3Xd direct_points;
                Eigen::VectorXd direct_times;
                if (unpack_direct(best_direct->y, direct_points, direct_times))
                {
                    if (use_nubs_direct)
                    {
                        DirectNUBSCandidate final_candidate;
                        final_candidate.y = best_direct->y;
                        final_candidate.points = direct_points;
                        final_candidate.times = direct_times;
                        if (build_direct_nubs(final_candidate.points,
                                              final_candidate.times,
                                              final_candidate.trajectory,
                                              final_candidate.energy))
                        {
                            result.objective =
                                options.meta_optimizer_use_control_point_objective
                                    ? evaluate_control_point_nubs_candidate(
                                          final_candidate,
                                          &result.violations)
                                    : evaluate_simple_nubs_candidate(
                                          final_candidate,
                                          &result.violations);
                            if (std::isfinite(result.objective) &&
                                final_candidate.trajectory.getPieceNum() > 0)
                            {
                                result.has_solution = true;
                                result.total_duration =
                                    final_candidate.trajectory.getTotalDuration();
                                result.trajectory_length =
                                    approximateNUBSTrajectoryLength(
                                        final_candidate.trajectory,
                                        std::max(8, 4 * integralRes));
                                result.has_meta_direct_solution = true;
                                result.has_meta_direct_nubs_solution = true;
                                result.meta_direct_points = direct_points;
                                result.meta_direct_times = direct_times;
                            }
                        }
                    }
                    else
                    {
                        minco::MINCO_S3NU direct_minco;
                        MINCOBlackboxOptimizerS3::Candidate final_candidate;
                        final_candidate.y = best_direct->y;
                        final_candidate.points = direct_points;
                        final_candidate.times = direct_times;
                        if (build_direct_trajectory(final_candidate.points,
                                                    final_candidate.times,
                                                    direct_minco,
                                                    final_candidate.trajectory,
                                                    final_candidate.energy))
                        {
                            result.objective =
                                evaluate_simple_candidate(final_candidate,
                                                          &result.violations,
                                                          nullptr);
                            if (std::isfinite(result.objective) &&
                                final_candidate.trajectory.getPieceNum() > 0)
                            {
                                result.has_solution = true;
                                result.total_duration =
                                    final_candidate.trajectory.getTotalDuration();
                                result.trajectory_length =
                                    approximateTrajectoryLength(final_candidate.trajectory,
                                                                std::max(8, 4 * integralRes));
                                result.has_meta_direct_solution = true;
                                result.meta_direct_points = direct_points;
                                result.meta_direct_times = direct_times;
                            }
                        }
                    }
                }
            }
        }

        if (!result.has_solution)
        {
            Trajectory<5> local_traj;
            result.objective =
                evaluateMetaPlannerFitness(result.best_x, options,
                                           &result.violations, &local_traj);
            if (std::isfinite(result.objective) && local_traj.getPieceNum() > 0)
            {
                result.has_solution = true;
                result.total_duration = local_traj.getTotalDuration();
                result.trajectory_length =
                    approximateTrajectoryLength(local_traj, std::max(8, 4 * integralRes));
            }
        }
        const std::string objective_name =
            use_nubs_direct && options.meta_optimizer_use_control_point_objective
                ? "control-point objective"
                : "simple sampled objective";
        result.status = result.status + " [fixed " +
                        std::to_string(point_count) +
                        "-midpoint direct " + direct_family_name +
                        " " + direct_mode_name + " " +
                        objective_name + "]";
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
            bsplinetrajectory::NUBSTrajectory<3> nubs_trajectory;
            bool has_nubs_trajectory = false;
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
                if (solver.buildMetaDirectNUBSTrajectory(result.summary,
                                                         result.nubs_trajectory))
                {
                    result.has_nubs_trajectory = true;
                }
                else
                {
                    minco::MINCO_S3NU direct_minco;
                    if (solver.buildMetaDirectJerkOpt(result.summary, direct_minco))
                    {
                        direct_minco.getTrajectory(result.trajectory);
                    }
                    else
                    {
                        solver.evaluateObjectiveOnly(result.summary.best_x,
                                                     nullptr,
                                                     &result.trajectory);
                    }
                }
            }
            return result;
        }
    };
}

#endif
