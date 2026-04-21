#ifndef GCOPTER_MINCO_BLACKBOX_OPTIMIZER_HPP
#define GCOPTER_MINCO_BLACKBOX_OPTIMIZER_HPP

#include "gcopter/igo.hpp"
#include "gcopter/minco.hpp"
#include "gcopter/trajectory.hpp"

#include <Eigen/Eigen>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <utility>

namespace gcopter
{
    class MINCOBlackboxOptimizerS3
    {
    public:
        enum class TimeMode
        {
            Direct,
            Profile
        };

        struct Options
        {
            TimeMode time_mode = TimeMode::Profile;
            double time_lb = 0.1;
            double time_ub = 8.0;
            double simple_vel_max = 4.0;
            double target_vel = 3.6;
            double time_floor_scale = 1.10;
            double initial_time_scale = 1.30;
            double total_slack_min_scale = 0.04;
            double total_slack_max_scale = 0.65;
            double beta_bound = 3.0;
            double gamma_bound = 3.0;
            double local_box_radius = 2.0;
        };

        struct Candidate
        {
            Eigen::VectorXd y;
            Eigen::Matrix3Xd points;
            Eigen::VectorXd times;
            Trajectory<5> trajectory;
            double energy = std::numeric_limits<double>::infinity();
        };

        struct Result
        {
            bool has_solution = false;
            double best_cost = std::numeric_limits<double>::infinity();
            bool best_feasible = false;
            Candidate best_candidate;
            IGO::Result igo_result;
        };

        typedef std::function<std::pair<double, bool>(const Candidate &)> CostFunction;
        typedef CostFunction Objective;

    public:
        inline void setWarmStartDecision(const Eigen::VectorXd &decision)
        {
            warm_start_decision_ = decision;
            has_warm_start_decision_ = decision.size() > 0 && decision.allFinite();
            if (configured_ && has_warm_start_decision_ &&
                warm_start_decision_.size() == decision_dim_)
            {
                initial_ = warm_start_decision_.cwiseMax(lower_).cwiseMin(upper_);
            }
        }

        inline void setWarmStartPhysical(const Eigen::Matrix3Xd &points,
                                         const Eigen::VectorXd &times)
        {
            warm_start_points_ = points;
            warm_start_times_ = times;
            has_warm_start_physical_ =
                points.allFinite() && times.allFinite() &&
                points.cols() >= 0 && times.size() > 0;
            if (configured_ && has_warm_start_physical_)
            {
                Eigen::VectorXd encoded;
                if (encodePhysical(warm_start_points_, warm_start_times_, encoded))
                {
                    initial_ = encoded.cwiseMax(lower_).cwiseMin(upper_);
                }
            }
        }

        inline void clearWarmStart()
        {
            warm_start_decision_.resize(0);
            warm_start_points_.resize(3, 0);
            warm_start_times_.resize(0);
            has_warm_start_decision_ = false;
            has_warm_start_physical_ = false;
        }

        inline bool setup(const Eigen::Matrix3d &head_state,
                          const Eigen::Matrix3d &tail_state,
                          const Eigen::Matrix3Xd &anchor_points,
                          const Eigen::Vector3d &workspace_min,
                          const Eigen::Vector3d &workspace_max,
                          const Options &options)
        {
            head_state_ = head_state;
            tail_state_ = tail_state;
            options_ = options;
            point_count_ = static_cast<int>(anchor_points.cols());
            piece_num_ = point_count_ + 1;
            if (piece_num_ <= 0 ||
                !head_state_.allFinite() ||
                !tail_state_.allFinite() ||
                !workspace_min.allFinite() ||
                !workspace_max.allFinite() ||
                (workspace_max.array() < workspace_min.array()).any())
            {
                configured_ = false;
                return false;
            }

            workspace_min_ = workspace_min;
            workspace_max_ = workspace_max;
            anchors_.resize(3, point_count_);
            for (int i = 0; i < point_count_; ++i)
            {
                anchors_.col(i) =
                    anchor_points.col(i).cwiseMax(workspace_min_).cwiseMin(workspace_max_);
            }

            const int point_dim = 3 * point_count_;
            const int time_dim = piece_num_;
            gamma_offset_ = point_dim + time_dim;
            decision_dim_ =
                options_.time_mode == TimeMode::Profile
                    ? gamma_offset_ + 1
                    : point_dim + time_dim;

            lower_.resize(decision_dim_);
            upper_.resize(decision_dim_);
            initial_.resize(decision_dim_);
            lower_.setZero();
            upper_.setZero();
            initial_.setZero();

            const double local_box_radius = std::max(0.0, options_.local_box_radius);
            for (int i = 0; i < point_count_; ++i)
            {
                const Eigen::Vector3d anchor = anchors_.col(i);
                Eigen::Vector3d local_lower =
                    (anchor.array() - local_box_radius).matrix().cwiseMax(workspace_min_);
                Eigen::Vector3d local_upper =
                    (anchor.array() + local_box_radius).matrix().cwiseMin(workspace_max_);
                local_lower = local_lower.cwiseMin(anchor);
                local_upper = local_upper.cwiseMax(anchor);

                lower_.segment<3>(3 * i) = local_lower;
                upper_.segment<3>(3 * i) = local_upper;
                initial_.segment<3>(3 * i) = anchor;
            }

            const int time_offset = point_dim;
            if (options_.time_mode == TimeMode::Profile)
            {
                const double beta_bound = std::max(0.1, options_.beta_bound);
                const double gamma_bound = std::max(0.1, options_.gamma_bound);
                lower_.segment(time_offset, time_dim).setConstant(-beta_bound);
                upper_.segment(time_offset, time_dim).setConstant(beta_bound);
                lower_(gamma_offset_) = -gamma_bound;
                upper_(gamma_offset_) = gamma_bound;
                initial_.segment(time_offset, time_dim).setZero();
                initial_(gamma_offset_) = 0.0;
            }
            else
            {
                const double time_lb = std::max(eps(), options_.time_lb);
                const double time_ub = std::max(time_lb + eps(), options_.time_ub);
                lower_.segment(time_offset, time_dim).setConstant(time_lb);
                upper_.segment(time_offset, time_dim).setConstant(time_ub);
                initial_.segment(time_offset, time_dim) =
                    clampTimesToGeometry(anchors_, initialDirectTimes(anchors_));
            }

            initial_ = initial_.cwiseMax(lower_).cwiseMin(upper_);
            configured_ = true;
            Eigen::VectorXd warm_start_encoded;
            if (has_warm_start_decision_ &&
                warm_start_decision_.size() == decision_dim_ &&
                warm_start_decision_.allFinite())
            {
                initial_ = warm_start_decision_.cwiseMax(lower_).cwiseMin(upper_);
            }
            else if (has_warm_start_physical_ &&
                     encodePhysical(warm_start_points_,
                                    warm_start_times_,
                                    warm_start_encoded))
            {
                initial_ = warm_start_encoded.cwiseMax(lower_).cwiseMin(upper_);
            }
            configured_ = initial_.allFinite() && lower_.allFinite() && upper_.allFinite();
            return configured_;
        }

        inline Result optimize(const CostFunction &cost_function,
                               const IGO::Options &igo_options,
                               const IGO::Budget &budget) const
        {
            Result result;
            if (!configured_ || !cost_function)
            {
                return result;
            }

            IGO igo;
            result.igo_result = igo.optimize(
                lower_, upper_, initial_,
                [this, &cost_function](const Eigen::VectorXd &y) -> std::pair<double, bool>
                {
                    Candidate candidate;
                    if (!buildCandidate(y, candidate))
                    {
                        return std::make_pair(std::numeric_limits<double>::infinity(), false);
                    }
                    return cost_function(candidate);
                },
                igo_options,
                budget);

            result.best_cost = result.igo_result.best_cost;
            result.best_feasible = result.igo_result.best_feasible;
            if (result.igo_result.best_x.size() == decision_dim_ &&
                buildCandidate(result.igo_result.best_x, result.best_candidate))
            {
                result.has_solution = std::isfinite(result.best_cost);
            }
            return result;
        }

        inline Result solve(const CostFunction &cost_function,
                            const IGO::Options &igo_options,
                            const IGO::Budget &budget) const
        {
            return optimize(cost_function, igo_options, budget);
        }

        inline bool buildCandidate(const Eigen::VectorXd &y,
                                   Candidate &candidate) const
        {
            if (!configured_ || y.size() != decision_dim_ || !y.allFinite())
            {
                return false;
            }

            candidate.y = y;
            candidate.points.resize(3, point_count_);
            for (int i = 0; i < point_count_; ++i)
            {
                candidate.points.col(i) = y.segment<3>(3 * i);
            }

            const int time_offset = 3 * point_count_;
            if (options_.time_mode == TimeMode::Profile)
            {
                const Eigen::VectorXd beta = y.segment(time_offset, piece_num_);
                candidate.times = recoverProfileTimes(candidate.points, beta, y(gamma_offset_));
            }
            else
            {
                candidate.times =
                    clampTimesToGeometry(candidate.points,
                                         y.segment(time_offset, piece_num_));
            }

            if (candidate.points.cols() != piece_num_ - 1 ||
                candidate.times.size() != piece_num_ ||
                !candidate.points.allFinite() ||
                !candidate.times.allFinite() ||
                (candidate.times.array() <= eps()).any())
            {
                return false;
            }

            minco::MINCO_S3NU jerk_opt;
            jerk_opt.setConditions(head_state_, tail_state_, piece_num_);
            jerk_opt.setParameters(candidate.points, candidate.times);
            jerk_opt.getEnergy(candidate.energy);
            jerk_opt.getTrajectory(candidate.trajectory);
            return std::isfinite(candidate.energy) &&
                   candidate.trajectory.getPieceNum() > 0;
        }

        inline const Eigen::VectorXd &initialGuess() const
        {
            return initial_;
        }

        inline const Eigen::VectorXd &lowerBound() const
        {
            return lower_;
        }

        inline const Eigen::VectorXd &upperBound() const
        {
            return upper_;
        }

        inline int decisionDim() const
        {
            return decision_dim_;
        }

        inline int pieceNum() const
        {
            return piece_num_;
        }

        inline int pointCount() const
        {
            return point_count_;
        }

        inline std::string timeModeName() const
        {
            return options_.time_mode == TimeMode::Profile
                       ? "fixed P,beta,gamma"
                       : "fixed P,T";
        }

    private:
        static inline double eps()
        {
            return 1.0e-9;
        }

        static inline double sigmoid(const double x)
        {
            if (x >= 0.0)
            {
                const double z = std::exp(-x);
                return 1.0 / (1.0 + z);
            }
            const double z = std::exp(x);
            return z / (1.0 + z);
        }

        static inline Eigen::VectorXd softmax(const Eigen::VectorXd &x)
        {
            if (x.size() <= 0)
            {
                return Eigen::VectorXd();
            }
            const double max_coeff = x.maxCoeff();
            Eigen::VectorXd exp_x = (x.array() - max_coeff).exp();
            const double sum = std::max(eps(), exp_x.sum());
            return exp_x / sum;
        }

        inline Eigen::Vector3d pointAt(const Eigen::Matrix3Xd &points,
                                       const int idx) const
        {
            if (idx <= 0)
            {
                return head_state_.col(0);
            }
            if (idx >= point_count_ + 1)
            {
                return tail_state_.col(0);
            }
            return points.col(idx - 1);
        }

        inline Eigen::VectorXd initialDirectTimes(const Eigen::Matrix3Xd &points) const
        {
            Eigen::VectorXd times(piece_num_);
            const double time_lb = std::max(eps(), options_.time_lb);
            const double initial_time_scale = std::max(1.0, options_.initial_time_scale);
            const double target_vel = std::max(eps(), options_.target_vel);
            for (int i = 0; i < piece_num_; ++i)
            {
                const double segment_length =
                    (pointAt(points, i + 1) - pointAt(points, i)).norm();
                times(i) = std::max(time_lb,
                                    initial_time_scale * segment_length / target_vel);
            }
            return times;
        }

        inline Eigen::VectorXd clampTimesToGeometry(const Eigen::Matrix3Xd &points,
                                                    const Eigen::VectorXd &raw_times) const
        {
            Eigen::VectorXd times(piece_num_);
            if (raw_times.size() != piece_num_)
            {
                return Eigen::VectorXd::Constant(
                    piece_num_, std::numeric_limits<double>::quiet_NaN());
            }

            const double time_lb = std::max(eps(), options_.time_lb);
            const double time_ub = std::max(time_lb + eps(), options_.time_ub);
            const double simple_vel_max = std::max(eps(), options_.simple_vel_max);
            const double target_vel = std::max(eps(), options_.target_vel);
            const double time_upper_scale =
                std::max(1.05, 1.0 + std::max(0.0, options_.total_slack_max_scale));

            for (int i = 0; i < piece_num_; ++i)
            {
                const double segment_length =
                    (pointAt(points, i + 1) - pointAt(points, i)).norm();
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
        }

        inline Eigen::VectorXd recoverProfileTimes(const Eigen::Matrix3Xd &points,
                                                   const Eigen::VectorXd &beta,
                                                   const double gamma) const
        {
            Eigen::VectorXd floor_times(piece_num_);
            const double time_lb = std::max(eps(), options_.time_lb);
            const double simple_vel_max = std::max(eps(), options_.simple_vel_max);
            const double time_floor_scale = std::max(0.0, options_.time_floor_scale);
            for (int i = 0; i < piece_num_; ++i)
            {
                const double segment_length =
                    (pointAt(points, i + 1) - pointAt(points, i)).norm();
                floor_times(i) =
                    std::max(time_lb,
                             time_floor_scale * segment_length / simple_vel_max);
            }

            const double floor_sum = std::max(eps(), floor_times.sum());
            const double slack_min =
                std::max(0.0, options_.total_slack_min_scale) * floor_sum;
            const double slack_max =
                std::max(slack_min, options_.total_slack_max_scale * floor_sum);
            const double slack_total =
                slack_min + sigmoid(gamma) * (slack_max - slack_min);
            return floor_times + softmax(beta) * slack_total;
        }

        inline bool encodePhysical(const Eigen::Matrix3Xd &points,
                                   const Eigen::VectorXd &times,
                                   Eigen::VectorXd &y) const
        {
            if (!configured_ ||
                points.cols() != point_count_ ||
                times.size() != piece_num_ ||
                !points.allFinite() ||
                !times.allFinite() ||
                (times.array() <= eps()).any())
            {
                return false;
            }

            y.resize(decision_dim_);
            for (int i = 0; i < point_count_; ++i)
            {
                y.segment<3>(3 * i) =
                    points.col(i).cwiseMax(lower_.segment<3>(3 * i))
                                 .cwiseMin(upper_.segment<3>(3 * i));
            }

            const int time_offset = 3 * point_count_;
            Eigen::Matrix3Xd encoded_points(3, point_count_);
            for (int i = 0; i < point_count_; ++i)
            {
                encoded_points.col(i) = y.segment<3>(3 * i);
            }

            if (options_.time_mode == TimeMode::Direct)
            {
                y.segment(time_offset, piece_num_) =
                    clampTimesToGeometry(encoded_points, times);
                return y.allFinite();
            }

            Eigen::VectorXd floor_times(piece_num_);
            const double time_lb = std::max(eps(), options_.time_lb);
            const double simple_vel_max = std::max(eps(), options_.simple_vel_max);
            const double time_floor_scale = std::max(0.0, options_.time_floor_scale);
            for (int i = 0; i < piece_num_; ++i)
            {
                const double segment_length =
                    (pointAt(encoded_points, i + 1) - pointAt(encoded_points, i)).norm();
                floor_times(i) =
                    std::max(time_lb,
                             time_floor_scale * segment_length / simple_vel_max);
            }

            const Eigen::VectorXd extra_time =
                (times - floor_times).cwiseMax(Eigen::VectorXd::Zero(piece_num_));
            const double floor_sum = std::max(eps(), floor_times.sum());
            const double slack_min =
                std::max(0.0, options_.total_slack_min_scale) * floor_sum;
            const double slack_max =
                std::max(slack_min, options_.total_slack_max_scale * floor_sum);
            const double extra_sum = extra_time.sum();

            Eigen::VectorXd ratio =
                Eigen::VectorXd::Constant(piece_num_,
                                          1.0 / static_cast<double>(piece_num_));
            if (extra_sum > eps())
            {
                ratio = extra_time / extra_sum;
                ratio = ratio.array().max(1.0e-6);
                ratio /= std::max(eps(), ratio.sum());
            }

            Eigen::VectorXd beta = ratio.array().log().matrix();
            beta.array() -= beta.mean();
            const double beta_bound = std::max(0.1, options_.beta_bound);
            beta = beta.cwiseMax(Eigen::VectorXd::Constant(piece_num_, -beta_bound))
                       .cwiseMin(Eigen::VectorXd::Constant(piece_num_, beta_bound));
            y.segment(time_offset, piece_num_) = beta;

            double gamma = 0.0;
            if (slack_max > slack_min + eps())
            {
                const double normalized =
                    std::max(1.0e-6,
                             std::min(1.0 - 1.0e-6,
                                      (extra_sum - slack_min) /
                                          std::max(eps(), slack_max - slack_min)));
                gamma = std::log(normalized / (1.0 - normalized));
            }
            const double gamma_bound = std::max(0.1, options_.gamma_bound);
            y(gamma_offset_) = std::max(-gamma_bound, std::min(gamma_bound, gamma));
            return y.allFinite();
        }

    private:
        bool configured_ = false;
        int point_count_ = 0;
        int piece_num_ = 0;
        int decision_dim_ = 0;
        int gamma_offset_ = 0;

        Eigen::Matrix3d head_state_ = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d tail_state_ = Eigen::Matrix3d::Zero();
        Eigen::Matrix3Xd anchors_;
        Eigen::Vector3d workspace_min_ = Eigen::Vector3d::Zero();
        Eigen::Vector3d workspace_max_ = Eigen::Vector3d::Zero();
        Eigen::VectorXd initial_;
        Eigen::VectorXd lower_;
        Eigen::VectorXd upper_;
        Options options_;

        bool has_warm_start_decision_ = false;
        Eigen::VectorXd warm_start_decision_;
        bool has_warm_start_physical_ = false;
        Eigen::Matrix3Xd warm_start_points_;
        Eigen::VectorXd warm_start_times_;
    };
}

#endif
