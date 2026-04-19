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
            return std::numeric_limits<double>::infinity();
        }

        Eigen::Map<const Eigen::VectorXd> tau(x.data(), temporalDim);
        Eigen::Map<const Eigen::VectorXd> xi(x.data() + temporalDim, spatialDim);

        forwardT(tau, times);
        forwardP(xi, vPolyIdx, vPolytopes, points);
        minco.setParameters(points, times);

        double corridor_integral, velocity_integral, body_rate_integral;
        double tilt_integral, thrust_integral;
        TrajectoryViolationMetrics local_metrics;
        accumulateViolationIntegrals(times, minco.getCoeffs(),
                                     hPolyIdx, hPolytopes,
                                     integralRes,
                                     magnitudeBd, flatmap,
                                     corridor_integral,
                                     velocity_integral,
                                     body_rate_integral,
                                     tilt_integral,
                                     thrust_integral,
                                     &local_metrics);

        Trajectory<5> local_traj;
        minco.getTrajectory(local_traj);

        const double sample_dt =
            std::max(1.0e-3, options.meta_optimizer_sample_dt);
        const double max_acceleration =
            std::max(positiveEps(), options.meta_optimizer_max_acceleration);
        double acceleration_integral = 0.0;
        for (double t = 0.0; t <= local_traj.getTotalDuration() + 0.5 * sample_dt; t += sample_dt)
        {
            const double clamped_t =
                std::min(t, local_traj.getTotalDuration());
            const double acc_excess =
                std::max(0.0, local_traj.getAcc(clamped_t).norm() - max_acceleration);
            acceleration_integral += acc_excess * sample_dt;
        }

        double energy = 0.0;
        minco.getEnergy(energy);
        const double inv_sample_dt = 1.0 / sample_dt;
        const double penalty_cost =
            std::max(0.0, options.meta_optimizer_collision_weight) * corridor_integral * inv_sample_dt +
            std::max(0.0, options.meta_optimizer_velocity_weight) * velocity_integral * inv_sample_dt +
            std::max(0.0, options.meta_optimizer_acceleration_weight) * acceleration_integral * inv_sample_dt +
            std::max(0.0, options.meta_optimizer_body_rate_weight) * body_rate_integral * inv_sample_dt +
            std::max(0.0, options.meta_optimizer_tilt_weight) * tilt_integral * inv_sample_dt +
            std::max(0.0, options.meta_optimizer_thrust_weight) * thrust_integral * inv_sample_dt;
        const double fitness =
            std::max(0.0, options.meta_optimizer_time_weight) * times.sum() +
            std::max(0.0, options.meta_optimizer_length_weight) *
                approximateTrajectoryLength(local_traj, std::max(8, 4 * integralRes)) +
            std::max(0.0, options.meta_optimizer_energy_weight) * energy +
            penalty_cost;

        local_metrics.penalty_cost = penalty_cost;

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

        const int archive_limit = std::max(1, options.archive_top_k);
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
                TrajectoryViolationMetrics local_metrics;
                evaluated.x = candidate_x;
                evaluated.cost =
                    evaluateMetaPlannerFitness(candidate_x, options, &local_metrics, nullptr);
                evaluated.max_violation = local_metrics.maxViolation();
                evaluated.feasible =
                    std::isfinite(evaluated.cost) &&
                    evaluated.max_violation <= options.feasibility_tol;
                insertArchiveCandidate(archive, evaluated, archive_limit);
                return std::make_pair(evaluated.cost, std::isfinite(evaluated.cost));
            },
            igoOptions,
            budget);

        EvaluatedCandidate initial_candidate;
        TrajectoryViolationMetrics initial_metrics;
        initial_candidate.x = x0;
        initial_candidate.cost =
            evaluateMetaPlannerFitness(x0, options, &initial_metrics, nullptr);
        initial_candidate.max_violation = initial_metrics.maxViolation();
        initial_candidate.feasible =
            std::isfinite(initial_candidate.cost) &&
            initial_candidate.max_violation <= options.feasibility_tol;
        insertArchiveCandidate(archive, initial_candidate, archive_limit);

        if (!archive.empty())
        {
            result.best_x = archive.front().x;
        }
        else
        {
            Eigen::VectorXd fallback_seed;
            liftMetaToX(z0, fallback_seed);
            result.best_x =
                fallback_seed.size() == getDecisionDim() && fallback_seed.allFinite()
                    ? fallback_seed
                    : x0;
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
            result.status = "MetaPlanner-style IGO optimizer (wall-clock budget reached)";
        }
        else if (result.hit_eval_budget)
        {
            result.status = "MetaPlanner-style IGO optimizer (evaluation budget reached)";
        }
        else if (igoResult.converged)
        {
            result.status = "MetaPlanner-style IGO optimizer (converged)";
        }
        else
        {
            result.status = "MetaPlanner-style IGO optimizer";
        }

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
