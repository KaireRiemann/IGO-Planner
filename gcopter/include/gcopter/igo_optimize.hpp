#ifndef GCOPTER_IGO_OPTIMIZE_HPP
#define GCOPTER_IGO_OPTIMIZE_HPP

#include "gcopter/gcopter.hpp"
#include "gcopter/trajectory.hpp"

namespace gcopter
{
    inline GCOPTER_PolytopeSFC::SolverResult
    GCOPTER_PolytopeSFC::solveIGO(const Eigen::VectorXd &initialX,
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

        result.wall_time = elapsed;

        Trajectory<5> traj;
        result.objective = evaluateObjectiveOnly(result.best_x, &result.violations, &traj);
        if (std::isfinite(result.objective) && traj.getPieceNum() > 0)
        {
            result.has_solution = true;
            result.total_duration = traj.getTotalDuration();
            result.trajectory_length =
                approximateTrajectoryLength(traj, std::max(8, 4 * integralRes));
        }

        return result;
    }

    class CorridorIGOTrajectoryPlanner
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
            result.summary = solver.solveIGO(initial_x, options);
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
