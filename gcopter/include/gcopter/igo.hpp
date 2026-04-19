#ifndef GCOPTER_IGO_HPP
#define GCOPTER_IGO_HPP

#include <eigen3/Eigen/Eigen>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace gcopter
{
    class IGO
    {
    public:
        struct Options
        {
            int population = 32;
            int max_iterations = 200;
            double min_sigma = 1.0e-3;
            double eta_mean = 0.45;
            double eta_sigma = 0.20;
            double elite_ratio = 0.35;
            int stall_iterations = 20;
            double rel_cost_tol = 1.0e-8;
            double sigma_convergence_ratio = 2.0;
            unsigned int seed = 0;
        };

        struct Budget
        {
            int max_evaluations = 0;
            double max_wall_time = 0.0;
        };

        struct Result
        {
            double best_cost = std::numeric_limits<double>::infinity();
            bool best_feasible = false;
            Eigen::VectorXd best_x;
            int eval_count = 0;
            int iterations = 0;
            bool converged = false;
            bool hit_eval_budget = false;
            bool hit_time_budget = false;
            std::vector<double> convergence_curve;
        };

        typedef std::function<std::pair<double, bool>(const Eigen::VectorXd &)> Objective;

    private:
        struct Candidate
        {
            Eigen::VectorXd x;
            double cost = std::numeric_limits<double>::infinity();
            bool feasible = false;
        };

        static inline Eigen::VectorXd clampVector(const Eigen::VectorXd &x,
                                                  const Eigen::VectorXd &lower,
                                                  const Eigen::VectorXd &upper)
        {
            return x.cwiseMax(lower).cwiseMin(upper);
        }

        static inline bool budgetReached(const Budget &budget,
                                         const std::chrono::steady_clock::time_point &start_time,
                                         Result &result)
        {
            if (budget.max_evaluations > 0 && result.eval_count >= budget.max_evaluations)
            {
                result.hit_eval_budget = true;
                return true;
            }

            if (budget.max_wall_time > 0.0)
            {
                const std::chrono::duration<double> elapsed =
                    std::chrono::steady_clock::now() - start_time;
                if (elapsed.count() >= budget.max_wall_time)
                {
                    result.hit_time_budget = true;
                    return true;
                }
            }

            return false;
        }

    public:
        inline Result optimize(const Eigen::VectorXd &lower,
                               const Eigen::VectorXd &upper,
                               const Eigen::VectorXd &initial_guess,
                               const Objective &objective,
                               const Options &options,
                               const Budget &budget) const
        {
            return optimize(lower, upper, initial_guess, objective, options, budget, nullptr);
        }

        inline Result optimize(const Eigen::VectorXd &lower,
                               const Eigen::VectorXd &upper,
                               const Eigen::VectorXd &initial_guess,
                               const Objective &objective,
                               const Options &options,
                               const Budget &budget,
                               const std::vector<Eigen::VectorXd> *initial_population) const
        {
            Result result;

            const int dim = initial_guess.size();
            if (dim <= 0 || lower.size() != dim || upper.size() != dim)
            {
                return result;
            }

            const int population = std::max(2, options.population);
            const int elite_num = std::max(2, static_cast<int>(std::round(options.elite_ratio * population)));

            std::mt19937 gen(options.seed);
            std::normal_distribution<double> normal_dist(0.0, 1.0);
            const std::chrono::steady_clock::time_point start_time =
                std::chrono::steady_clock::now();

            Eigen::VectorXd mu = clampVector(initial_guess, lower, upper);
            Eigen::VectorXd sigma = ((upper - lower).array() * 0.12).matrix();
            sigma = sigma.cwiseMax(Eigen::VectorXd::Constant(dim, options.min_sigma));

            auto sampleAround = [&](const Eigen::VectorXd &mean,
                                    const Eigen::VectorXd &stddev) -> Eigen::VectorXd
            {
                Eigen::VectorXd sampled(dim);
                for (int i = 0; i < dim; ++i)
                {
                    sampled(i) = mean(i) + stddev(i) * normal_dist(gen);
                }
                return clampVector(sampled, lower, upper);
            };

            auto evaluatePopulation = [&](std::vector<Candidate> &population_vec,
                                          int start_index = 0) -> bool
            {
                for (int i = start_index; i < static_cast<int>(population_vec.size()); ++i)
                {
                    if (budgetReached(budget, start_time, result))
                    {
                        return false;
                    }

                    const std::pair<double, bool> eval = objective(population_vec[i].x);
                    population_vec[i].cost = eval.first;
                    population_vec[i].feasible = eval.second;
                    ++result.eval_count;

                    if (population_vec[i].feasible)
                    {
                        if (!result.best_feasible || population_vec[i].cost < result.best_cost)
                        {
                            result.best_cost = population_vec[i].cost;
                            result.best_feasible = true;
                            result.best_x = population_vec[i].x;
                        }
                    }
                    else if (!result.best_feasible && population_vec[i].cost < result.best_cost)
                    {
                        result.best_cost = population_vec[i].cost;
                        result.best_x = population_vec[i].x;
                    }
                }
                return true;
            };

            auto updateDistribution = [&](const std::vector<Candidate> &population_vec) -> void
            {
                std::vector<int> order(population);
                std::iota(order.begin(), order.end(), 0);

                double min_cost = std::numeric_limits<double>::infinity();
                double max_cost = -std::numeric_limits<double>::infinity();
                for (const Candidate &cand : population_vec)
                {
                    if (std::isfinite(cand.cost))
                    {
                        min_cost = std::min(min_cost, cand.cost);
                        max_cost = std::max(max_cost, cand.cost);
                    }
                }
                if (!std::isfinite(min_cost) || !std::isfinite(max_cost))
                {
                    min_cost = 0.0;
                    max_cost = 1.0;
                }

                const double invalid_penalty = 2.0 * std::max(1.0, max_cost - min_cost) + 1.0;
                auto rankedCost = [&](int idx) -> double
                {
                    return population_vec[idx].cost +
                           (population_vec[idx].feasible ? 0.0 : invalid_penalty);
                };

                std::sort(order.begin(), order.end(),
                          [&](int lhs, int rhs)
                          {
                              return rankedCost(lhs) < rankedCost(rhs);
                          });

                Eigen::VectorXd weights(elite_num);
                for (int i = 0; i < elite_num; ++i)
                {
                    weights(i) = std::log(elite_num + 0.5) - std::log(i + 1.0);
                }
                weights = weights.cwiseMax(0.0);
                weights /= std::max(weights.sum(), 1.0e-12);

                Eigen::VectorXd new_mu = Eigen::VectorXd::Zero(dim);
                for (int i = 0; i < elite_num; ++i)
                {
                    new_mu += weights(i) * population_vec[order[i]].x;
                }

                Eigen::VectorXd centered_var = Eigen::VectorXd::Zero(dim);
                for (int i = 0; i < elite_num; ++i)
                {
                    const Eigen::VectorXd diff = population_vec[order[i]].x - new_mu;
                    centered_var += weights(i) * diff.cwiseProduct(diff);
                }

                mu = ((1.0 - options.eta_mean) * mu + options.eta_mean * new_mu)
                         .cwiseMax(lower)
                         .cwiseMin(upper);

                const Eigen::ArrayXd log_sigma_old = sigma.array().max(options.min_sigma).log();
                const Eigen::ArrayXd log_sigma_new =
                    centered_var.array().max(1.0e-16).sqrt().max(options.min_sigma).log();

                sigma = ((1.0 - options.eta_sigma) * log_sigma_old +
                         options.eta_sigma * log_sigma_new)
                            .exp()
                            .matrix();
                sigma = sigma.cwiseMax(Eigen::VectorXd::Constant(dim, options.min_sigma));
            };

            std::vector<Candidate> population_vec(population);
            for (Candidate &cand : population_vec)
            {
                cand.x.resize(dim);
            }

            int init_index = 0;
            population_vec[init_index++].x = mu;
            if (initial_population != nullptr)
            {
                for (const Eigen::VectorXd &seed : *initial_population)
                {
                    if (init_index >= population)
                    {
                        break;
                    }
                    if (seed.size() != dim)
                    {
                        continue;
                    }
                    population_vec[init_index++].x =
                        clampVector(seed, lower, upper);
                }
            }
            for (int i = init_index; i < population; ++i)
            {
                population_vec[i].x = sampleAround(mu, sigma);
            }

            if (!evaluatePopulation(population_vec))
            {
                return result;
            }

            std::vector<double> best_cost_history;
            best_cost_history.reserve(std::max(1, options.max_iterations) + 1);

            for (int iter = 0; options.max_iterations == 0 || iter < options.max_iterations; ++iter)
            {
                updateDistribution(population_vec);

                if (result.best_x.size() == dim)
                {
                    population_vec.front().x = result.best_x;
                }
                else
                {
                    population_vec.front().x = mu;
                }
                for (int i = 1; i < population; ++i)
                {
                    population_vec[i].x = sampleAround(mu, sigma);
                }

                if (!evaluatePopulation(population_vec))
                {
                    break;
                }

                result.iterations = iter + 1;
                result.convergence_curve.push_back(result.best_cost);
                best_cost_history.push_back(result.best_cost);

                const int stall_iterations = std::max(5, options.stall_iterations);
                if (static_cast<int>(best_cost_history.size()) >= stall_iterations)
                {
                    double mean_abs_diff = 0.0;
                    for (int i = 1; i < stall_iterations; ++i)
                    {
                        mean_abs_diff += std::abs(best_cost_history[best_cost_history.size() - i] -
                                                  best_cost_history[best_cost_history.size() - i - 1]);
                    }
                    mean_abs_diff /= static_cast<double>(stall_iterations - 1);

                    const double scale =
                        std::max(1.0, std::abs(result.best_cost));
                    const double sigma_threshold =
                        std::max(options.min_sigma * std::max(1.0, options.sigma_convergence_ratio),
                                 options.min_sigma + 1.0e-12);
                    if (mean_abs_diff <= options.rel_cost_tol * scale &&
                        sigma.maxCoeff() <= sigma_threshold)
                    {
                        result.converged = true;
                        break;
                    }
                }
            }

            return result;
        }
    };
}

#endif
