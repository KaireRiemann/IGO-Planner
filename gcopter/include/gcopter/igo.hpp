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
            double abs_cost_tol = 0.0;
            double sigma_convergence_ratio = 2.0;
            double initial_sigma_scale = 0.12;
            int exploration_boost_after_invalid_iters = 0;
            double invalid_exploration_boost = 1.0;
            int conservative_tail_dims = 0;
            double min_tail_sigma = 0.0;
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

        static inline Eigen::VectorXd rowToVector(const Eigen::RowVectorXd &row)
        {
            Eigen::VectorXd vec(row.size());
            vec = row.transpose();
            return vec;
        }

        static inline Eigen::RowVectorXd vectorToRow(const Eigen::VectorXd &vec)
        {
            Eigen::RowVectorXd row(vec.size());
            row = vec.transpose();
            return row;
        }

        int legacy_dim_ = 0;
        int legacy_population_ = 0;
        int legacy_max_iterations_ = 0;
        double legacy_min_sigma_ = 1.0e-3;
        double legacy_eta_mean_ = 0.45;
        double legacy_eta_sigma_ = 0.20;
        double legacy_elite_ratio_ = 0.35;
        double legacy_initial_sigma_scale_ = 0.12;
        unsigned int legacy_seed_ = 0;
        bool legacy_initialized_ = false;
        Eigen::VectorXd legacy_lower_;
        Eigen::VectorXd legacy_upper_;
        Eigen::VectorXd legacy_initial_guess_;
        std::vector<Eigen::VectorXd> legacy_initial_population_;
        std::vector<double> legacy_convergence_curve_;
        std::mt19937 legacy_gen_{std::random_device{}()};
        std::normal_distribution<double> legacy_normal_{0.0, 1.0};

        inline bool setupLegacy(const int nP,
                                const int MaxIt,
                                const Eigen::VectorXd &ub,
                                const Eigen::VectorXd &lb,
                                const int dim)
        {
            legacy_initialized_ = false;
            legacy_dim_ = 0;
            legacy_population_ = 0;
            legacy_max_iterations_ = 0;
            legacy_initial_guess_.resize(0);
            legacy_initial_population_.clear();
            legacy_convergence_curve_.clear();

            if (dim <= 0 || nP <= 0 || ub.size() != dim || lb.size() != dim)
            {
                return false;
            }

            legacy_dim_ = dim;
            legacy_population_ = std::max(2, nP);
            legacy_max_iterations_ = std::max(0, MaxIt);
            legacy_lower_ = lb;
            legacy_upper_ = ub;
            legacy_seed_ = legacy_gen_();
            legacy_convergence_curve_.assign(
                legacy_max_iterations_, std::numeric_limits<double>::infinity());
            return true;
        }

        inline bool legacySetupMatches(const int nP,
                                       const int MaxIt,
                                       const Eigen::VectorXd &ub,
                                       const Eigen::VectorXd &lb,
                                       const int dim) const
        {
            return legacy_initialized_ &&
                   legacy_dim_ == dim &&
                   legacy_population_ == std::max(2, nP) &&
                   legacy_max_iterations_ == std::max(0, MaxIt) &&
                   legacy_lower_.size() == dim &&
                   legacy_upper_.size() == dim &&
                   legacy_lower_.isApprox(lb) &&
                   legacy_upper_.isApprox(ub);
        }

        inline int legacyTimeDims() const
        {
            return (legacy_dim_ + 3) / 4;
        }

        inline Eigen::VectorXd sampleLegacyAround(const Eigen::VectorXd &mean,
                                                  const Eigen::VectorXd &stddev)
        {
            Eigen::VectorXd sampled(legacy_dim_);
            for (int i = 0; i < legacy_dim_; ++i)
            {
                sampled(i) = mean(i) + stddev(i) * legacy_normal_(legacy_gen_);
            }
            return clampVector(sampled, legacy_lower_, legacy_upper_);
        }

        inline void storeLegacyCurve(const Result &result)
        {
            legacy_convergence_curve_.assign(
                legacy_max_iterations_, std::numeric_limits<double>::infinity());
            const std::size_t count =
                std::min(legacy_convergence_curve_.size(),
                         result.convergence_curve.size());
            for (std::size_t i = 0; i < count; ++i)
            {
                legacy_convergence_curve_[i] = result.convergence_curve[i];
            }
        }

    public:
        inline void initialize(const int nP,
                               const int MaxIt,
                               const Eigen::VectorXd &ub,
                               const Eigen::VectorXd &lb,
                               const int dim)
        {
            if (!setupLegacy(nP, MaxIt, ub, lb, dim))
            {
                return;
            }

            legacy_initial_sigma_scale_ = 0.25;
            legacy_initial_guess_ = 0.5 * (legacy_lower_ + legacy_upper_);
            Eigen::VectorXd sigma =
                ((legacy_upper_ - legacy_lower_).array() * legacy_initial_sigma_scale_).matrix();
            sigma = sigma.cwiseMax(
                Eigen::VectorXd::Constant(legacy_dim_, legacy_min_sigma_));

            legacy_initial_population_.clear();
            legacy_initial_population_.reserve(std::max(0, legacy_population_ - 1));
            for (int i = 1; i < legacy_population_; ++i)
            {
                legacy_initial_population_.push_back(
                    sampleLegacyAround(legacy_initial_guess_, sigma));
            }
            legacy_initialized_ = true;
        }

        inline void initialize(const int nP,
                               const int MaxIt,
                               const Eigen::VectorXd &ub,
                               const Eigen::VectorXd &lb,
                               const int dim,
                               const Eigen::RowVectorXd &previous_best)
        {
            if (!setupLegacy(nP, MaxIt, ub, lb, dim))
            {
                return;
            }

            legacy_initial_sigma_scale_ = 0.12;
            legacy_initial_guess_ =
                previous_best.size() == dim
                    ? clampVector(rowToVector(previous_best), legacy_lower_, legacy_upper_)
                    : 0.5 * (legacy_lower_ + legacy_upper_);

            Eigen::VectorXd sigma =
                ((legacy_upper_ - legacy_lower_).array() * legacy_initial_sigma_scale_).matrix();
            sigma = sigma.cwiseMax(
                Eigen::VectorXd::Constant(legacy_dim_, legacy_min_sigma_));

            legacy_initial_population_.clear();
            legacy_initial_population_.reserve(std::max(0, legacy_population_ - 1));
            for (int i = 1; i < legacy_population_; ++i)
            {
                legacy_initial_population_.push_back(
                    sampleLegacyAround(legacy_initial_guess_, sigma));
            }
            legacy_initialized_ = true;
        }

        inline void initializeWithPriority(const int nP,
                                           const int MaxIt,
                                           const Eigen::VectorXd &ub,
                                           const Eigen::VectorXd &lb,
                                           const int dim,
                                           const int priority_based_count,
                                           const double priority_based_variation_radius,
                                           const Eigen::RowVectorXd &priority_waypoints)
        {
            if (!setupLegacy(nP, MaxIt, ub, lb, dim))
            {
                return;
            }

            legacy_initial_sigma_scale_ = 0.20;
            legacy_initial_guess_ =
                priority_waypoints.size() == dim
                    ? clampVector(rowToVector(priority_waypoints), legacy_lower_, legacy_upper_)
                    : 0.5 * (legacy_lower_ + legacy_upper_);

            Eigen::VectorXd sigma =
                ((legacy_upper_ - legacy_lower_).array() * legacy_initial_sigma_scale_).matrix();
            sigma = sigma.cwiseMax(
                Eigen::VectorXd::Constant(legacy_dim_, legacy_min_sigma_));

            legacy_initial_population_.clear();
            legacy_initial_population_.reserve(std::max(0, legacy_population_ - 1));

            const int guided_count =
                std::min(std::max(priority_based_count, 1), legacy_population_);
            for (int i = 1; i < guided_count; ++i)
            {
                Eigen::VectorXd guided_sigma = sigma;
                const int spatial_dims = std::max(0, legacy_dim_ - legacyTimeDims());
                if (spatial_dims > 0)
                {
                    guided_sigma.head(spatial_dims).setConstant(
                        std::max(priority_based_variation_radius, legacy_min_sigma_));
                }
                legacy_initial_population_.push_back(
                    sampleLegacyAround(legacy_initial_guess_, guided_sigma));
            }

            while (static_cast<int>(legacy_initial_population_.size()) <
                   legacy_population_ - 1)
            {
                legacy_initial_population_.push_back(
                    sampleLegacyAround(legacy_initial_guess_, sigma));
            }
            legacy_initialized_ = true;
        }

        inline std::pair<double, Eigen::RowVectorXd> optimize(
            const int nP,
            const int MaxIt,
            const Eigen::VectorXd &ub,
            const Eigen::VectorXd &lb,
            const int dim,
            std::function<std::pair<double, bool>(const Eigen::RowVectorXd &)> fobj)
        {
            if (!legacySetupMatches(nP, MaxIt, ub, lb, dim))
            {
                initialize(nP, MaxIt, ub, lb, dim);
            }
            if (!legacy_initialized_)
            {
                return std::make_pair(std::numeric_limits<double>::infinity(),
                                      Eigen::RowVectorXd());
            }

            Options options;
            options.population = legacy_population_;
            options.max_iterations =
                legacy_max_iterations_ > 0 ? legacy_max_iterations_ : 1;
            options.min_sigma = legacy_min_sigma_;
            options.eta_mean = legacy_eta_mean_;
            options.eta_sigma = legacy_eta_sigma_;
            options.elite_ratio = legacy_elite_ratio_;
            options.stall_iterations = 5;
            options.rel_cost_tol = 0.0;
            options.abs_cost_tol = 1.0e-3;
            options.sigma_convergence_ratio =
                std::numeric_limits<double>::infinity();
            options.initial_sigma_scale = legacy_initial_sigma_scale_;
            options.exploration_boost_after_invalid_iters = 3;
            options.invalid_exploration_boost = 1.10;
            options.conservative_tail_dims = legacyTimeDims();
            options.min_tail_sigma = 0.05;
            options.seed = legacy_seed_;

            Budget budget;
            if (legacy_max_iterations_ <= 0)
            {
                budget.max_evaluations = legacy_population_;
            }

            const Result result = optimize(
                legacy_lower_, legacy_upper_, legacy_initial_guess_,
                [&fobj](const Eigen::VectorXd &candidate) -> std::pair<double, bool>
                {
                    return fobj(vectorToRow(candidate));
                },
                options,
                budget,
                &legacy_initial_population_);

            storeLegacyCurve(result);
            legacy_initialized_ = false;

            if (result.best_x.size() != dim)
            {
                return std::make_pair(result.best_cost, Eigen::RowVectorXd());
            }
            return std::make_pair(result.best_cost, vectorToRow(result.best_x));
        }

        inline std::vector<double> getConvergenceCurve() const
        {
            return legacy_convergence_curve_;
        }

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
            Eigen::VectorXd sigma =
                ((upper - lower).array() *
                 std::max(0.0, options.initial_sigma_scale))
                    .matrix();
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

            int last_valid_count = 0;
            auto evaluatePopulation = [&](std::vector<Candidate> &population_vec,
                                          int start_index = 0) -> bool
            {
                last_valid_count = 0;
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
                        ++last_valid_count;
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

                const int tail_dims =
                    std::min(dim, std::max(0, options.conservative_tail_dims));
                if (tail_dims > 0 && options.min_tail_sigma > 0.0)
                {
                    sigma.tail(tail_dims) =
                        sigma.tail(tail_dims)
                            .cwiseMax(Eigen::VectorXd::Constant(
                                tail_dims, options.min_tail_sigma));
                }
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
            int no_valid_iters = last_valid_count == 0 ? 1 : 0;

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
                Eigen::VectorXd sample_sigma = sigma;
                if (options.exploration_boost_after_invalid_iters > 0 &&
                    no_valid_iters >= options.exploration_boost_after_invalid_iters &&
                    options.invalid_exploration_boost > 1.0)
                {
                    sample_sigma *= options.invalid_exploration_boost;
                }
                for (int i = 1; i < population; ++i)
                {
                    population_vec[i].x = sampleAround(mu, sample_sigma);
                }

                if (!evaluatePopulation(population_vec))
                {
                    break;
                }
                if (last_valid_count == 0)
                {
                    ++no_valid_iters;
                }
                else
                {
                    no_valid_iters = 0;
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
                    const double cost_tolerance =
                        std::max(options.abs_cost_tol,
                                 options.rel_cost_tol * scale);
                    const double sigma_threshold =
                        std::max(options.min_sigma * std::max(1.0, options.sigma_convergence_ratio),
                                 options.min_sigma + 1.0e-12);
                    if (mean_abs_diff <= cost_tolerance &&
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
