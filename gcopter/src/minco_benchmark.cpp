#include "gcopter/gcopter.hpp"
#include "gcopter/igo_optimize.hpp"
#include "gcopter/meta_optimize.hpp"
#include "gcopter/sfc_gen.hpp"
#include "gcopter/voxel_map.hpp"

#include <yaml-cpp/yaml.h>

#include <Eigen/Eigen>

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace
{
    struct BenchmarkConfig
    {
        std::string mode = "all";
        std::string output_dir = "src/GCOPTER/gcopter/benchmark_runs/default";

        int num_instances = 50;
        int generator_seed = 20260417;
        int max_map_attempts_per_instance = 50;
        int start_goal_attempts_per_map = 50;
        double min_start_goal_distance = 8.0;

        std::vector<double> map_bound = {-25.0, 25.0, -25.0, 25.0, 0.0, 5.0};
        double voxel_width = 0.25;
        double dilate_radius = 0.5;
        int obstacle_number = 10;
        double obstacle_width_min = 0.6;
        double obstacle_width_max = 1.5;

        double time_weight = 20.0;
        double length_per_piece = 1.0e18;
        double smoothing_eps = 1.0e-2;
        int integral_intervals = 16;
        Eigen::VectorXd magnitude_bounds;
        Eigen::VectorXd penalty_weights;
        Eigen::VectorXd physical_params;

        double feasibility_tolerance = 1.0e-4;

        bool run_lbfgs = true;
        bool run_igo = true;
        bool run_meta = true;
        bool run_meta_pt = true;

        gcopter::GCOPTER_PolytopeSFC::LBFGSSolveOptions lbfgs_options;
        gcopter::GCOPTER_PolytopeSFC::IGOSolveOptions igo_options;
        gcopter::GCOPTER_PolytopeSFC::IGOSolveOptions meta_options;
        std::vector<unsigned int> igo_seeds;
        std::vector<unsigned int> meta_seeds;

        BenchmarkConfig()
        {
            magnitude_bounds.resize(5);
            magnitude_bounds << 4.0, 2.1, 1.05, 2.0, 12.0;

            penalty_weights.resize(5);
            penalty_weights << 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e5;

            physical_params.resize(6);
            physical_params << 0.61, 9.8, 0.70, 0.80, 0.01, 1.0e-4;

            lbfgs_options.rel_cost_tol = 1.0e-5;
            lbfgs_options.max_iterations = 200;
            lbfgs_options.max_evaluations = 500;
            lbfgs_options.max_wall_time = 0.5;

            igo_options.population = 32;
            igo_options.max_iterations = 200;
            igo_options.max_evaluations = 500;
            igo_options.max_wall_time = 0.5;
            igo_options.tau_box_radius = 2.5;
            igo_options.xi_box_bound = 1.5;

            meta_options = igo_options;
            meta_options.population = 50;
            meta_options.max_iterations = 80;
            meta_options.max_evaluations = 500;
            meta_options.max_wall_time = 0.0;
            meta_options.meta_optimizer_midpoints = 4;
            meta_options.meta_optimizer_sample_dt = 0.10;
            meta_options.meta_optimizer_collision_weight = 30.0;
            meta_options.meta_optimizer_velocity_weight = 20.0;
            meta_options.meta_optimizer_acceleration_weight = 25.0;
            meta_options.meta_optimizer_jerk_weight = 2.0;
            meta_options.meta_optimizer_energy_weight = 0.02;
            meta_options.meta_optimizer_time_weight = 4.0;
            meta_options.meta_optimizer_simple_max_velocity = 4.0;
            meta_options.meta_optimizer_simple_max_acceleration = 15.0;
            meta_options.meta_optimizer_simple_max_jerk = 50.0;
            meta_options.meta_optimizer_target_velocity_ratio = 0.90;
            meta_options.meta_optimizer_time_lb = 0.10;
            meta_options.meta_optimizer_time_ub = 8.0;
            meta_options.meta_optimizer_local_box_radius = 2.0;
            meta_options.meta_time_floor_scale = 1.10;
            meta_options.meta_initial_time_scale = 1.18;
            meta_options.meta_total_slack_min_scale = 0.04;
            meta_options.meta_total_slack_max_scale = 0.65;
            meta_options.meta_optimizer_use_time_profile = true;

            igo_seeds = {0, 1, 2, 3, 4};
            meta_seeds = {0, 1};
        }
    };

    struct BenchmarkInstance
    {
        int instance_id = -1;
        int generator_seed = 0;
        std::string generator_name;

        Eigen::Matrix3d initial_pva;
        Eigen::Matrix3d terminal_pva;
        gcopter::GCOPTER_PolytopeSFC::PolyhedraH corridor;
        std::vector<Eigen::Vector3d> route;

        double time_weight = 0.0;
        double length_per_piece = 0.0;
        double smoothing_eps = 0.0;
        int integral_intervals = 0;
        Eigen::VectorXd magnitude_bounds;
        Eigen::VectorXd penalty_weights;
        Eigen::VectorXd physical_params;
        Eigen::VectorXd common_initial_guess;
    };

    static std::string csvEscape(const std::string &input)
    {
        if (input.find(',') == std::string::npos && input.find('"') == std::string::npos)
        {
            return input;
        }

        std::string escaped = "\"";
        for (char c : input)
        {
            if (c == '"')
            {
                escaped += "\"\"";
            }
            else
            {
                escaped.push_back(c);
            }
        }
        escaped.push_back('"');
        return escaped;
    }

    static bool isAbsolutePath(const std::string &path)
    {
        return !path.empty() && path.front() == '/';
    }

    static std::string joinPath(const std::string &lhs, const std::string &rhs)
    {
        if (lhs.empty())
        {
            return rhs;
        }
        if (rhs.empty())
        {
            return lhs;
        }
        if (lhs.back() == '/')
        {
            return lhs + rhs;
        }
        return lhs + "/" + rhs;
    }

    static std::string getCurrentWorkingDirectory()
    {
        char buffer[4096];
        if (getcwd(buffer, sizeof(buffer)) == nullptr)
        {
            return ".";
        }
        return std::string(buffer);
    }

    static std::string resolvePath(const std::string &base_dir, const std::string &path)
    {
        if (isAbsolutePath(path))
        {
            return path;
        }
        return joinPath(base_dir, path);
    }

    static bool ensureDirectory(const std::string &path)
    {
        if (path.empty())
        {
            return false;
        }

        if (access(path.c_str(), F_OK) == 0)
        {
            return true;
        }

        std::string partial;
        if (path.front() == '/')
        {
            partial = "/";
        }

        std::stringstream ss(path);
        std::string item;
        while (std::getline(ss, item, '/'))
        {
            if (item.empty())
            {
                continue;
            }

            partial = partial == "/" ? partial + item : joinPath(partial, item);
            if (access(partial.c_str(), F_OK) == 0)
            {
                continue;
            }

            if (mkdir(partial.c_str(), 0755) != 0 && errno != EEXIST)
            {
                std::cerr << "Failed to create directory: " << partial << std::endl;
                return false;
            }
        }

        return true;
    }

    static std::vector<double> yamlToDoubleVector(const YAML::Node &node)
    {
        std::vector<double> values;
        if (!node || !node.IsSequence())
        {
            return values;
        }
        for (std::size_t i = 0; i < node.size(); ++i)
        {
            values.push_back(node[i].as<double>());
        }
        return values;
    }

    static Eigen::VectorXd yamlToEigenVector(const YAML::Node &node)
    {
        const std::vector<double> values = yamlToDoubleVector(node);
        Eigen::VectorXd vec(values.size());
        for (int i = 0; i < static_cast<int>(values.size()); ++i)
        {
            vec(i) = values[i];
        }
        return vec;
    }

    static YAML::Node eigenVectorToYaml(const Eigen::VectorXd &vec)
    {
        YAML::Node node;
        for (int i = 0; i < vec.size(); ++i)
        {
            node.push_back(vec(i));
        }
        return node;
    }

    static YAML::Node matrixToYaml(const Eigen::MatrixXd &mat)
    {
        YAML::Node node;
        for (int r = 0; r < mat.rows(); ++r)
        {
            YAML::Node row;
            for (int c = 0; c < mat.cols(); ++c)
            {
                row.push_back(mat(r, c));
            }
            node.push_back(row);
        }
        return node;
    }

    static Eigen::MatrixXd yamlToMatrix(const YAML::Node &node)
    {
        if (!node || !node.IsSequence() || node.size() == 0)
        {
            return Eigen::MatrixXd();
        }

        const int rows = static_cast<int>(node.size());
        const int cols = static_cast<int>(node[0].size());
        Eigen::MatrixXd mat(rows, cols);
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                mat(r, c) = node[r][c].as<double>();
            }
        }
        return mat;
    }

    static YAML::Node vector3ListToYaml(const std::vector<Eigen::Vector3d> &points)
    {
        YAML::Node node;
        for (const Eigen::Vector3d &point : points)
        {
            YAML::Node row;
            row.push_back(point.x());
            row.push_back(point.y());
            row.push_back(point.z());
            node.push_back(row);
        }
        return node;
    }

    static std::vector<Eigen::Vector3d> yamlToVector3List(const YAML::Node &node)
    {
        std::vector<Eigen::Vector3d> points;
        if (!node || !node.IsSequence())
        {
            return points;
        }
        for (std::size_t i = 0; i < node.size(); ++i)
        {
            points.emplace_back(node[i][0].as<double>(),
                                node[i][1].as<double>(),
                                node[i][2].as<double>());
        }
        return points;
    }

    static std::vector<unsigned int> yamlToUIntVector(const YAML::Node &node)
    {
        std::vector<unsigned int> values;
        if (!node || !node.IsSequence())
        {
            return values;
        }
        for (std::size_t i = 0; i < node.size(); ++i)
        {
            values.push_back(node[i].as<unsigned int>());
        }
        return values;
    }

    static BenchmarkConfig loadBenchmarkConfig(const std::string &config_path)
    {
        BenchmarkConfig cfg;
        const YAML::Node root = YAML::LoadFile(config_path);

        const YAML::Node run = root["run"];
        if (run && run["mode"])
        {
            cfg.mode = run["mode"].as<std::string>();
        }
        if (run && run["output_dir"])
        {
            cfg.output_dir = run["output_dir"].as<std::string>();
        }

        const YAML::Node instances = root["instances"];
        if (instances && instances["num_instances"])
        {
            cfg.num_instances = instances["num_instances"].as<int>();
        }
        if (instances && instances["generator_seed"])
        {
            cfg.generator_seed = instances["generator_seed"].as<int>();
        }
        if (instances && instances["max_map_attempts_per_instance"])
        {
            cfg.max_map_attempts_per_instance = instances["max_map_attempts_per_instance"].as<int>();
        }
        if (instances && instances["start_goal_attempts_per_map"])
        {
            cfg.start_goal_attempts_per_map = instances["start_goal_attempts_per_map"].as<int>();
        }
        if (instances && instances["min_start_goal_distance"])
        {
            cfg.min_start_goal_distance = instances["min_start_goal_distance"].as<double>();
        }

        const YAML::Node map = root["map"];
        if (map && map["bound"])
        {
            cfg.map_bound = yamlToDoubleVector(map["bound"]);
        }
        if (map && map["voxel_width"])
        {
            cfg.voxel_width = map["voxel_width"].as<double>();
        }
        if (map && map["dilate_radius"])
        {
            cfg.dilate_radius = map["dilate_radius"].as<double>();
        }

        const YAML::Node generator = root["generator"];
        if (generator && generator["obstacle_number"])
        {
            cfg.obstacle_number = generator["obstacle_number"].as<int>();
        }
        if (generator && generator["obstacle_width_min"])
        {
            cfg.obstacle_width_min = generator["obstacle_width_min"].as<double>();
        }
        if (generator && generator["obstacle_width_max"])
        {
            cfg.obstacle_width_max = generator["obstacle_width_max"].as<double>();
        }

        const YAML::Node planner = root["planner"];
        if (planner && planner["time_weight"])
        {
            cfg.time_weight = planner["time_weight"].as<double>();
        }
        if (planner && planner["length_per_piece"])
        {
            cfg.length_per_piece = planner["length_per_piece"].as<double>();
        }
        if (planner && planner["smoothing_eps"])
        {
            cfg.smoothing_eps = planner["smoothing_eps"].as<double>();
        }
        if (planner && planner["integral_intervals"])
        {
            cfg.integral_intervals = planner["integral_intervals"].as<int>();
        }
        if (planner && planner["magnitude_bounds"])
        {
            cfg.magnitude_bounds = yamlToEigenVector(planner["magnitude_bounds"]);
        }
        if (planner && planner["penalty_weights"])
        {
            cfg.penalty_weights = yamlToEigenVector(planner["penalty_weights"]);
        }
        if (planner && planner["physical_params"])
        {
            cfg.physical_params = yamlToEigenVector(planner["physical_params"]);
        }

        const YAML::Node benchmark = root["benchmark"];
        if (benchmark && benchmark["feasibility_tolerance"])
        {
            cfg.feasibility_tolerance = benchmark["feasibility_tolerance"].as<double>();
        }
        if (benchmark && benchmark["run_lbfgs"])
        {
            cfg.run_lbfgs = benchmark["run_lbfgs"].as<bool>();
        }
        if (benchmark && benchmark["run_igo"])
        {
            cfg.run_igo = benchmark["run_igo"].as<bool>();
        }

        const YAML::Node lbfgs = benchmark["lbfgs"];
        if (lbfgs && lbfgs["rel_cost_tol"])
        {
            cfg.lbfgs_options.rel_cost_tol = lbfgs["rel_cost_tol"].as<double>();
        }
        if (lbfgs && lbfgs["max_iterations"])
        {
            cfg.lbfgs_options.max_iterations = lbfgs["max_iterations"].as<int>();
        }
        if (lbfgs && lbfgs["max_evaluations"])
        {
            cfg.lbfgs_options.max_evaluations = lbfgs["max_evaluations"].as<int>();
        }
        if (lbfgs && lbfgs["max_wall_time"])
        {
            cfg.lbfgs_options.max_wall_time = lbfgs["max_wall_time"].as<double>();
        }

        const YAML::Node igo = benchmark["igo"];
        if (igo && igo["population"])
        {
            cfg.igo_options.population = igo["population"].as<int>();
        }
        if (igo && igo["max_iterations"])
        {
            cfg.igo_options.max_iterations = igo["max_iterations"].as<int>();
        }
        if (igo && igo["max_evaluations"])
        {
            cfg.igo_options.max_evaluations = igo["max_evaluations"].as<int>();
        }
        if (igo && igo["max_wall_time"])
        {
            cfg.igo_options.max_wall_time = igo["max_wall_time"].as<double>();
        }
        if (igo && igo["tau_box_radius"])
        {
            cfg.igo_options.tau_box_radius = igo["tau_box_radius"].as<double>();
        }
        if (igo && igo["xi_box_bound"])
        {
            cfg.igo_options.xi_box_bound = igo["xi_box_bound"].as<double>();
        }
        if (igo && igo["min_sigma"])
        {
            cfg.igo_options.min_sigma = igo["min_sigma"].as<double>();
        }
        if (igo && igo["eta_mean"])
        {
            cfg.igo_options.eta_mean = igo["eta_mean"].as<double>();
        }
        if (igo && igo["eta_sigma"])
        {
            cfg.igo_options.eta_sigma = igo["eta_sigma"].as<double>();
        }
        if (igo && igo["elite_ratio"])
        {
            cfg.igo_options.elite_ratio = igo["elite_ratio"].as<double>();
        }
        if (igo && igo["seeds"])
        {
            cfg.igo_seeds = yamlToUIntVector(igo["seeds"]);
        }

        const YAML::Node meta = benchmark["meta"];
        if (meta && meta["enabled"])
        {
            cfg.run_meta = meta["enabled"].as<bool>();
        }
        if (meta && meta["run_pt"])
        {
            cfg.run_meta_pt = meta["run_pt"].as<bool>();
        }
        if (meta && meta["population"])
        {
            cfg.meta_options.population = meta["population"].as<int>();
        }
        if (meta && meta["max_iterations"])
        {
            cfg.meta_options.max_iterations = meta["max_iterations"].as<int>();
        }
        if (meta && meta["max_evaluations"])
        {
            cfg.meta_options.max_evaluations = meta["max_evaluations"].as<int>();
        }
        if (meta && meta["max_wall_time"])
        {
            cfg.meta_options.max_wall_time = meta["max_wall_time"].as<double>();
        }
        if (meta && meta["min_sigma"])
        {
            cfg.meta_options.min_sigma = meta["min_sigma"].as<double>();
        }
        if (meta && meta["eta_mean"])
        {
            cfg.meta_options.eta_mean = meta["eta_mean"].as<double>();
        }
        if (meta && meta["eta_sigma"])
        {
            cfg.meta_options.eta_sigma = meta["eta_sigma"].as<double>();
        }
        if (meta && meta["elite_ratio"])
        {
            cfg.meta_options.elite_ratio = meta["elite_ratio"].as<double>();
        }
        if (meta && meta["time_weight"])
        {
            cfg.meta_options.meta_optimizer_time_weight = meta["time_weight"].as<double>();
        }
        if (meta && meta["length_weight"])
        {
            cfg.meta_options.meta_optimizer_length_weight = meta["length_weight"].as<double>();
        }
        if (meta && meta["energy_weight"])
        {
            cfg.meta_options.meta_optimizer_energy_weight = meta["energy_weight"].as<double>();
        }
        if (meta && meta["waypoint_smooth_weight"])
        {
            cfg.meta_options.meta_optimizer_waypoint_smooth_weight = meta["waypoint_smooth_weight"].as<double>();
        }
        if (meta && meta["collision_weight"])
        {
            cfg.meta_options.meta_optimizer_collision_weight = meta["collision_weight"].as<double>();
        }
        if (meta && meta["velocity_weight"])
        {
            cfg.meta_options.meta_optimizer_velocity_weight = meta["velocity_weight"].as<double>();
        }
        if (meta && meta["acceleration_weight"])
        {
            cfg.meta_options.meta_optimizer_acceleration_weight = meta["acceleration_weight"].as<double>();
        }
        if (meta && meta["jerk_weight"])
        {
            cfg.meta_options.meta_optimizer_jerk_weight = meta["jerk_weight"].as<double>();
        }
        if (meta && meta["sample_dt"])
        {
            cfg.meta_options.meta_optimizer_sample_dt = meta["sample_dt"].as<double>();
        }
        if (meta && meta["midpoints"])
        {
            cfg.meta_options.meta_optimizer_midpoints = meta["midpoints"].as<int>();
        }
        if (meta && meta["time_lower_bound"])
        {
            cfg.meta_options.meta_optimizer_time_lb = meta["time_lower_bound"].as<double>();
        }
        if (meta && meta["time_upper_bound"])
        {
            cfg.meta_options.meta_optimizer_time_ub = meta["time_upper_bound"].as<double>();
        }
        if (meta && meta["simple_max_velocity"])
        {
            cfg.meta_options.meta_optimizer_simple_max_velocity = meta["simple_max_velocity"].as<double>();
        }
        if (meta && meta["simple_max_acceleration"])
        {
            cfg.meta_options.meta_optimizer_simple_max_acceleration = meta["simple_max_acceleration"].as<double>();
        }
        if (meta && meta["simple_max_jerk"])
        {
            cfg.meta_options.meta_optimizer_simple_max_jerk = meta["simple_max_jerk"].as<double>();
        }
        if (meta && meta["target_velocity_ratio"])
        {
            cfg.meta_options.meta_optimizer_target_velocity_ratio = meta["target_velocity_ratio"].as<double>();
        }
        if (meta && meta["local_box_radius"])
        {
            cfg.meta_options.meta_optimizer_local_box_radius = meta["local_box_radius"].as<double>();
        }
        if (meta && meta["use_collision_length_cost"])
        {
            cfg.meta_options.meta_optimizer_use_collision_length_cost =
                meta["use_collision_length_cost"].as<bool>();
        }
        if (meta && meta["time_floor_scale"])
        {
            cfg.meta_options.meta_time_floor_scale = meta["time_floor_scale"].as<double>();
        }
        if (meta && meta["initial_time_scale"])
        {
            cfg.meta_options.meta_initial_time_scale = meta["initial_time_scale"].as<double>();
        }
        if (meta && meta["total_slack_min_scale"])
        {
            cfg.meta_options.meta_total_slack_min_scale = meta["total_slack_min_scale"].as<double>();
        }
        if (meta && meta["total_slack_max_scale"])
        {
            cfg.meta_options.meta_total_slack_max_scale = meta["total_slack_max_scale"].as<double>();
        }
        if (meta && meta["seeds"])
        {
            cfg.meta_seeds = yamlToUIntVector(meta["seeds"]);
        }

        return cfg;
    }

    static std::vector<Eigen::Vector3d> generateRandomBoxMap(const BenchmarkConfig &cfg,
                                                             const int seed)
    {
        std::default_random_engine eng(seed);

        std::uniform_real_distribution<double> rand_x(cfg.map_bound[0], cfg.map_bound[1]);
        std::uniform_real_distribution<double> rand_y(cfg.map_bound[2], cfg.map_bound[3]);
        std::uniform_real_distribution<double> rand_w(cfg.obstacle_width_min, cfg.obstacle_width_max);
        std::uniform_real_distribution<double> rand_h(0.0, cfg.map_bound[5] - cfg.map_bound[4]);

        std::vector<Eigen::Vector3d> points;
        for (int i = 0; i < cfg.obstacle_number; ++i)
        {
            const double x = rand_x(eng);
            const double y = rand_y(eng);
            const double w = rand_w(eng);
            const double h = rand_h(eng);

            const int wid_num = static_cast<int>(std::ceil(w / cfg.voxel_width));
            const int hei_num = static_cast<int>(std::ceil(h / cfg.voxel_width));

            const int rl = -wid_num / 2;
            const int rh = wid_num / 2;
            const int sl = -wid_num / 2;
            const int sh = wid_num / 2;

            for (int r = rl; r < rh; ++r)
            {
                for (int s = sl; s < sh; ++s)
                {
                    for (int t = 0; t < hei_num; ++t)
                    {
                        if ((r - rl) * (r - rh + 1) *
                                (s - sl) * (s - sh + 1) *
                                t * (t - hei_num + 1) ==
                            0)
                        {
                            points.emplace_back(x + r * cfg.voxel_width,
                                                y + s * cfg.voxel_width,
                                                cfg.map_bound[4] + t * cfg.voxel_width);
                        }
                    }
                }
            }
        }

        return points;
    }

    static voxel_map::VoxelMap buildVoxelMap(const BenchmarkConfig &cfg,
                                             const std::vector<Eigen::Vector3d> &occupied_points)
    {
        const Eigen::Vector3i xyz(
            static_cast<int>((cfg.map_bound[1] - cfg.map_bound[0]) / cfg.voxel_width),
            static_cast<int>((cfg.map_bound[3] - cfg.map_bound[2]) / cfg.voxel_width),
            static_cast<int>((cfg.map_bound[5] - cfg.map_bound[4]) / cfg.voxel_width));
        const Eigen::Vector3d origin(cfg.map_bound[0], cfg.map_bound[2], cfg.map_bound[4]);

        voxel_map::VoxelMap map(xyz, origin, cfg.voxel_width);
        for (const Eigen::Vector3d &point : occupied_points)
        {
            map.setOccupied(point);
        }
        map.dilate(static_cast<int>(std::ceil(cfg.dilate_radius / map.getScale())));
        return map;
    }

    static bool sampleFreePoint(const BenchmarkConfig &cfg,
                                const voxel_map::VoxelMap &map,
                                std::mt19937 &rng,
                                Eigen::Vector3d &point)
    {
        std::uniform_real_distribution<double> x_dist(cfg.map_bound[0] + cfg.dilate_radius,
                                                      cfg.map_bound[1] - cfg.dilate_radius);
        std::uniform_real_distribution<double> y_dist(cfg.map_bound[2] + cfg.dilate_radius,
                                                      cfg.map_bound[3] - cfg.dilate_radius);
        std::uniform_real_distribution<double> z_dist(cfg.map_bound[4] + cfg.dilate_radius,
                                                      cfg.map_bound[5] - cfg.dilate_radius);

        for (int attempt = 0; attempt < 200; ++attempt)
        {
            point = Eigen::Vector3d(x_dist(rng), y_dist(rng), z_dist(rng));
            if (!map.query(point))
            {
                return true;
            }
        }

        return false;
    }

    static bool buildProblemFromMap(const BenchmarkConfig &cfg,
                                    const voxel_map::VoxelMap &map,
                                    std::mt19937 &rng,
                                    BenchmarkInstance &instance)
    {
        for (int attempt = 0; attempt < cfg.start_goal_attempts_per_map; ++attempt)
        {
            Eigen::Vector3d start;
            Eigen::Vector3d goal;
            if (!sampleFreePoint(cfg, map, rng, start) || !sampleFreePoint(cfg, map, rng, goal))
            {
                continue;
            }

            if ((goal - start).norm() < cfg.min_start_goal_distance)
            {
                continue;
            }

            std::vector<Eigen::Vector3d> route;
            sfc_gen::planPath<voxel_map::VoxelMap>(start,
                                                   goal,
                                                   map.getOrigin(),
                                                   map.getCorner(),
                                                   const_cast<voxel_map::VoxelMap *>(&map),
                                                   0.01,
                                                   route);

            if (route.size() <= 1)
            {
                continue;
            }

            std::vector<Eigen::MatrixX4d> h_polys;
            std::vector<Eigen::Vector3d> surface_points;
            map.getSurf(surface_points);
            sfc_gen::convexCover(route,
                                 surface_points,
                                 map.getOrigin(),
                                 map.getCorner(),
                                 7.0,
                                 3.0,
                                 h_polys);
            sfc_gen::shortCut(h_polys);

            if (h_polys.empty())
            {
                continue;
            }

            Eigen::Matrix3d initial_pva;
            Eigen::Matrix3d terminal_pva;
            initial_pva << start, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
            terminal_pva << goal, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

            gcopter::GCOPTER_PolytopeSFC solver;
            if (!solver.setup(cfg.time_weight,
                              initial_pva,
                              terminal_pva,
                              h_polys,
                              cfg.length_per_piece,
                              cfg.smoothing_eps,
                              cfg.integral_intervals,
                              cfg.magnitude_bounds,
                              cfg.penalty_weights,
                              cfg.physical_params))
            {
                continue;
            }

            instance.initial_pva = initial_pva;
            instance.terminal_pva = terminal_pva;
            instance.corridor = h_polys;
            instance.route = route;
            instance.time_weight = cfg.time_weight;
            instance.length_per_piece = cfg.length_per_piece;
            instance.smoothing_eps = cfg.smoothing_eps;
            instance.integral_intervals = cfg.integral_intervals;
            instance.magnitude_bounds = cfg.magnitude_bounds;
            instance.penalty_weights = cfg.penalty_weights;
            instance.physical_params = cfg.physical_params;
            instance.common_initial_guess = solver.getCommonInitialGuess();

            return true;
        }

        return false;
    }

    static bool generateInstance(const BenchmarkConfig &cfg,
                                 const int instance_id,
                                 BenchmarkInstance &instance)
    {
        instance.instance_id = instance_id;
        instance.generator_name = "gcopter_mockamap_random_boxes";

        for (int map_attempt = 0; map_attempt < cfg.max_map_attempts_per_instance; ++map_attempt)
        {
            const int seed = cfg.generator_seed + instance_id * 1000 + map_attempt;
            const std::vector<Eigen::Vector3d> occupied_points = generateRandomBoxMap(cfg, seed);
            const voxel_map::VoxelMap map = buildVoxelMap(cfg, occupied_points);

            std::mt19937 rng(seed + 7919);
            if (buildProblemFromMap(cfg, map, rng, instance))
            {
                instance.generator_seed = seed;
                return true;
            }
        }

        return false;
    }

    static void saveInstanceYaml(const std::string &path,
                                 const BenchmarkInstance &instance)
    {
        YAML::Node root;
        root["instance_id"] = instance.instance_id;
        root["generator"]["name"] = instance.generator_name;
        root["generator"]["seed"] = instance.generator_seed;
        root["boundary_states"]["initial_pva"] = matrixToYaml(instance.initial_pva);
        root["boundary_states"]["terminal_pva"] = matrixToYaml(instance.terminal_pva);
        root["route"] = vector3ListToYaml(instance.route);
        root["corridor"] = YAML::Node(YAML::NodeType::Sequence);
        for (const Eigen::MatrixX4d &poly : instance.corridor)
        {
            root["corridor"].push_back(matrixToYaml(poly));
        }
        root["cost_params"]["time_weight"] = instance.time_weight;
        root["cost_params"]["length_per_piece"] = instance.length_per_piece;
        root["cost_params"]["smoothing_eps"] = instance.smoothing_eps;
        root["cost_params"]["integral_intervals"] = instance.integral_intervals;
        root["dynamics_params"]["magnitude_bounds"] = eigenVectorToYaml(instance.magnitude_bounds);
        root["cost_params"]["penalty_weights"] = eigenVectorToYaml(instance.penalty_weights);
        root["dynamics_params"]["physical_params"] = eigenVectorToYaml(instance.physical_params);
        root["common_initial_guess"] = eigenVectorToYaml(instance.common_initial_guess);

        YAML::Emitter emitter;
        emitter << root;
        std::ofstream out(path.c_str());
        out << emitter.c_str() << std::endl;
    }

    static BenchmarkInstance loadInstanceYaml(const std::string &path)
    {
        BenchmarkInstance instance;
        const YAML::Node root = YAML::LoadFile(path);

        instance.instance_id = root["instance_id"].as<int>();
        instance.generator_name = root["generator"]["name"].as<std::string>();
        instance.generator_seed = root["generator"]["seed"].as<int>();
        instance.initial_pva = yamlToMatrix(root["boundary_states"]["initial_pva"]);
        instance.terminal_pva = yamlToMatrix(root["boundary_states"]["terminal_pva"]);
        instance.route = yamlToVector3List(root["route"]);

        const YAML::Node corridor_node = root["corridor"];
        for (std::size_t i = 0; i < corridor_node.size(); ++i)
        {
            instance.corridor.push_back(yamlToMatrix(corridor_node[i]));
        }

        instance.time_weight = root["cost_params"]["time_weight"].as<double>();
        instance.length_per_piece = root["cost_params"]["length_per_piece"].as<double>();
        instance.smoothing_eps = root["cost_params"]["smoothing_eps"].as<double>();
        instance.integral_intervals = root["cost_params"]["integral_intervals"].as<int>();
        instance.magnitude_bounds = yamlToEigenVector(root["dynamics_params"]["magnitude_bounds"]);
        instance.penalty_weights = yamlToEigenVector(root["cost_params"]["penalty_weights"]);
        instance.physical_params = yamlToEigenVector(root["dynamics_params"]["physical_params"]);
        instance.common_initial_guess = yamlToEigenVector(root["common_initial_guess"]);

        return instance;
    }

    static void writeInstanceManifest(const std::string &path,
                                      const std::vector<BenchmarkInstance> &instances)
    {
        std::ofstream out(path.c_str());
        out << "instance_id,filename,generator_seed\n";
        for (const BenchmarkInstance &instance : instances)
        {
            std::ostringstream name;
            name << "instance_" << std::setw(4) << std::setfill('0') << instance.instance_id << ".yaml";
            out << instance.instance_id << "," << name.str() << "," << instance.generator_seed << "\n";
        }
    }

    static std::vector<BenchmarkInstance> loadInstancesFromManifest(const std::string &instances_dir,
                                                                    const std::string &manifest_path)
    {
        std::vector<BenchmarkInstance> instances;
        std::ifstream in(manifest_path.c_str());
        std::string line;
        bool first = true;
        while (std::getline(in, line))
        {
            if (first)
            {
                first = false;
                continue;
            }
            if (line.empty())
            {
                continue;
            }

            std::stringstream ss(line);
            std::string id_token, file_token, seed_token;
            std::getline(ss, id_token, ',');
            std::getline(ss, file_token, ',');
            std::getline(ss, seed_token, ',');
            (void)id_token;
            (void)seed_token;
            instances.push_back(loadInstanceYaml(joinPath(instances_dir, file_token)));
        }

        return instances;
    }

    static void appendResultRow(std::ofstream &out,
                                const BenchmarkInstance &instance,
                                const std::string &solver_name,
                                const int seed,
                                const gcopter::GCOPTER_PolytopeSFC::SolverResult &result,
                                const double feasibility_tolerance)
    {
        const bool success = result.has_solution &&
                             result.violations.maxViolation() <= feasibility_tolerance;
        out << instance.instance_id << ","
            << solver_name << ","
            << seed << ","
            << (success ? 1 : 0) << ","
            << (result.has_solution ? 1 : 0) << ","
            << (result.converged ? 1 : 0) << ","
            << (result.hit_eval_budget ? 1 : 0) << ","
            << (result.hit_time_budget ? 1 : 0) << ","
            << result.solver_status << ","
            << result.iterations << ","
            << result.eval_count << ","
            << std::setprecision(10) << result.wall_time << ","
            << result.objective << ","
            << result.total_duration << ","
            << result.trajectory_length << ","
            << result.violations.maxViolation() << ","
            << result.violations.max_corridor_violation << ","
            << result.violations.max_velocity_violation << ","
            << result.violations.max_acceleration_violation << ","
            << result.violations.max_body_rate_violation << ","
            << result.violations.max_tilt_violation << ","
            << result.violations.max_thrust_violation << ","
            << result.violations.penalty_cost << ","
            << result.violations.sample_count << ","
            << instance.generator_seed << ","
            << csvEscape(result.status) << "\n";
    }

    static int runBenchmark(const BenchmarkConfig &cfg,
                            const std::string &output_dir,
                            const std::vector<BenchmarkInstance> &instances)
    {
        const std::string results_path = joinPath(output_dir, "raw_results.csv");
        std::ofstream out(results_path.c_str());
        out << "instance_id,solver,seed,success,has_solution,converged,hit_eval_budget,hit_time_budget,"
               "solver_status,iterations,eval_count,wall_time_sec,objective,total_duration_sec,"
               "trajectory_length,max_violation,max_corridor_violation,max_velocity_violation,"
               "max_acceleration_violation,max_body_rate_violation,max_tilt_violation,"
               "max_thrust_violation,penalty_cost,"
               "sample_count,generator_seed,status\n";

        for (const BenchmarkInstance &instance : instances)
        {
            gcopter::GCOPTER_PolytopeSFC solver;
            if (!solver.setup(instance.time_weight,
                              instance.initial_pva,
                              instance.terminal_pva,
                              instance.corridor,
                              instance.length_per_piece,
                              instance.smoothing_eps,
                              instance.integral_intervals,
                              instance.magnitude_bounds,
                              instance.penalty_weights,
                              instance.physical_params))
            {
                std::cerr << "Failed to setup instance " << instance.instance_id << std::endl;
                return 1;
            }

            Eigen::VectorXd x0;
            if (instance.common_initial_guess.size() == solver.getDecisionDim())
            {
                x0 = instance.common_initial_guess;
            }
            else
            {
                x0 = solver.getCommonInitialGuess();
            }

            if (cfg.run_lbfgs)
            {
                const gcopter::GCOPTER_PolytopeSFC::SolverResult lbfgs_result =
                    solver.solveLBFGS(x0, cfg.lbfgs_options);
                appendResultRow(out, instance, "LBFGS", -1, lbfgs_result, cfg.feasibility_tolerance);
            }

            if (cfg.run_igo)
            {
                for (std::size_t i = 0; i < cfg.igo_seeds.size(); ++i)
                {
                    gcopter::GCOPTER_PolytopeSFC::IGOSolveOptions igo_options = cfg.igo_options;
                    igo_options.seed = cfg.igo_seeds[i];
                    const gcopter::GCOPTER_PolytopeSFC::SolverResult igo_result =
                        solver.solveIGO(x0, igo_options);
                    appendResultRow(out, instance, "IGO", static_cast<int>(cfg.igo_seeds[i]),
                                    igo_result, cfg.feasibility_tolerance);
                }
            }

            if (cfg.run_meta)
            {
                for (std::size_t i = 0; i < cfg.meta_seeds.size(); ++i)
                {
                    gcopter::GCOPTER_PolytopeSFC::IGOSolveOptions meta_options = cfg.meta_options;
                    meta_options.seed = cfg.meta_seeds[i];
                    meta_options.meta_optimizer_use_time_profile = true;
                    const gcopter::GCOPTER_PolytopeSFC::SolverResult meta_result =
                        solver.solveMetaOptimizer(x0, meta_options);
                    appendResultRow(out, instance, "META", static_cast<int>(cfg.meta_seeds[i]),
                                    meta_result, cfg.feasibility_tolerance);

                    if (cfg.run_meta_pt)
                    {
                        gcopter::GCOPTER_PolytopeSFC::IGOSolveOptions meta_pt_options = meta_options;
                        meta_pt_options.meta_optimizer_use_time_profile = false;
                        meta_pt_options.meta_optimizer_use_nubs_direct = true;
                        meta_pt_options.meta_optimizer_use_control_point_objective = true;
                        const gcopter::GCOPTER_PolytopeSFC::SolverResult meta_pt_result =
                            solver.solveMetaOptimizer(x0, meta_pt_options);
                        appendResultRow(out, instance, "NUBS_META_PT", static_cast<int>(cfg.meta_seeds[i]),
                                        meta_pt_result, cfg.feasibility_tolerance);
                    }
                }
            }
        }

        return 0;
    }

    static void printUsage(const char *prog)
    {
        std::cerr << "Usage: " << prog << " --config <benchmark_config.yaml>" << std::endl;
    }
}

int main(int argc, char **argv)
{
    std::string config_path;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg == "--config" && i + 1 < argc)
        {
            config_path = argv[++i];
        }
    }

    if (config_path.empty())
    {
        printUsage(argv[0]);
        return 1;
    }

    const BenchmarkConfig cfg = loadBenchmarkConfig(config_path);
    const std::string cwd = getCurrentWorkingDirectory();
    const std::string output_dir = resolvePath(cwd, cfg.output_dir);
    const std::string instances_dir = joinPath(output_dir, "instances");
    const std::string manifest_path = joinPath(output_dir, "instances_manifest.csv");

    if (!ensureDirectory(output_dir) || !ensureDirectory(instances_dir))
    {
        return 1;
    }

    std::vector<BenchmarkInstance> instances;
    if (cfg.mode == "generate" || cfg.mode == "all")
    {
        instances.reserve(cfg.num_instances);
        for (int i = 0; i < cfg.num_instances; ++i)
        {
            BenchmarkInstance instance;
            if (!generateInstance(cfg, i, instance))
            {
                std::cerr << "Failed to generate instance " << i << std::endl;
                return 1;
            }

            std::ostringstream name;
            name << "instance_" << std::setw(4) << std::setfill('0') << i << ".yaml";
            saveInstanceYaml(joinPath(instances_dir, name.str()), instance);
            instances.push_back(instance);
            std::cout << "Generated instance " << i
                      << " with seed " << instance.generator_seed << std::endl;
        }
        writeInstanceManifest(manifest_path, instances);
    }

    if (cfg.mode == "run")
    {
        instances = loadInstancesFromManifest(instances_dir, manifest_path);
        if (instances.empty())
        {
            std::cerr << "No instances found in " << manifest_path << std::endl;
            return 1;
        }
    }

    if (cfg.mode == "run" || cfg.mode == "all")
    {
        return runBenchmark(cfg, output_dir, instances);
    }

    return 0;
}
