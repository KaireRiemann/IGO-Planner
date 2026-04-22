#include "misc/visualizer.hpp"
#include "gcopter/trajectory.hpp"
#include "gcopter/gcopter.hpp"
#include "gcopter/meta_optimize.hpp"
#include "gcopter/firi.hpp"
#include "gcopter/flatness.hpp"
#include "gcopter/voxel_map.hpp"
#include "gcopter/sfc_gen.hpp"

#include <ros/ros.h>
#include <ros/console.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <chrono>
#include <random>
#include <sys/stat.h>
#include <sys/types.h>

struct Config
{
    std::string mapTopic;
    std::string targetTopic;
    double dilateRadius;
    double voxelWidth;
    std::vector<double> mapBound;
    double timeoutRRT;
    double maxVelMag;
    double maxBdrMag;
    double maxTiltAngle;
    double minThrust;
    double maxThrust;
    double vehicleMass;
    double gravAcc;
    double horizDrag;
    double vertDrag;
    double parasDrag;
    double speedEps;
    double weightT;
    std::vector<double> chiVec;
    double smoothingEps;
    int integralIntervs;
    double relCostTol;
    bool runNUBSComparison;
    std::string nubsCompareOutput;
    double diagnosticsSampleDt;
    int nubsMetaPopulation;
    int nubsMetaMaxIterations;
    int nubsMetaMaxEvaluations;
    double nubsMetaMaxWallTime;
    int nubsMetaSeed;
    int nubsMetaMidpoints;
    bool nubsMetaUseTimeProfile;
    double nubsMetaSampleDt;
    double nubsMetaTimeWeight;
    double nubsMetaCollisionWeight;
    double nubsMetaVelocityWeight;
    double nubsMetaAccelerationWeight;
    double nubsMetaJerkWeight;
    double nubsMetaEnergyWeight;
    double nubsMetaTimeLowerBound;
    double nubsMetaTimeUpperBound;
    double nubsMetaSimpleMaxVelocity;
    double nubsMetaSimpleMaxAcceleration;
    double nubsMetaSimpleMaxJerk;
    double nubsMetaTargetVelocityRatio;
    double nubsMetaLocalBoxRadius;
    double nubsMetaTimeFloorScale;
    double nubsMetaInitialTimeScale;
    double nubsMetaTotalSlackMinScale;
    double nubsMetaTotalSlackMaxScale;

    Config(const ros::NodeHandle &nh_priv)
    {
        nh_priv.getParam("MapTopic", mapTopic);
        nh_priv.getParam("TargetTopic", targetTopic);
        nh_priv.getParam("DilateRadius", dilateRadius);
        nh_priv.getParam("VoxelWidth", voxelWidth);
        nh_priv.getParam("MapBound", mapBound);
        nh_priv.getParam("TimeoutRRT", timeoutRRT);
        nh_priv.getParam("MaxVelMag", maxVelMag);
        nh_priv.getParam("MaxBdrMag", maxBdrMag);
        nh_priv.getParam("MaxTiltAngle", maxTiltAngle);
        nh_priv.getParam("MinThrust", minThrust);
        nh_priv.getParam("MaxThrust", maxThrust);
        nh_priv.getParam("VehicleMass", vehicleMass);
        nh_priv.getParam("GravAcc", gravAcc);
        nh_priv.getParam("HorizDrag", horizDrag);
        nh_priv.getParam("VertDrag", vertDrag);
        nh_priv.getParam("ParasDrag", parasDrag);
        nh_priv.getParam("SpeedEps", speedEps);
        nh_priv.getParam("WeightT", weightT);
        nh_priv.getParam("ChiVec", chiVec);
        nh_priv.getParam("SmoothingEps", smoothingEps);
        nh_priv.getParam("IntegralIntervs", integralIntervs);
        nh_priv.getParam("RelCostTol", relCostTol);
        if (!nh_priv.getParam("RunNUBSComparison", runNUBSComparison))
        {
            runNUBSComparison = true;
        }
        if (!nh_priv.getParam("NUBSCompareOutput", nubsCompareOutput))
        {
            nubsCompareOutput = "auto";
        }
        nh_priv.param("DiagnosticsSampleDt", diagnosticsSampleDt, 0.05);
        nh_priv.param("NUBSMetaPopulation", nubsMetaPopulation, 50);
        nh_priv.param("NUBSMetaMaxIterations", nubsMetaMaxIterations, 80);
        nh_priv.param("NUBSMetaMaxEvaluations", nubsMetaMaxEvaluations, 500);
        nh_priv.param("NUBSMetaMaxWallTime", nubsMetaMaxWallTime, 0.0);
        nh_priv.param("NUBSMetaSeed", nubsMetaSeed, 0);
        nh_priv.param("NUBSMetaMidpoints", nubsMetaMidpoints, 4);
        nh_priv.param("NUBSMetaUseTimeProfile", nubsMetaUseTimeProfile, false);
        nh_priv.param("NUBSMetaSampleDt", nubsMetaSampleDt, 0.10);
        nh_priv.param("NUBSMetaTimeWeight", nubsMetaTimeWeight, 4.0);
        nh_priv.param("NUBSMetaCollisionWeight", nubsMetaCollisionWeight, 30.0);
        nh_priv.param("NUBSMetaVelocityWeight", nubsMetaVelocityWeight, 20.0);
        nh_priv.param("NUBSMetaAccelerationWeight", nubsMetaAccelerationWeight, 25.0);
        nh_priv.param("NUBSMetaJerkWeight", nubsMetaJerkWeight, 2.0);
        nh_priv.param("NUBSMetaEnergyWeight", nubsMetaEnergyWeight, 0.02);
        nh_priv.param("NUBSMetaTimeLowerBound", nubsMetaTimeLowerBound, 0.10);
        nh_priv.param("NUBSMetaTimeUpperBound", nubsMetaTimeUpperBound, 8.0);
        nh_priv.param("NUBSMetaSimpleMaxVelocity", nubsMetaSimpleMaxVelocity, maxVelMag);
        nh_priv.param("NUBSMetaSimpleMaxAcceleration", nubsMetaSimpleMaxAcceleration, 15.0);
        nh_priv.param("NUBSMetaSimpleMaxJerk", nubsMetaSimpleMaxJerk, 50.0);
        nh_priv.param("NUBSMetaTargetVelocityRatio", nubsMetaTargetVelocityRatio, 0.90);
        nh_priv.param("NUBSMetaLocalBoxRadius", nubsMetaLocalBoxRadius, 2.0);
        nh_priv.param("NUBSMetaTimeFloorScale", nubsMetaTimeFloorScale, 1.10);
        nh_priv.param("NUBSMetaInitialTimeScale", nubsMetaInitialTimeScale, 1.18);
        nh_priv.param("NUBSMetaTotalSlackMinScale", nubsMetaTotalSlackMinScale, 0.04);
        nh_priv.param("NUBSMetaTotalSlackMaxScale", nubsMetaTotalSlackMaxScale, 0.65);
    }
};

class GlobalPlanner
{
private:
    Config config;

    ros::NodeHandle nh;
    ros::Subscriber mapSub;
    ros::Subscriber targetSub;

    bool mapInitialized;
    voxel_map::VoxelMap voxelMap;
    Visualizer visualizer;
    std::vector<Eigen::Vector3d> startGoal;

    Trajectory<5> traj;
    bsplinetrajectory::NUBSTrajectory<3> nubsTraj;
    double trajStamp;
    int planCounter = 0;
    bool hasNUBSComparisonTraj = false;
    std::string nubsCompareDirectory;

    static inline std::string packageRootFromSourcePath()
    {
        const std::string source_path = __FILE__;
        const std::string suffix = "/src/global_planning.cpp";
        const std::size_t pos = source_path.rfind(suffix);
        if (pos != std::string::npos)
        {
            return source_path.substr(0, pos);
        }
        return ".";
    }

    static inline std::string currentTimestampString()
    {
        const std::time_t now = std::time(nullptr);
        std::tm local_tm;
        localtime_r(&now, &local_tm);

        char buffer[32];
        std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &local_tm);
        return std::string(buffer);
    }

    static inline bool ensureDirectory(const std::string &directory)
    {
        if (directory.empty())
        {
            return false;
        }
        if (::mkdir(directory.c_str(), 0755) == 0)
        {
            return true;
        }
        return errno == EEXIST;
    }

    static inline std::string joinPath(const std::string &lhs,
                                       const std::string &rhs)
    {
        if (lhs.empty())
        {
            return rhs;
        }
        if (lhs.back() == '/')
        {
            return lhs + rhs;
        }
        return lhs + "/" + rhs;
    }

    static inline std::string parentDirectory(const std::string &path)
    {
        const std::size_t pos = path.find_last_of('/');
        if (pos == std::string::npos)
        {
            return ".";
        }
        if (pos == 0)
        {
            return "/";
        }
        return path.substr(0, pos);
    }

    static inline std::string fileStem(const std::string &path)
    {
        const std::size_t slash = path.find_last_of('/');
        const std::size_t begin = slash == std::string::npos ? 0 : slash + 1;
        std::size_t end = path.find_last_of('.');
        if (end == std::string::npos || end < begin)
        {
            end = path.size();
        }
        return path.substr(begin, end - begin);
    }

    static inline bool ensureParentDirectory(const std::string &file_path)
    {
        return ensureDirectory(parentDirectory(file_path));
    }

    static inline std::string makeDefaultNUBSCompareDirectory()
    {
        const std::string root_dir =
            packageRootFromSourcePath() + "/script/nubs_outputs";
        ensureDirectory(root_dir);
        const std::string output_dir = joinPath(root_dir, currentTimestampString());
        ensureDirectory(output_dir);
        return output_dir;
    }

    static inline std::string resolveNUBSCompareDirectory(
        const std::string &configured,
        std::string &record_csv)
    {
        if (configured.empty() || configured == "auto")
        {
            const std::string output_dir = makeDefaultNUBSCompareDirectory();
            record_csv = joinPath(output_dir, "optimization_records.csv");
            return output_dir;
        }
        if (configured.back() == '/')
        {
            const std::string root_dir = configured.substr(0, configured.size() - 1);
            ensureDirectory(root_dir);
            const std::string output_dir = joinPath(root_dir, currentTimestampString());
            ensureDirectory(output_dir);
            record_csv = joinPath(output_dir, "optimization_records.csv");
            return output_dir;
        }
        if (configured.find(".csv") == std::string::npos)
        {
            ensureDirectory(configured);
            const std::string output_dir = joinPath(configured, currentTimestampString());
            ensureDirectory(output_dir);
            record_csv = joinPath(output_dir, "optimization_records.csv");
            return output_dir;
        }

        ensureParentDirectory(configured);
        const std::string output_dir =
            joinPath(parentDirectory(configured), fileStem(configured));
        ensureDirectory(output_dir);
        record_csv = configured;
        return output_dir;
    }

public:
    GlobalPlanner(const Config &conf,
                  ros::NodeHandle &nh_)
        : config(conf),
          nh(nh_),
          mapInitialized(false),
          visualizer(nh)
    {
        nubsCompareDirectory =
            resolveNUBSCompareDirectory(config.nubsCompareOutput,
                                        config.nubsCompareOutput);

        const Eigen::Vector3i xyz((config.mapBound[1] - config.mapBound[0]) / config.voxelWidth,
                                  (config.mapBound[3] - config.mapBound[2]) / config.voxelWidth,
                                  (config.mapBound[5] - config.mapBound[4]) / config.voxelWidth);

        const Eigen::Vector3d offset(config.mapBound[0], config.mapBound[2], config.mapBound[4]);

        voxelMap = voxel_map::VoxelMap(xyz, offset, config.voxelWidth);

        mapSub = nh.subscribe(config.mapTopic, 1, &GlobalPlanner::mapCallBack, this,
                              ros::TransportHints().tcpNoDelay());

        targetSub = nh.subscribe(config.targetTopic, 1, &GlobalPlanner::targetCallBack, this,
                                 ros::TransportHints().tcpNoDelay());

        ROS_INFO_STREAM("NUBS compare output: " << nubsCompareDirectory);
    }

    inline void appendOptimizationRecord(
        const int plan_id,
        const std::string &representation,
        const gcopter::GCOPTER_PolytopeSFC::SolverResult &result) const
    {
        if (config.nubsCompareOutput.empty())
        {
            return;
        }

        std::ifstream existing(config.nubsCompareOutput);
        const bool write_header =
            !existing.good() || existing.peek() == std::ifstream::traits_type::eof();
        std::ofstream out(config.nubsCompareOutput, std::ios::app);
        if (!out.is_open())
        {
            ROS_WARN_STREAM("Failed to open NUBS compare output: "
                            << config.nubsCompareOutput);
            return;
        }

        if (write_header)
        {
            out << "plan_id,representation,success,objective,wall_time_sec,"
                << "evaluation_time_sec,mean_evaluation_time_sec,iterations,"
                << "eval_count,total_duration_sec,trajectory_length,"
                << "mean_generate_time_sec,mean_coeff_grad_time_sec,"
                << "mean_time_grad_time_sec,mean_propagate_time_sec,status\n";
        }

        const double mean_eval_time =
            result.eval_count > 0
                ? result.evaluation_time / static_cast<double>(result.eval_count)
                : 0.0;
        const double inv_eval_count =
            result.eval_count > 0 ? 1.0 / static_cast<double>(result.eval_count) : 0.0;
        out << plan_id << ','
            << representation << ','
            << (result.has_solution ? 1 : 0) << ','
            << std::setprecision(12) << result.objective << ','
            << result.wall_time << ','
            << result.evaluation_time << ','
            << mean_eval_time << ','
            << result.iterations << ','
            << result.eval_count << ','
            << result.total_duration << ','
            << result.trajectory_length << ','
            << result.evaluation_breakdown.generate_time * inv_eval_count << ','
            << result.evaluation_breakdown.coeff_grad_time * inv_eval_count << ','
            << result.evaluation_breakdown.time_grad_time * inv_eval_count << ','
            << result.evaluation_breakdown.propagate_time * inv_eval_count << ','
            << '"' << result.status << '"' << '\n';
    }

    struct TrajectoryDiagnostics
    {
        bool valid = false;
        int sample_count = 0;
        int collision_samples = 0;
        double total_duration = std::numeric_limits<double>::infinity();
        double trajectory_length = 0.0;
        double collision_length = 0.0;
        double max_collision_distance = 0.0;
        double min_obstacle_distance = std::numeric_limits<double>::infinity();
        double max_speed = 0.0;
        double max_acceleration = 0.0;
        double acceleration_energy = 0.0;
        double jerk_energy = 0.0;
        std::vector<Eigen::Vector3d> samples;
    };

    inline std::string getDiagnosticsPath() const
    {
        return nubsCompareDirectory.empty()
                   ? std::string()
                   : joinPath(nubsCompareDirectory, "trajectory_metrics.csv");
    }

    inline std::string getTrajectoryDataPath(const int plan_id) const
    {
        return nubsCompareDirectory.empty()
                   ? std::string()
                   : joinPath(nubsCompareDirectory,
                              "request_" + std::to_string(plan_id) + "_trajectory.csv");
    }

    inline std::string getPlotScriptPath() const
    {
        return joinPath(packageRootFromSourcePath(), "script/plot_compare_results.py");
    }

    inline double estimateVoxelDistance(const Eigen::Vector3d &position,
                                        const bool target_occupied) const
    {
        const Eigen::Vector3i center = voxelMap.posD2I(position);
        const int max_radius = 10;
        double best_distance = std::numeric_limits<double>::infinity();

        for (int dx = -max_radius; dx <= max_radius; ++dx)
        {
            for (int dy = -max_radius; dy <= max_radius; ++dy)
            {
                for (int dz = -max_radius; dz <= max_radius; ++dz)
                {
                    const Eigen::Vector3i query_index =
                        center + Eigen::Vector3i(dx, dy, dz);
                    const bool occupied = voxelMap.query(query_index);
                    if (occupied != target_occupied)
                    {
                        continue;
                    }

                    const Eigen::Vector3d cell_center =
                        voxelMap.posI2D(query_index);
                    best_distance =
                        std::min(best_distance, (cell_center - position).norm());
                }
            }
        }

        if (std::isfinite(best_distance))
        {
            return best_distance;
        }
        return max_radius * voxelMap.getScale();
    }

    inline TrajectoryDiagnostics computeTrajectoryDiagnostics(
        const Trajectory<5> &trajectory,
        const double jerk_energy) const
    {
        TrajectoryDiagnostics diagnostics;
        if (trajectory.getPieceNum() <= 0)
        {
            return diagnostics;
        }

        diagnostics.valid = true;
        diagnostics.total_duration = trajectory.getTotalDuration();
        diagnostics.jerk_energy = jerk_energy;
        const double sample_dt = std::max(0.02, config.diagnosticsSampleDt);
        const int sample_count =
            std::max(2, static_cast<int>(std::ceil(diagnostics.total_duration / sample_dt)) + 1);
        diagnostics.sample_count = sample_count;
        diagnostics.samples.reserve(sample_count);

        Eigen::Vector3d previous_position = trajectory.getPos(0.0);
        diagnostics.samples.push_back(previous_position);

        for (int i = 0; i < sample_count; ++i)
        {
            const double t =
                i == sample_count - 1
                    ? diagnostics.total_duration
                    : std::min(diagnostics.total_duration, i * sample_dt);
            const Eigen::Vector3d position = trajectory.getPos(t);
            const Eigen::Vector3d velocity = trajectory.getVel(t);
            const Eigen::Vector3d acceleration = trajectory.getAcc(t);

            if (i > 0)
            {
                diagnostics.trajectory_length +=
                    (position - previous_position).norm();
            }

            const bool occupied = voxelMap.query(position);
            if (occupied)
            {
                ++diagnostics.collision_samples;
                diagnostics.max_collision_distance =
                    std::max(diagnostics.max_collision_distance,
                             estimateVoxelDistance(position, false));
                if (i > 0)
                {
                    diagnostics.collision_length +=
                        (position - previous_position).norm();
                }
            }
            else
            {
                diagnostics.min_obstacle_distance =
                    std::min(diagnostics.min_obstacle_distance,
                             estimateVoxelDistance(position, true));
            }

            diagnostics.max_speed =
                std::max(diagnostics.max_speed, velocity.norm());
            diagnostics.max_acceleration =
                std::max(diagnostics.max_acceleration, acceleration.norm());
            diagnostics.acceleration_energy +=
                acceleration.squaredNorm() * sample_dt;
            if (i > 0)
            {
                diagnostics.samples.push_back(position);
            }
            previous_position = position;
        }

        if (!std::isfinite(diagnostics.min_obstacle_distance))
        {
            diagnostics.min_obstacle_distance = 0.0;
        }
        return diagnostics;
    }

    inline TrajectoryDiagnostics computeTrajectoryDiagnostics(
        const minco::MINCO_S3NU &jerk_opt) const
    {
        double jerk_energy = std::numeric_limits<double>::infinity();
        jerk_opt.getEnergy(jerk_energy);
        if (!std::isfinite(jerk_energy))
        {
            return TrajectoryDiagnostics();
        }

        Trajectory<5> trajectory;
        jerk_opt.getTrajectory(trajectory);
        return computeTrajectoryDiagnostics(trajectory, jerk_energy);
    }

    inline TrajectoryDiagnostics computeTrajectoryDiagnostics(
        const bsplinetrajectory::NUBSTrajectory<3> &trajectory) const
    {
        TrajectoryDiagnostics diagnostics;
        if (trajectory.getPieceNum() <= 0)
        {
            return diagnostics;
        }

        diagnostics.valid = true;
        diagnostics.total_duration = trajectory.getTotalDuration();
        diagnostics.jerk_energy = trajectory.getEnergy();
        const double sample_dt = std::max(0.02, config.diagnosticsSampleDt);
        const int sample_count =
            std::max(2, static_cast<int>(std::ceil(diagnostics.total_duration / sample_dt)) + 1);
        diagnostics.sample_count = sample_count;
        diagnostics.samples.reserve(sample_count);

        Eigen::Vector3d previous_position = trajectory.evaluate(0.0, 0);
        diagnostics.samples.push_back(previous_position);

        for (int i = 0; i < sample_count; ++i)
        {
            const double t =
                i == sample_count - 1
                    ? diagnostics.total_duration
                    : std::min(diagnostics.total_duration, i * sample_dt);
            const Eigen::Vector3d position = trajectory.evaluate(t, 0);
            const Eigen::Vector3d velocity = trajectory.evaluate(t, 1);
            const Eigen::Vector3d acceleration = trajectory.evaluate(t, 2);

            if (i > 0)
            {
                diagnostics.trajectory_length +=
                    (position - previous_position).norm();
            }

            const bool occupied = voxelMap.query(position);
            if (occupied)
            {
                ++diagnostics.collision_samples;
                diagnostics.max_collision_distance =
                    std::max(diagnostics.max_collision_distance,
                             estimateVoxelDistance(position, false));
                if (i > 0)
                {
                    diagnostics.collision_length +=
                        (position - previous_position).norm();
                }
            }
            else
            {
                diagnostics.min_obstacle_distance =
                    std::min(diagnostics.min_obstacle_distance,
                             estimateVoxelDistance(position, true));
            }

            diagnostics.max_speed =
                std::max(diagnostics.max_speed, velocity.norm());
            diagnostics.max_acceleration =
                std::max(diagnostics.max_acceleration, acceleration.norm());
            diagnostics.acceleration_energy +=
                acceleration.squaredNorm() * sample_dt;
            if (i > 0)
            {
                diagnostics.samples.push_back(position);
            }
            previous_position = position;
        }

        if (!std::isfinite(diagnostics.min_obstacle_distance))
        {
            diagnostics.min_obstacle_distance = 0.0;
        }
        return diagnostics;
    }

    inline void appendTrajectoryMetrics(
        const int plan_id,
        const std::string &solver_name,
        const gcopter::GCOPTER_PolytopeSFC::SolverResult &result,
        const TrajectoryDiagnostics &diagnostics) const
    {
        const std::string path = getDiagnosticsPath();
        if (path.empty())
        {
            return;
        }

        std::ifstream existing(path);
        const bool write_header =
            !existing.good() ||
            existing.peek() == std::ifstream::traits_type::eof();
        std::ofstream out(path, std::ios::app);
        if (!out.is_open())
        {
            ROS_WARN_STREAM("Failed to open trajectory metrics output: " << path);
            return;
        }

        if (write_header)
        {
            out << "request_id,solver,seed,success,objective,wall_time_sec,"
                << "optimization_time_sec,frontend_time_sec,path_search_time_sec,"
                << "corridor_generation_time_sec,iterations,eval_count,total_duration_sec,"
                << "trajectory_length,max_violation,collision_length,max_collision_distance,"
                << "collision_samples,max_speed,max_acceleration,acceleration_energy,jerk_energy,"
                << "max_corridor_violation,max_velocity_violation,max_acceleration_violation,"
                << "max_body_rate_violation,max_tilt_violation,max_thrust_violation,"
                << "min_obstacle_distance,status\n";
        }

        const double collision_violation =
            diagnostics.valid ? diagnostics.collision_length : 0.0;
        const double velocity_violation =
            diagnostics.valid
                ? std::max(0.0, diagnostics.max_speed - config.maxVelMag)
                : 0.0;
        const double acceleration_violation =
            diagnostics.valid
                ? std::max(0.0, diagnostics.max_acceleration -
                                    config.nubsMetaSimpleMaxAcceleration)
                : 0.0;
        const double max_violation =
            std::max(std::max(result.violations.maxViolation(), collision_violation),
                     std::max(velocity_violation, acceleration_violation));
        const bool success =
            result.has_solution && diagnostics.valid &&
            diagnostics.collision_samples == 0 &&
            max_violation <= 1.0e-4;

        out << plan_id << ','
            << solver_name << ','
            << -1 << ','
            << (success ? 1 : 0) << ','
            << std::setprecision(12) << result.objective << ','
            << result.wall_time << ','
            << result.wall_time << ','
            << 0.0 << ','
            << 0.0 << ','
            << 0.0 << ','
            << result.iterations << ','
            << result.eval_count << ','
            << diagnostics.total_duration << ','
            << diagnostics.trajectory_length << ','
            << max_violation << ','
            << diagnostics.collision_length << ','
            << diagnostics.max_collision_distance << ','
            << diagnostics.collision_samples << ','
            << diagnostics.max_speed << ','
            << diagnostics.max_acceleration << ','
            << diagnostics.acceleration_energy << ','
            << diagnostics.jerk_energy << ','
            << result.violations.max_corridor_violation << ','
            << std::max(result.violations.max_velocity_violation,
                        velocity_violation) << ','
            << std::max(result.violations.max_acceleration_violation,
                        acceleration_violation) << ','
            << result.violations.max_body_rate_violation << ','
            << result.violations.max_tilt_violation << ','
            << result.violations.max_thrust_violation << ','
            << diagnostics.min_obstacle_distance << ','
            << '"' << result.status << '"' << '\n';
    }

    inline void writeTrajectorySamples(
        const int plan_id,
        const std::vector<Eigen::Vector3d> &route,
        const std::vector<std::pair<std::string, TrajectoryDiagnostics>> &series) const
    {
        const std::string path = getTrajectoryDataPath(plan_id);
        if (path.empty())
        {
            return;
        }

        std::ofstream csv(path.c_str(), std::ios::out | std::ios::trunc);
        if (!csv.is_open())
        {
            ROS_WARN_STREAM("Failed to open trajectory sample output: " << path);
            return;
        }

        csv << "request_id,series,sample_index,x,y,z\n";
        auto writeSeries =
            [&](const std::string &name,
                const std::vector<Eigen::Vector3d> &samples) -> void
        {
            for (std::size_t i = 0; i < samples.size(); ++i)
            {
                csv << plan_id << ','
                    << name << ','
                    << i << ','
                    << samples[i].x() << ','
                    << samples[i].y() << ','
                    << samples[i].z() << '\n';
            }
        };

        writeSeries("ROUTE", route);
        for (const auto &item : series)
        {
            writeSeries(item.first, item.second.samples);
        }
    }

    inline void refreshPlotArtifacts(const int plan_id) const
    {
        if (nubsCompareDirectory.empty())
        {
            return;
        }

        const std::string script_path = getPlotScriptPath();
        std::ifstream script(script_path);
        if (!script.good())
        {
            ROS_WARN_STREAM("Plot script not found: " << script_path);
            return;
        }

        std::ostringstream cmd;
        cmd << "python3 "
            << "\"" << script_path << "\""
            << " --output-dir "
            << "\"" << nubsCompareDirectory << "\""
            << " --request-id "
            << plan_id
            << " >/dev/null 2>&1";
        const int ret = std::system(cmd.str().c_str());
        if (ret != 0)
        {
            ROS_WARN_STREAM("Failed to refresh NUBS comparison plots with command: "
                            << cmd.str());
        }
    }

    inline gcopter::GCOPTER_PolytopeSFC::IGOSolveOptions
    makeNUBSMetaOptions() const
    {
        gcopter::GCOPTER_PolytopeSFC::IGOSolveOptions options;
        options.population = std::max(2, config.nubsMetaPopulation);
        options.max_iterations = std::max(1, config.nubsMetaMaxIterations);
        options.max_evaluations = config.nubsMetaMaxEvaluations;
        options.max_wall_time = config.nubsMetaMaxWallTime;
        options.seed = static_cast<unsigned int>(std::max(0, config.nubsMetaSeed));
        options.meta_optimizer_use_nubs_direct = true;
        options.meta_optimizer_use_time_profile = config.nubsMetaUseTimeProfile;
        options.meta_optimizer_midpoints = std::max(0, config.nubsMetaMidpoints);
        options.meta_optimizer_sample_dt = config.nubsMetaSampleDt;
        options.meta_optimizer_time_weight = config.nubsMetaTimeWeight;
        options.meta_optimizer_energy_weight = config.nubsMetaEnergyWeight;
        options.meta_optimizer_collision_weight = config.nubsMetaCollisionWeight;
        options.meta_optimizer_velocity_weight = config.nubsMetaVelocityWeight;
        options.meta_optimizer_acceleration_weight = config.nubsMetaAccelerationWeight;
        options.meta_optimizer_jerk_weight = config.nubsMetaJerkWeight;
        options.meta_optimizer_time_lb = config.nubsMetaTimeLowerBound;
        options.meta_optimizer_time_ub = config.nubsMetaTimeUpperBound;
        options.meta_optimizer_simple_max_velocity = config.nubsMetaSimpleMaxVelocity;
        options.meta_optimizer_simple_max_acceleration = config.nubsMetaSimpleMaxAcceleration;
        options.meta_optimizer_simple_max_jerk = config.nubsMetaSimpleMaxJerk;
        options.meta_optimizer_target_velocity_ratio = config.nubsMetaTargetVelocityRatio;
        options.meta_optimizer_local_box_radius = config.nubsMetaLocalBoxRadius;
        options.meta_time_floor_scale = config.nubsMetaTimeFloorScale;
        options.meta_initial_time_scale = config.nubsMetaInitialTimeScale;
        options.meta_total_slack_min_scale = config.nubsMetaTotalSlackMinScale;
        options.meta_total_slack_max_scale = config.nubsMetaTotalSlackMaxScale;
        options.meta_optimizer_use_collision_length_cost = true;
        options.meta_optimizer_use_control_point_objective = true;
        options.meta_optimizer_has_workspace_bounds = true;
        options.meta_optimizer_workspace_min =
            Eigen::Vector3d(config.mapBound[0], config.mapBound[2], config.mapBound[4]);
        options.meta_optimizer_workspace_max =
            Eigen::Vector3d(config.mapBound[1], config.mapBound[3], config.mapBound[5]);
        options.meta_optimizer_collision_checker =
            [this](const Eigen::Vector3d &position) -> bool
        {
            return voxelMap.query(position);
        };
        return options;
    }

    inline void mapCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        if (!mapInitialized)
        {
            size_t cur = 0;
            const size_t total = msg->data.size() / msg->point_step;
            float *fdata = (float *)(&msg->data[0]);
            for (size_t i = 0; i < total; i++)
            {
                cur = msg->point_step / sizeof(float) * i;

                if (std::isnan(fdata[cur + 0]) || std::isinf(fdata[cur + 0]) ||
                    std::isnan(fdata[cur + 1]) || std::isinf(fdata[cur + 1]) ||
                    std::isnan(fdata[cur + 2]) || std::isinf(fdata[cur + 2]))
                {
                    continue;
                }
                voxelMap.setOccupied(Eigen::Vector3d(fdata[cur + 0],
                                                     fdata[cur + 1],
                                                     fdata[cur + 2]));
            }

            voxelMap.dilate(std::ceil(config.dilateRadius / voxelMap.getScale()));

            mapInitialized = true;
        }
    }

    inline void plan()
    {
        if (startGoal.size() == 2)
        {
            std::vector<Eigen::Vector3d> route;
            sfc_gen::planPath<voxel_map::VoxelMap>(startGoal[0],
                                                   startGoal[1],
                                                   voxelMap.getOrigin(),
                                                   voxelMap.getCorner(),
                                                   &voxelMap, 0.01,
                                                   route);
            std::vector<Eigen::MatrixX4d> hPolys;
            std::vector<Eigen::Vector3d> pc;
            voxelMap.getSurf(pc);

            sfc_gen::convexCover(route,
                                 pc,
                                 voxelMap.getOrigin(),
                                 voxelMap.getCorner(),
                                 7.0,
                                 3.0,
                                 hPolys);
            sfc_gen::shortCut(hPolys);

            if (route.size() > 1)
            {
                visualizer.visualizePolytope(hPolys);

                Eigen::Matrix3d iniState;
                Eigen::Matrix3d finState;
                iniState << route.front(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
                finState << route.back(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

                gcopter::GCOPTER_PolytopeSFC gcopter;

                // magnitudeBounds = [v_max, omg_max, theta_max, thrust_min, thrust_max]^T
                // penaltyWeights = [pos_weight, vel_weight, omg_weight, theta_weight, thrust_weight]^T
                // physicalParams = [vehicle_mass, gravitational_acceleration, horitonral_drag_coeff,
                //                   vertical_drag_coeff, parasitic_drag_coeff, speed_smooth_factor]^T
                // initialize some constraint parameters
                Eigen::VectorXd magnitudeBounds(5);
                Eigen::VectorXd penaltyWeights(5);
                Eigen::VectorXd physicalParams(6);
                magnitudeBounds(0) = config.maxVelMag;
                magnitudeBounds(1) = config.maxBdrMag;
                magnitudeBounds(2) = config.maxTiltAngle;
                magnitudeBounds(3) = config.minThrust;
                magnitudeBounds(4) = config.maxThrust;
                penaltyWeights(0) = (config.chiVec)[0];
                penaltyWeights(1) = (config.chiVec)[1];
                penaltyWeights(2) = (config.chiVec)[2];
                penaltyWeights(3) = (config.chiVec)[3];
                penaltyWeights(4) = (config.chiVec)[4];
                physicalParams(0) = config.vehicleMass;
                physicalParams(1) = config.gravAcc;
                physicalParams(2) = config.horizDrag;
                physicalParams(3) = config.vertDrag;
                physicalParams(4) = config.parasDrag;
                physicalParams(5) = config.speedEps;
                const int quadratureRes = config.integralIntervs;

                traj.clear();

                if (!gcopter.setup(config.weightT,
                                   iniState, finState,
                                   hPolys, INFINITY,
                                   config.smoothingEps,
                                   quadratureRes,
                                   magnitudeBounds,
                                   penaltyWeights,
                                   physicalParams))
                {
                    return;
                }

                const int current_plan_id = planCounter++;
                const Eigen::VectorXd initialGuess = gcopter.getCommonInitialGuess();
                const gcopter::GCOPTER_PolytopeSFC::SolverResult mincoResult =
                    gcopter.solveLBFGSOriginal(initialGuess, config.relCostTol);
                appendOptimizationRecord(current_plan_id, "MINCO_LBFGS", mincoResult);

                if (!mincoResult.has_solution)
                {
                    return;
                }

                minco::MINCO_S3NU mincoJerkOpt;
                if (!gcopter.buildJerkOpt(mincoResult.best_x, mincoJerkOpt))
                {
                    return;
                }
                mincoJerkOpt.getTrajectory(traj);
                const TrajectoryDiagnostics mincoDiagnostics =
                    computeTrajectoryDiagnostics(mincoJerkOpt);
                appendTrajectoryMetrics(current_plan_id,
                                        "MINCO_LBFGS",
                                        mincoResult,
                                        mincoDiagnostics);

                hasNUBSComparisonTraj = false;
                std::vector<std::pair<std::string, TrajectoryDiagnostics>> plotSeries;
                plotSeries.push_back(std::make_pair("MINCO_LBFGS", mincoDiagnostics));
                if (config.runNUBSComparison)
                {
                    const gcopter::GCOPTER_PolytopeSFC::SolverResult nubsResult =
                        gcopter.solveNUBSLBFGSOriginal(initialGuess, config.relCostTol);
                    appendOptimizationRecord(current_plan_id, "NUBS_ANALYTIC", nubsResult);
                    bsplinetrajectory::NUBSTrajectory<3> nubsAnalyticTraj;
                    TrajectoryDiagnostics nubsDiagnostics;
                    if (nubsResult.has_solution &&
                        gcopter.buildNUBSTrajectory(nubsResult.best_x, nubsAnalyticTraj))
                    {
                        nubsDiagnostics =
                            computeTrajectoryDiagnostics(nubsAnalyticTraj);
                        nubsTraj = nubsAnalyticTraj;
                        hasNUBSComparisonTraj = true;
                    }
                    appendTrajectoryMetrics(current_plan_id,
                                            "NUBS_ANALYTIC",
                                            nubsResult,
                                            nubsDiagnostics);
                    plotSeries.push_back(std::make_pair("NUBS_ANALYTIC",
                                                        nubsDiagnostics));

                    const gcopter::GCOPTER_PolytopeSFC::SolverResult nubsFiniteDiffResult =
                        gcopter.solveNUBSFiniteDiffLBFGSOriginal(initialGuess, config.relCostTol);
                    appendOptimizationRecord(current_plan_id,
                                             "NUBS_CENTER_DIFF",
                                             nubsFiniteDiffResult);
                    bsplinetrajectory::NUBSTrajectory<3> nubsFiniteDiffTraj;
                    TrajectoryDiagnostics nubsFiniteDiffDiagnostics;
                    if (nubsFiniteDiffResult.has_solution &&
                        gcopter.buildNUBSTrajectory(nubsFiniteDiffResult.best_x,
                                                    nubsFiniteDiffTraj))
                    {
                        nubsFiniteDiffDiagnostics =
                            computeTrajectoryDiagnostics(nubsFiniteDiffTraj);
                        nubsTraj = nubsFiniteDiffTraj;
                        hasNUBSComparisonTraj = true;
                    }
                    appendTrajectoryMetrics(current_plan_id,
                                            "NUBS_CENTER_DIFF",
                                            nubsFiniteDiffResult,
                                            nubsFiniteDiffDiagnostics);
                    plotSeries.push_back(std::make_pair("NUBS_CENTER_DIFF",
                                                        nubsFiniteDiffDiagnostics));

                    gcopter::GCOPTER_PolytopeSFC::IGOSolveOptions nubsMetaOptions =
                        makeNUBSMetaOptions();
                    const gcopter::GCOPTER_PolytopeSFC::SolverResult nubsMetaResult =
                        gcopter.solveMetaOptimizer(initialGuess, nubsMetaOptions);
                    appendOptimizationRecord(current_plan_id,
                                             "NUBS_META",
                                             nubsMetaResult);
                    bsplinetrajectory::NUBSTrajectory<3> nubsMetaTraj;
                    TrajectoryDiagnostics nubsMetaDiagnostics;
                    if (nubsMetaResult.has_solution &&
                        gcopter.buildMetaDirectNUBSTrajectory(nubsMetaResult,
                                                              nubsMetaTraj))
                    {
                        nubsMetaDiagnostics =
                            computeTrajectoryDiagnostics(nubsMetaTraj);
                        nubsTraj = nubsMetaTraj;
                        hasNUBSComparisonTraj = true;
                    }
                    appendTrajectoryMetrics(current_plan_id,
                                            "NUBS_META",
                                            nubsMetaResult,
                                            nubsMetaDiagnostics);
                    plotSeries.push_back(std::make_pair("NUBS_META",
                                                        nubsMetaDiagnostics));

                    writeTrajectorySamples(current_plan_id, route, plotSeries);
                    refreshPlotArtifacts(current_plan_id);

                    if (nubsResult.has_solution &&
                        nubsFiniteDiffResult.has_solution)
                    {
                        const double nubsMeanEval =
                            nubsResult.eval_count > 0
                                ? nubsResult.evaluation_time / nubsResult.eval_count
                                : 0.0;
                        const double nubsInvEval =
                            nubsResult.eval_count > 0
                                ? 1.0 / static_cast<double>(nubsResult.eval_count)
                                : 0.0;
                        const double nubsFiniteDiffMeanEval =
                            nubsFiniteDiffResult.eval_count > 0
                                ? nubsFiniteDiffResult.evaluation_time /
                                      nubsFiniteDiffResult.eval_count
                                : 0.0;
                        const double nubsFiniteDiffInvEval =
                            nubsFiniteDiffResult.eval_count > 0
                                ? 1.0 / static_cast<double>(nubsFiniteDiffResult.eval_count)
                                : 0.0;
                        const double nubsMetaMeanEval =
                            nubsMetaResult.eval_count > 0
                                ? nubsMetaResult.wall_time / nubsMetaResult.eval_count
                                : 0.0;
                        ROS_INFO_STREAM("Trajectory representation LBFGS compare: "
                                        << "MINCO-LBFGS eval=" << mincoResult.eval_count
                                        << ", iter=" << mincoResult.iterations
                                        << ", wall_ms=" << mincoResult.wall_time * 1000.0
                                        << "; NUBS analytic eval=" << nubsResult.eval_count
                                        << ", iter=" << nubsResult.iterations
                                        << ", mean_eval_ms=" << nubsMeanEval * 1000.0
                                        << " [generate="
                                        << nubsResult.evaluation_breakdown.generate_time * nubsInvEval * 1000.0
                                        << ", coeffGrad="
                                        << nubsResult.evaluation_breakdown.coeff_grad_time * nubsInvEval * 1000.0
                                        << ", timeGrad="
                                        << nubsResult.evaluation_breakdown.time_grad_time * nubsInvEval * 1000.0
                                        << ", propagate="
                                        << nubsResult.evaluation_breakdown.propagate_time * nubsInvEval * 1000.0
                                        << "]; NUBS-FD eval=" << nubsFiniteDiffResult.eval_count
                                        << ", iter=" << nubsFiniteDiffResult.iterations
                                        << ", mean_eval_ms=" << nubsFiniteDiffMeanEval * 1000.0
                                        << " [generate="
                                        << nubsFiniteDiffResult.evaluation_breakdown.generate_time * nubsFiniteDiffInvEval * 1000.0
                                        << ", coeffGrad="
                                        << nubsFiniteDiffResult.evaluation_breakdown.coeff_grad_time * nubsFiniteDiffInvEval * 1000.0
                                        << ", timeGrad="
                                        << nubsFiniteDiffResult.evaluation_breakdown.time_grad_time * nubsFiniteDiffInvEval * 1000.0
                                        << ", propagate="
                                        << nubsFiniteDiffResult.evaluation_breakdown.propagate_time * nubsFiniteDiffInvEval * 1000.0
                                        << "]; NUBS-META eval=" << nubsMetaResult.eval_count
                                        << ", iter=" << nubsMetaResult.iterations
                                        << ", mean_eval_ms=" << nubsMetaMeanEval * 1000.0);
                    }
                }
                else
                {
                    writeTrajectorySamples(current_plan_id, route, plotSeries);
                    refreshPlotArtifacts(current_plan_id);
                }

                if (traj.getPieceNum() > 0)
                {
                    trajStamp = ros::Time::now().toSec();
                    visualizer.visualize(traj, route);
                    if (hasNUBSComparisonTraj)
                    {
                        visualizer.visualizeTrajectory(nubsTraj);
                    }
                }
            }
        }
    }

    inline void targetCallBack(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        if (mapInitialized)
        {
            if (startGoal.size() >= 2)
            {
                startGoal.clear();
            }
            const double zGoal = config.mapBound[4] + config.dilateRadius +
                                 fabs(msg->pose.orientation.z) *
                                     (config.mapBound[5] - config.mapBound[4] - 2 * config.dilateRadius);
            const Eigen::Vector3d goal(msg->pose.position.x, msg->pose.position.y, zGoal);
            if (voxelMap.query(goal) == 0)
            {
                visualizer.visualizeStartGoal(goal, 0.5, startGoal.size());
                startGoal.emplace_back(goal);
            }
            else
            {
                ROS_WARN("Infeasible Position Selected !!!\n");
            }

            plan();
        }
        return;
    }

    inline void process()
    {
        Eigen::VectorXd physicalParams(6);
        physicalParams(0) = config.vehicleMass;
        physicalParams(1) = config.gravAcc;
        physicalParams(2) = config.horizDrag;
        physicalParams(3) = config.vertDrag;
        physicalParams(4) = config.parasDrag;
        physicalParams(5) = config.speedEps;

        flatness::FlatnessMap flatmap;
        flatmap.reset(physicalParams(0), physicalParams(1), physicalParams(2),
                      physicalParams(3), physicalParams(4), physicalParams(5));

        if (traj.getPieceNum() > 0)
        {
            const double delta = ros::Time::now().toSec() - trajStamp;
            if (delta > 0.0 && delta < traj.getTotalDuration())
            {
                double thr;
                Eigen::Vector4d quat;
                Eigen::Vector3d omg;

                flatmap.forward(traj.getVel(delta),
                                traj.getAcc(delta),
                                traj.getJer(delta),
                                0.0, 0.0,
                                thr, quat, omg);
                double speed = traj.getVel(delta).norm();
                double bodyratemag = omg.norm();
                double tiltangle = acos(1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2)));
                std_msgs::Float64 speedMsg, thrMsg, tiltMsg, bdrMsg;
                speedMsg.data = speed;
                thrMsg.data = thr;
                tiltMsg.data = tiltangle;
                bdrMsg.data = bodyratemag;
                visualizer.speedPub.publish(speedMsg);
                visualizer.thrPub.publish(thrMsg);
                visualizer.tiltPub.publish(tiltMsg);
                visualizer.bdrPub.publish(bdrMsg);

                visualizer.visualizeSphere(traj.getPos(delta),
                                           config.dilateRadius);
                if (hasNUBSComparisonTraj &&
                    nubsTraj.getPieceNum() > 0 &&
                    delta < nubsTraj.getTotalDuration())
                {
                    visualizer.visualizeNUBSSphere(nubsTraj.evaluate(delta, 0),
                                                   config.dilateRadius);
                }
            }
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "global_planning_node");
    ros::NodeHandle nh_;

    GlobalPlanner global_planner(Config(ros::NodeHandle("~")), nh_);

    ros::Rate lr(1000);
    while (ros::ok())
    {
        global_planner.process();
        ros::spinOnce();
        lr.sleep();
    }

    return 0;
}
