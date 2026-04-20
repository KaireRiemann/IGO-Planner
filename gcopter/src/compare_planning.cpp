#include "misc/visualizer.hpp"
#include "gcopter/trajectory.hpp"
#include "gcopter/gcopter.hpp"
#include "gcopter/igo_optimize.hpp"
#include "gcopter/meta_optimize.hpp"
#include "gcopter/voxel_map.hpp"
#include "gcopter/sfc_gen.hpp"

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>

#include <Eigen/Eigen>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
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
    static std::string getPackageDirectory()
    {
        const std::string file_path(__FILE__);
        const std::string suffix("/src/compare_planning.cpp");
        const std::size_t pos = file_path.rfind(suffix);
        if (pos != std::string::npos)
        {
            return file_path.substr(0, pos);
        }
        return std::string(".");
    }

    static std::string joinPath(const std::string &dir,
                                const std::string &filename)
    {
        if (dir.empty())
        {
            return filename;
        }
        if (dir.back() == '/')
        {
            return dir + filename;
        }
        return dir + "/" + filename;
    }

    static std::string getDefaultOutputDirectory()
    {
        return joinPath(getPackageDirectory(), "script/compare_outputs");
    }

    static std::string getCurrentTimestampString()
    {
        const std::time_t now = std::time(nullptr);
        std::tm local_tm;
#if defined(_WIN32)
        localtime_s(&local_tm, &now);
#else
        localtime_r(&now, &local_tm);
#endif

        char buffer[32];
        std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &local_tm);
        return std::string(buffer);
    }

    static std::string getBasename(const std::string &path)
    {
        const std::size_t slash = path.find_last_of('/');
        if (slash == std::string::npos)
        {
            return path;
        }
        return path.substr(slash + 1);
    }

    static std::string getPlotScriptPath()
    {
        return joinPath(getPackageDirectory(), "script/plot_compare_results.py");
    }

    struct Config
    {
        std::string mapTopic;
        std::string targetTopic;
        double dilateRadius;
        double voxelWidth;
        std::vector<double> mapBound;
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
        double lengthPerPiece;

        double feasibilityTol;
        double igoFeasibilityTol;
        std::string recordPath;

        double lbfgsRelCostTol;
        int lbfgsMaxIterations;
        int lbfgsMaxEvaluations;
        double lbfgsMaxWallTime;

        int igoPopulation;
        int igoMaxIterations;
        int igoMaxEvaluations;
        double igoMaxWallTime;
        double igoTauBoxRadius;
        double igoXiBoxBound;
        bool runMetaOptimizer;
        bool runIGOXSpaceBenchmark;
        bool runIGOHeuristicPlanner;
        double igoMetaSpatialLogitBound;
        double igoMetaTimeLogitBound;
        double igoMetaGammaBoxRadius;
        double igoMetaTimeFloorScale;
        double igoMetaInitialTimeScale;
        double igoMetaTotalSlackMinScale;
        double igoMetaTotalSlackMaxScale;
        int igoArchiveTopK;
        int igoRefineTopK;
        double igoOuterBudgetRatio;
        double igoRefineLBFGSRelCostTol;
        int igoRefineLBFGSMaxIterations;
        int igoRefineLBFGSMaxEvaluations;
        double igoRefineLBFGSMaxWallTime;
        int igoRefineLBFGSMemSize;
        int igoRefineLBFGSPast;
        double metaOptimizerTimeWeight;
        double metaOptimizerLengthWeight;
        double metaOptimizerEnergyWeight;
        double metaOptimizerWaypointSmoothWeight;
        double metaOptimizerCollisionWeight;
        double metaOptimizerVelocityWeight;
        double metaOptimizerAccelerationWeight;
        double metaOptimizerBodyRateWeight;
        double metaOptimizerTiltWeight;
        double metaOptimizerThrustWeight;
        double metaOptimizerMaxAcceleration;
        double metaOptimizerSampleDt;
        int metaOptimizerMidpoints;
        double metaOptimizerTimeLowerBound;
        double metaOptimizerTimeUpperBound;
        double metaOptimizerSimpleMaxVelocity;
        double metaOptimizerSimpleMaxAcceleration;
        double metaOptimizerTargetVelocityRatio;
        double metaOptimizerTimeFloorScale;
        double metaOptimizerInitialTimeScale;
        double metaOptimizerTotalSlackMinScale;
        double metaOptimizerTotalSlackMaxScale;
        int metaOptimizerPopulation;
        int metaOptimizerMaxIterations;
        int metaOptimizerMaxEvaluations;
        double metaOptimizerMaxWallTime;
        std::vector<int> metaOptimizerSeeds;
        double igoPlannerLengthPerPiece;
        double igoPlannerWaypointBoxRadius;
        double igoPlannerTimeLowerScale;
        double igoPlannerTimeUpperScale;
        double igoPlannerSampleDt;
        double igoPlannerSafetyRadius;
        double igoPlannerCollisionWeight;
        double igoPlannerVelocityWeight;
        double igoPlannerAccelerationWeight;
        double igoPlannerLengthWeight;
        double igoPlannerRouteWeight;
        double igoPlannerEnergyWeight;
        double igoPlannerMaxAcceleration;
        bool igoPlannerUsePriorityInitialization;
        int igoPlannerPriorityBasedCount;
        double igoPlannerPriorityVariationRadius;
        std::string figureDirectory;
        double diagnosticsSampleDt;
        std::vector<int> igoSeeds;

        Config(const ros::NodeHandle &nh_priv)
        {
            nh_priv.getParam("MapTopic", mapTopic);
            nh_priv.getParam("TargetTopic", targetTopic);
            nh_priv.getParam("DilateRadius", dilateRadius);
            nh_priv.getParam("VoxelWidth", voxelWidth);
            nh_priv.getParam("MapBound", mapBound);
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
            nh_priv.param("LengthPerPiece", lengthPerPiece, 2.0);

            nh_priv.param("FeasibilityTol", feasibilityTol, 1.0e-4);
            igoFeasibilityTol = feasibilityTol;
            nh_priv.param("IGOFeasibilityTol", igoFeasibilityTol, feasibilityTol);
            nh_priv.param("RecordPath", recordPath,
                          joinPath(getDefaultOutputDirectory(), "compare_records.csv"));

            nh_priv.param("LBFGSRelCostTol", lbfgsRelCostTol, 1.0e-5);
            nh_priv.param("LBFGSMaxIterations", lbfgsMaxIterations, 200);
            nh_priv.param("LBFGSMaxEvaluations", lbfgsMaxEvaluations, 500);
            nh_priv.param("LBFGSMaxWallTime", lbfgsMaxWallTime, 0.5);

            nh_priv.param("IGOPopulation", igoPopulation, 64);
            nh_priv.param("IGOMaxIterations", igoMaxIterations, 200);
            nh_priv.param("IGOMaxEvaluations", igoMaxEvaluations, 2500);
            nh_priv.param("IGOMaxWallTime", igoMaxWallTime, 1.5);
            nh_priv.param("IGOTauBoxRadius", igoTauBoxRadius, 2.5);
            nh_priv.param("IGOXiBoxBound", igoXiBoxBound, 1.5);
            nh_priv.param("RunMetaOptimizer", runMetaOptimizer, true);
            nh_priv.param("RunIGOXSpaceBenchmark", runIGOXSpaceBenchmark, false);
            nh_priv.param("RunIGOHeuristicPlanner", runIGOHeuristicPlanner, false);
            nh_priv.param("IGOMetaSpatialLogitBound", igoMetaSpatialLogitBound, 3.0);
            nh_priv.param("IGOMetaTimeLogitBound", igoMetaTimeLogitBound, 3.0);
            nh_priv.param("IGOMetaGammaBoxRadius", igoMetaGammaBoxRadius, 3.0);
            nh_priv.param("IGOMetaTimeFloorScale", igoMetaTimeFloorScale, 1.15);
            nh_priv.param("IGOMetaInitialTimeScale", igoMetaInitialTimeScale, 1.30);
            nh_priv.param("IGOMetaTotalSlackMinScale", igoMetaTotalSlackMinScale, 0.10);
            nh_priv.param("IGOMetaTotalSlackMaxScale", igoMetaTotalSlackMaxScale, 1.20);
            nh_priv.param("IGOArchiveTopK", igoArchiveTopK, 12);
            nh_priv.param("IGORefineTopK", igoRefineTopK, 4);
            nh_priv.param("IGOOuterBudgetRatio", igoOuterBudgetRatio, 0.45);
            nh_priv.param("IGORefineLBFGSRelCostTol", igoRefineLBFGSRelCostTol, 1.0e-5);
            nh_priv.param("IGORefineLBFGSMaxIterations", igoRefineLBFGSMaxIterations, 120);
            nh_priv.param("IGORefineLBFGSMaxEvaluations", igoRefineLBFGSMaxEvaluations, 500);
            nh_priv.param("IGORefineLBFGSMaxWallTime", igoRefineLBFGSMaxWallTime, 0.50);
            nh_priv.param("IGORefineLBFGSMemSize", igoRefineLBFGSMemSize, 64);
            nh_priv.param("IGORefineLBFGSPast", igoRefineLBFGSPast, 3);
            nh_priv.param("MetaOptimizerTimeWeight", metaOptimizerTimeWeight, 4.0);
            nh_priv.param("MetaOptimizerLengthWeight", metaOptimizerLengthWeight, 0.0);
            nh_priv.param("MetaOptimizerEnergyWeight", metaOptimizerEnergyWeight, 0.0);
            nh_priv.param("MetaOptimizerWaypointSmoothWeight", metaOptimizerWaypointSmoothWeight, 0.0);
            nh_priv.param("MetaOptimizerCollisionWeight", metaOptimizerCollisionWeight, 30.0);
            nh_priv.param("MetaOptimizerVelocityWeight", metaOptimizerVelocityWeight, 20.0);
            nh_priv.param("MetaOptimizerAccelerationWeight", metaOptimizerAccelerationWeight, 25.0);
            nh_priv.param("MetaOptimizerBodyRateWeight", metaOptimizerBodyRateWeight, 0.0);
            nh_priv.param("MetaOptimizerTiltWeight", metaOptimizerTiltWeight, 0.0);
            nh_priv.param("MetaOptimizerThrustWeight", metaOptimizerThrustWeight, 0.0);
            nh_priv.param("MetaOptimizerMaxAcceleration", metaOptimizerMaxAcceleration, 15.0);
            nh_priv.param("MetaOptimizerSampleDt", metaOptimizerSampleDt, 0.10);
            nh_priv.param("MetaOptimizerMidpoints", metaOptimizerMidpoints, 3);
            nh_priv.param("MetaOptimizerTimeLowerBound", metaOptimizerTimeLowerBound, 0.10);
            nh_priv.param("MetaOptimizerTimeUpperBound", metaOptimizerTimeUpperBound, 8.0);
            nh_priv.param("MetaOptimizerSimpleMaxVelocity", metaOptimizerSimpleMaxVelocity, maxVelMag);
            nh_priv.param("MetaOptimizerSimpleMaxAcceleration", metaOptimizerSimpleMaxAcceleration, 15.0);
            nh_priv.param("MetaOptimizerTargetVelocityRatio", metaOptimizerTargetVelocityRatio, 0.90);
            nh_priv.param("MetaOptimizerTimeFloorScale", metaOptimizerTimeFloorScale, 1.10);
            nh_priv.param("MetaOptimizerInitialTimeScale", metaOptimizerInitialTimeScale, 1.18);
            nh_priv.param("MetaOptimizerTotalSlackMinScale", metaOptimizerTotalSlackMinScale, 0.04);
            nh_priv.param("MetaOptimizerTotalSlackMaxScale", metaOptimizerTotalSlackMaxScale, 0.65);
            nh_priv.param("MetaOptimizerPopulation", metaOptimizerPopulation, 50);
            nh_priv.param("MetaOptimizerMaxIterations", metaOptimizerMaxIterations, 500);
            nh_priv.param("MetaOptimizerMaxEvaluations", metaOptimizerMaxEvaluations, 0);
            nh_priv.param("MetaOptimizerMaxWallTime", metaOptimizerMaxWallTime, 0.0);
            if (!nh_priv.getParam("MetaOptimizerSeeds", metaOptimizerSeeds) ||
                metaOptimizerSeeds.empty())
            {
                metaOptimizerSeeds = {0, 1};
            }
            nh_priv.param("IGOPlannerLengthPerPiece", igoPlannerLengthPerPiece, 4.0);
            nh_priv.param("IGOPlannerWaypointBoxRadius", igoPlannerWaypointBoxRadius, 3.0);
            nh_priv.param("IGOPlannerTimeLowerScale", igoPlannerTimeLowerScale, 0.60);
            nh_priv.param("IGOPlannerTimeUpperScale", igoPlannerTimeUpperScale, 3.50);
            nh_priv.param("IGOPlannerSampleDt", igoPlannerSampleDt, 0.10);
            nh_priv.param("IGOPlannerSafetyRadius", igoPlannerSafetyRadius,
                          std::max(dilateRadius + voxelWidth, 2.0 * voxelWidth));
            nh_priv.param("IGOPlannerCollisionWeight", igoPlannerCollisionWeight, 30.0);
            nh_priv.param("IGOPlannerVelocityWeight", igoPlannerVelocityWeight, 20.0);
            nh_priv.param("IGOPlannerAccelerationWeight", igoPlannerAccelerationWeight, 25.0);
            nh_priv.param("IGOPlannerLengthWeight", igoPlannerLengthWeight, 0.0);
            nh_priv.param("IGOPlannerRouteWeight", igoPlannerRouteWeight, 0.0);
            nh_priv.param("IGOPlannerEnergyWeight", igoPlannerEnergyWeight, 0.0);
            nh_priv.param("IGOPlannerMaxAcceleration", igoPlannerMaxAcceleration,
                          std::max(2.0 * maxVelMag, 6.0));
            nh_priv.param("IGOPlannerUsePriorityInitialization", igoPlannerUsePriorityInitialization, true);
            nh_priv.param("IGOPlannerPriorityBasedCount", igoPlannerPriorityBasedCount, 20);
            nh_priv.param("IGOPlannerPriorityVariationRadius", igoPlannerPriorityVariationRadius, 1.0);
            nh_priv.param("FigureDirectory", figureDirectory, getDefaultOutputDirectory());
            nh_priv.param("DiagnosticsSampleDt", diagnosticsSampleDt, 0.05);
            if (!nh_priv.getParam("IGOSeeds", igoSeeds) || igoSeeds.empty())
            {
                igoSeeds = {0, 1, 2, 3, 4};
            }
        }
    };

    static bool ensureParentDirectory(const std::string &file_path)
    {
        const std::size_t slash = file_path.find_last_of('/');
        if (slash == std::string::npos)
        {
            return true;
        }

        const std::string dir = file_path.substr(0, slash);
        if (dir.empty())
        {
            return true;
        }

        if (access(dir.c_str(), F_OK) == 0)
        {
            return true;
        }

        std::string partial;
        if (dir.front() == '/')
        {
            partial = "/";
        }

        std::stringstream ss(dir);
        std::string item;
        while (std::getline(ss, item, '/'))
        {
            if (item.empty())
            {
                continue;
            }

            partial = partial == "/" ? partial + item : partial + "/" + item;
            if (access(partial.c_str(), F_OK) == 0)
            {
                continue;
            }
            if (mkdir(partial.c_str(), 0755) != 0)
            {
                return false;
            }
        }

        return true;
    }

    static bool isBetterResult(const gcopter::GCOPTER_PolytopeSFC::SolverResult &candidate,
                               const gcopter::GCOPTER_PolytopeSFC::SolverResult &best,
                               const double feasibility_tol)
    {
        const bool candidate_success =
            candidate.has_solution && candidate.violations.maxViolation() <= feasibility_tol;
        const bool best_success =
            best.has_solution && best.violations.maxViolation() <= feasibility_tol;

        if (candidate_success != best_success)
        {
            return candidate_success;
        }

        if (candidate_success && best_success)
        {
            if (candidate.objective != best.objective)
            {
                return candidate.objective < best.objective;
            }

            return candidate.violations.maxViolation() < best.violations.maxViolation();
        }

        if (candidate.violations.maxViolation() != best.violations.maxViolation())
        {
            return candidate.violations.maxViolation() < best.violations.maxViolation();
        }

        return candidate.objective < best.objective;
    }

}

class ComparePlanner
{
private:
    Config config;

    ros::NodeHandle nh;
    ros::Subscriber mapSub;
    ros::Subscriber targetSub;

    bool mapInitialized;
    voxel_map::VoxelMap voxelMap;

    Visualizer sharedVisualizer;
    Visualizer lbfgsVisualizer;
    Visualizer igoVisualizer;
    Visualizer metaVisualizer;

    std::vector<Eigen::Vector3d> startGoal;
    std::vector<Eigen::Vector3d> lastRoute;
    Trajectory<5> lbfgsTraj;
    Trajectory<5> igoTraj;
    Trajectory<5> metaTraj;
    double trajStamp;
    int requestId;

    std::string runOutputDirectory;
    std::string resolvedRecordPath;
    std::ofstream recordFile;
    std::ofstream diagnosticsFile;

public:
    ComparePlanner(const Config &conf,
                   ros::NodeHandle &nh_)
        : config(conf),
          nh(nh_),
          mapInitialized(false),
          sharedVisualizer(nh_),
          lbfgsVisualizer(nh_, makeLBFGSStyle()),
          igoVisualizer(nh_, makeIGOStyle()),
          metaVisualizer(nh_, makeMetaStyle()),
          trajStamp(-1.0),
          requestId(0)
    {
        const Eigen::Vector3i xyz((config.mapBound[1] - config.mapBound[0]) / config.voxelWidth,
                                  (config.mapBound[3] - config.mapBound[2]) / config.voxelWidth,
                                  (config.mapBound[5] - config.mapBound[4]) / config.voxelWidth);

        const Eigen::Vector3d offset(config.mapBound[0], config.mapBound[2], config.mapBound[4]);
        voxelMap = voxel_map::VoxelMap(xyz, offset, config.voxelWidth);

        mapSub = nh.subscribe(config.mapTopic, 1, &ComparePlanner::mapCallBack, this,
                              ros::TransportHints().tcpNoDelay());
        targetSub = nh.subscribe(config.targetTopic, 1, &ComparePlanner::targetCallBack, this,
                                 ros::TransportHints().tcpNoDelay());

        initializeOutputPaths();
        openRecordFile();
        openDiagnosticsFile();
    }

private:
    static Visualizer::Style makeLBFGSStyle()
    {
        Visualizer::Style style;
        style.topicPrefix = "/visualizer_lbfgs";
        style.trajectoryColor = {0.0, 0.45, 0.95, 1.0};
        style.waypointsColor = {0.0, 0.45, 0.95, 1.0};
        style.sphereColor = {0.0, 0.45, 0.95, 1.0};
        style.trajectoryWidth = 0.22;
        style.waypointScale = 0.22;
        return style;
    }

    static Visualizer::Style makeIGOStyle()
    {
        Visualizer::Style style;
        style.topicPrefix = "/visualizer_igo";
        style.trajectoryColor = {1.0, 0.45, 0.0, 1.0};
        style.waypointsColor = {1.0, 0.45, 0.0, 1.0};
        style.sphereColor = {1.0, 0.45, 0.0, 1.0};
        style.trajectoryWidth = 0.22;
        style.waypointScale = 0.22;
        return style;
    }

    static Visualizer::Style makeMetaStyle()
    {
        Visualizer::Style style;
        style.topicPrefix = "/visualizer_meta";
        style.trajectoryColor = {0.0, 0.62, 0.28, 1.0};
        style.waypointsColor = {0.0, 0.62, 0.28, 1.0};
        style.sphereColor = {0.0, 0.62, 0.28, 1.0};
        style.trajectoryWidth = 0.22;
        style.waypointScale = 0.22;
        return style;
    }

    inline void initializeOutputPaths()
    {
        const std::string root_output_dir =
            config.figureDirectory.empty() ? getDefaultOutputDirectory()
                                           : config.figureDirectory;
        runOutputDirectory = joinPath(root_output_dir, getCurrentTimestampString());

        std::string record_name("compare_records.csv");
        if (!config.recordPath.empty())
        {
            const std::string candidate_name = getBasename(config.recordPath);
            if (!candidate_name.empty())
            {
                record_name = candidate_name;
            }
        }
        resolvedRecordPath = joinPath(runOutputDirectory, record_name);

        ROS_INFO_STREAM("Compare outputs will be written to: " << runOutputDirectory);
    }

    inline void openRecordFile()
    {
        if (resolvedRecordPath.empty())
        {
            return;
        }

        if (!ensureParentDirectory(resolvedRecordPath))
        {
            ROS_WARN("Failed to create directory for record file: %s", resolvedRecordPath.c_str());
            return;
        }

        const bool needHeader = access(resolvedRecordPath.c_str(), F_OK) != 0;
        recordFile.open(resolvedRecordPath.c_str(), std::ios::out | std::ios::app);
        if (!recordFile.is_open())
        {
            ROS_WARN("Failed to open record file: %s", resolvedRecordPath.c_str());
            return;
        }

        if (needHeader)
        {
            recordFile << "request_id,timestamp,solver,seed,success,has_solution,converged,"
                          "hit_eval_budget,hit_time_budget,solver_status,iterations,eval_count,"
                          "wall_time_sec,optimization_time_sec,frontend_time_sec,"
                          "path_search_time_sec,corridor_generation_time_sec,"
                          "objective,total_duration_sec,trajectory_length,max_violation,"
                          "max_corridor_violation,max_velocity_violation,max_acceleration_violation,"
                          "max_body_rate_violation,"
                          "max_tilt_violation,max_thrust_violation,start_x,start_y,start_z,goal_x,"
                          "goal_y,goal_z,route_size,corridor_size,status\n";
            recordFile.flush();
        }
    }

    struct BenchmarkTiming
    {
        double wall_time = 0.0;
        double optimization_time = 0.0;
        double frontend_time = 0.0;
        double path_search_time = 0.0;
        double corridor_generation_time = 0.0;
    };

    static inline bool solverUsesFrontend(const std::string &solver_name)
    {
        return solver_name != "META";
    }

    static inline BenchmarkTiming makeBenchmarkTiming(const std::string &solver_name,
                                                      const double optimization_time,
                                                      const double path_search_time,
                                                      const double corridor_generation_time)
    {
        BenchmarkTiming timing;
        timing.optimization_time = optimization_time;
        if (solverUsesFrontend(solver_name))
        {
            timing.path_search_time = path_search_time;
            timing.corridor_generation_time = corridor_generation_time;
            timing.frontend_time = path_search_time + corridor_generation_time;
        }
        timing.wall_time = timing.optimization_time + timing.frontend_time;
        return timing;
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
        if (runOutputDirectory.empty())
        {
            return std::string();
        }
        return runOutputDirectory + "/trajectory_metrics.csv";
    }

    inline std::string getMetricsPlotPath() const
    {
        if (runOutputDirectory.empty())
        {
            return std::string();
        }
        return runOutputDirectory + "/metrics_mean.png";
    }

    inline std::string getMetricsMeanCsvPath() const
    {
        if (runOutputDirectory.empty())
        {
            return std::string();
        }
        return runOutputDirectory + "/solver_mean_metrics.csv";
    }

    inline std::string getTrajectoryDataPath() const
    {
        if (runOutputDirectory.empty())
        {
            return std::string();
        }
        return runOutputDirectory + "/request_" + std::to_string(requestId) + "_trajectory.csv";
    }

    inline std::string getTrajectoryPlotPath() const
    {
        if (runOutputDirectory.empty())
        {
            return std::string();
        }
        return runOutputDirectory + "/request_" + std::to_string(requestId) + "_trajectory.png";
    }

    inline void openDiagnosticsFile()
    {
        const std::string path = getDiagnosticsPath();
        if (path.empty())
        {
            return;
        }

        if (!ensureParentDirectory(path))
        {
            ROS_WARN("Failed to create directory for diagnostics file: %s", path.c_str());
            return;
        }

        const bool needHeader = access(path.c_str(), F_OK) != 0;
        diagnosticsFile.open(path.c_str(), std::ios::out | std::ios::app);
        if (!diagnosticsFile.is_open())
        {
            ROS_WARN("Failed to open diagnostics file: %s", path.c_str());
            return;
        }

        if (needHeader)
        {
            diagnosticsFile << "request_id,solver,seed,success,objective,wall_time_sec,"
                               "optimization_time_sec,frontend_time_sec,path_search_time_sec,"
                               "corridor_generation_time_sec,iterations,eval_count,total_duration_sec,trajectory_length,"
                               "max_violation,collision_length,max_collision_distance,"
                               "collision_samples,max_speed,max_acceleration,"
                               "acceleration_energy,jerk_energy,"
                               "max_corridor_violation,max_velocity_violation,"
                               "max_acceleration_violation,max_body_rate_violation,max_tilt_violation,"
                               "max_thrust_violation,min_obstacle_distance,status\n";
            diagnosticsFile.flush();
        }
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

    inline TrajectoryDiagnostics computeTrajectoryDiagnostics(const minco::MINCO_S3NU &jerk_opt) const
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

    inline TrajectoryDiagnostics computeTrajectoryDiagnostics(const Trajectory<5> &trajectory,
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

    inline void appendDiagnostics(const std::string &solver_name,
                                  const int seed,
                                  const gcopter::GCOPTER_PolytopeSFC::SolverResult &result,
                                  const TrajectoryDiagnostics &diagnostics,
                                  const double path_search_time_sec,
                                  const double corridor_generation_time_sec)
    {
        if (!diagnosticsFile.is_open())
        {
            return;
        }

        const bool success = result.has_solution &&
                             result.violations.maxViolation() <= config.igoFeasibilityTol;
        const BenchmarkTiming timing =
            makeBenchmarkTiming(solver_name,
                                result.wall_time,
                                path_search_time_sec,
                                corridor_generation_time_sec);
        diagnosticsFile << requestId << ","
                        << solver_name << ","
                        << seed << ","
                        << (success ? 1 : 0) << ","
                        << result.objective << ","
                        << timing.wall_time << ","
                        << timing.optimization_time << ","
                        << timing.frontend_time << ","
                        << timing.path_search_time << ","
                        << timing.corridor_generation_time << ","
                        << result.iterations << ","
                        << result.eval_count << ","
                        << diagnostics.total_duration << ","
                        << diagnostics.trajectory_length << ","
                        << result.violations.maxViolation() << ","
                        << diagnostics.collision_length << ","
                        << diagnostics.max_collision_distance << ","
                        << diagnostics.collision_samples << ","
                        << diagnostics.max_speed << ","
                        << diagnostics.max_acceleration << ","
                        << diagnostics.acceleration_energy << ","
                        << diagnostics.jerk_energy << ","
                        << result.violations.max_corridor_violation << ","
                        << result.violations.max_velocity_violation << ","
                        << result.violations.max_acceleration_violation << ","
                        << result.violations.max_body_rate_violation << ","
                        << result.violations.max_tilt_violation << ","
                        << result.violations.max_thrust_violation << ","
                        << diagnostics.min_obstacle_distance << ","
                        << "\"" << result.status << "\"\n";
        diagnosticsFile.flush();
    }

    inline void writeTrajectorySamples(const std::vector<Eigen::Vector3d> &route,
                                       const TrajectoryDiagnostics &lbfgsDiagnostics,
                                       const TrajectoryDiagnostics &igoDiagnostics,
                                       const TrajectoryDiagnostics &metaDiagnostics) const
    {
        const std::string path = getTrajectoryDataPath();
        if (path.empty())
        {
            return;
        }

        if (!ensureParentDirectory(path))
        {
            ROS_WARN("Failed to create directory for trajectory data: %s", path.c_str());
            return;
        }

        std::ofstream csv(path.c_str(), std::ios::out | std::ios::trunc);
        if (!csv.is_open())
        {
            ROS_WARN("Failed to open trajectory data file: %s", path.c_str());
            return;
        }

        csv << "request_id,series,sample_index,x,y,z\n";
        auto writeSeries =
            [&](const std::string &series,
                const std::vector<Eigen::Vector3d> &samples) -> void
        {
            for (std::size_t i = 0; i < samples.size(); ++i)
            {
                csv << requestId << ","
                    << series << ","
                    << i << ","
                    << samples[i].x() << ","
                    << samples[i].y() << ","
                    << samples[i].z() << "\n";
            }
        };

        writeSeries("ROUTE", route);
        writeSeries("LBFGS", lbfgsDiagnostics.samples);
        writeSeries("IGO", igoDiagnostics.samples);
        writeSeries("META", metaDiagnostics.samples);
    }

    inline void refreshPlotArtifacts() const
    {
        const std::string output_dir = runOutputDirectory;
        if (output_dir.empty())
        {
            return;
        }

        const std::string script_path = getPlotScriptPath();
        if (access(script_path.c_str(), F_OK) != 0)
        {
            ROS_WARN("Plot script not found: %s", script_path.c_str());
            return;
        }

        std::ostringstream cmd;
        cmd << "python3 "
            << "\"" << script_path << "\""
            << " --output-dir "
            << "\"" << output_dir << "\""
            << " --request-id "
            << requestId
            << " >/dev/null 2>&1";
        const int ret = std::system(cmd.str().c_str());
        if (ret != 0)
        {
            ROS_WARN("Failed to refresh compare plots with command: %s", cmd.str().c_str());
        }
    }

    inline void clearVisualizers()
    {
        sharedVisualizer.clear();
        lbfgsVisualizer.clear();
        igoVisualizer.clear();
        metaVisualizer.clear();
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

    inline gcopter::GCOPTER_PolytopeSFC::LBFGSSolveOptions makeLBFGSOptions() const
    {
        gcopter::GCOPTER_PolytopeSFC::LBFGSSolveOptions options;
        options.rel_cost_tol = config.lbfgsRelCostTol;
        options.max_iterations = config.lbfgsMaxIterations;
        options.max_evaluations = config.lbfgsMaxEvaluations;
        options.max_wall_time = config.lbfgsMaxWallTime;
        return options;
    }

    inline gcopter::GCOPTER_PolytopeSFC::IGOSolveOptions makeIGOOptions(const int seed) const
    {
        gcopter::GCOPTER_PolytopeSFC::IGOSolveOptions options;
        options.population = config.igoPopulation;
        options.max_iterations = config.igoMaxIterations;
        options.max_evaluations = config.igoMaxEvaluations;
        options.max_wall_time = config.igoMaxWallTime;
        options.feasibility_tol = config.igoFeasibilityTol;
        options.tau_box_radius = config.igoTauBoxRadius;
        options.xi_box_bound = config.igoXiBoxBound;
        options.meta_spatial_logit_bound = config.igoMetaSpatialLogitBound;
        options.meta_time_logit_bound = config.igoMetaTimeLogitBound;
        options.meta_gamma_box_radius = config.igoMetaGammaBoxRadius;
        options.meta_time_floor_scale = config.igoMetaTimeFloorScale;
        options.meta_initial_time_scale = config.igoMetaInitialTimeScale;
        options.meta_total_slack_min_scale = config.igoMetaTotalSlackMinScale;
        options.meta_total_slack_max_scale = config.igoMetaTotalSlackMaxScale;
        options.archive_top_k = config.igoArchiveTopK;
        options.refine_top_k = config.igoRefineTopK;
        options.outer_budget_ratio = config.igoOuterBudgetRatio;
        options.refine_rel_cost_tol = config.igoRefineLBFGSRelCostTol;
        options.refine_max_iterations = config.igoRefineLBFGSMaxIterations;
        options.refine_max_evaluations = config.igoRefineLBFGSMaxEvaluations;
        options.refine_max_wall_time = config.igoRefineLBFGSMaxWallTime;
        options.refine_mem_size = config.igoRefineLBFGSMemSize;
        options.refine_past = config.igoRefineLBFGSPast;
        options.meta_optimizer_time_weight = config.metaOptimizerTimeWeight;
        options.meta_optimizer_length_weight = config.metaOptimizerLengthWeight;
        options.meta_optimizer_energy_weight = config.metaOptimizerEnergyWeight;
        options.meta_optimizer_waypoint_smooth_weight = config.metaOptimizerWaypointSmoothWeight;
        options.meta_optimizer_collision_weight = config.metaOptimizerCollisionWeight;
        options.meta_optimizer_velocity_weight = config.metaOptimizerVelocityWeight;
        options.meta_optimizer_acceleration_weight = config.metaOptimizerAccelerationWeight;
        options.meta_optimizer_body_rate_weight = config.metaOptimizerBodyRateWeight;
        options.meta_optimizer_tilt_weight = config.metaOptimizerTiltWeight;
        options.meta_optimizer_thrust_weight = config.metaOptimizerThrustWeight;
        options.meta_optimizer_max_acceleration = config.metaOptimizerMaxAcceleration;
        options.meta_optimizer_sample_dt = config.metaOptimizerSampleDt;
        options.meta_optimizer_midpoints = config.metaOptimizerMidpoints;
        options.meta_optimizer_time_lb = config.metaOptimizerTimeLowerBound;
        options.meta_optimizer_time_ub = config.metaOptimizerTimeUpperBound;
        options.meta_optimizer_simple_max_velocity = config.metaOptimizerSimpleMaxVelocity;
        options.meta_optimizer_simple_max_acceleration = config.metaOptimizerSimpleMaxAcceleration;
        options.meta_optimizer_target_velocity_ratio = config.metaOptimizerTargetVelocityRatio;
        if (config.mapBound.size() == 6)
        {
            options.meta_optimizer_has_workspace_bounds = true;
            options.meta_optimizer_workspace_min =
                Eigen::Vector3d(config.mapBound[0],
                                config.mapBound[2],
                                config.mapBound[4]);
            options.meta_optimizer_workspace_max =
                Eigen::Vector3d(config.mapBound[1],
                                config.mapBound[3],
                                config.mapBound[5]);
        }
        options.meta_optimizer_collision_checker =
            [this](const Eigen::Vector3d &position) -> bool
        {
            return voxelMap.query(position);
        };
        options.seed = static_cast<unsigned int>(seed);
        return options;
    }

    inline gcopter::GCOPTER_PolytopeSFC::IGOSolveOptions makeMetaOptions(const int seed) const
    {
        gcopter::GCOPTER_PolytopeSFC::IGOSolveOptions options = makeIGOOptions(seed);
        options.meta_time_floor_scale = config.metaOptimizerTimeFloorScale;
        options.meta_initial_time_scale = config.metaOptimizerInitialTimeScale;
        options.meta_total_slack_min_scale = config.metaOptimizerTotalSlackMinScale;
        options.meta_total_slack_max_scale = config.metaOptimizerTotalSlackMaxScale;
        options.population = config.metaOptimizerPopulation;
        options.max_iterations = config.metaOptimizerMaxIterations;
        options.max_evaluations = config.metaOptimizerMaxEvaluations;
        options.max_wall_time = config.metaOptimizerMaxWallTime;
        return options;
    }

    inline void appendRecord(const std::string &solver_name,
                             const int seed,
                             const gcopter::GCOPTER_PolytopeSFC::SolverResult &result,
                             const std::vector<Eigen::Vector3d> &route,
                             const std::vector<Eigen::MatrixX4d> &hPolys,
                             const double path_search_time_sec,
                             const double corridor_generation_time_sec)
    {
        if (!recordFile.is_open() || startGoal.size() != 2)
        {
            return;
        }

        const bool success = result.has_solution &&
                             result.violations.maxViolation() <= config.igoFeasibilityTol;
        const BenchmarkTiming timing =
            makeBenchmarkTiming(solver_name,
                                result.wall_time,
                                path_search_time_sec,
                                corridor_generation_time_sec);
        recordFile << requestId << ","
                   << std::fixed << std::setprecision(6) << ros::Time::now().toSec() << ","
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
                   << timing.wall_time << ","
                   << timing.optimization_time << ","
                   << timing.frontend_time << ","
                   << timing.path_search_time << ","
                   << timing.corridor_generation_time << ","
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
                   << startGoal[0](0) << ","
                   << startGoal[0](1) << ","
                   << startGoal[0](2) << ","
                   << startGoal[1](0) << ","
                   << startGoal[1](1) << ","
                   << startGoal[1](2) << ","
                   << route.size() << ","
                   << hPolys.size() << ","
                   << "\"" << result.status << "\"\n";
        recordFile.flush();
    }

    inline void plan()
    {
        if (startGoal.size() != 2)
        {
            return;
        }

        std::vector<Eigen::Vector3d> route;
        const auto path_search_start = std::chrono::steady_clock::now();
        sfc_gen::planPath<voxel_map::VoxelMap>(startGoal[0],
                                               startGoal[1],
                                               voxelMap.getOrigin(),
                                               voxelMap.getCorner(),
                                               &voxelMap, 0.01,
                                               route);
        const double path_search_time_sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          path_search_start)
                .count();

        std::vector<Eigen::MatrixX4d> hPolys;
        if (route.size() <= 1)
        {
            ROS_WARN("Path generation failed for the selected points.");
            return;
        }

        const auto corridor_generation_start = std::chrono::steady_clock::now();
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
        const double corridor_generation_time_sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          corridor_generation_start)
                .count();
        sharedVisualizer.visualizeRoute(route);
        lastRoute = route;
        if (hPolys.empty())
        {
            ROS_WARN("Convex corridor generation failed.");
            return;
        }
        sharedVisualizer.visualizePolytope(hPolys);

        Eigen::Matrix3d iniState;
        Eigen::Matrix3d finState;
        iniState << startGoal[0], Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
        finState << startGoal[1], Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

        gcopter::GCOPTER_PolytopeSFC solver;
        Eigen::VectorXd initialGuess;
        gcopter::GCOPTER_PolytopeSFC::SolverResult lbfgsResult;
        {
            Eigen::VectorXd magnitudeBounds(5);
            Eigen::VectorXd penaltyWeights(5);
            Eigen::VectorXd physicalParams(6);
            magnitudeBounds << config.maxVelMag, config.maxBdrMag, config.maxTiltAngle,
                config.minThrust, config.maxThrust;
            penaltyWeights << config.chiVec[0], config.chiVec[1], config.chiVec[2],
                config.chiVec[3], config.chiVec[4];
            physicalParams << config.vehicleMass, config.gravAcc, config.horizDrag,
                config.vertDrag, config.parasDrag, config.speedEps;

            const bool solver_ready = solver.setup(config.weightT,
                                                   iniState, finState,
                                                   hPolys, config.lengthPerPiece,
                                                   config.smoothingEps,
                                                   config.integralIntervs,
                                                   magnitudeBounds,
                                                   penaltyWeights,
                                                   physicalParams);
            if (!solver_ready)
            {
                ROS_WARN("GCOPTER setup failed.");
                return;
            }
        }

        initialGuess = solver.getCommonInitialGuess();
        lbfgsResult =
            solver.solveLBFGSOriginal(initialGuess, config.lbfgsRelCostTol);
        appendRecord("LBFGS", -1, lbfgsResult, route, hPolys,
                     path_search_time_sec, corridor_generation_time_sec);

        const gcopter::CorridorIGOTrajectoryPlanner igoPlanner;
        gcopter::GCOPTER_PolytopeSFC::SolverResult bestIGOResult;
        bool haveIGOResult = false;
        int bestIGOSeed = -1;
        for (const int seed : config.igoSeeds)
        {
            const gcopter::CorridorIGOTrajectoryPlanner::Result igoResult =
                igoPlanner.solve(solver, initialGuess, makeIGOOptions(seed));
            appendRecord("IGO", seed, igoResult.summary, route, hPolys,
                         path_search_time_sec, corridor_generation_time_sec);

            if (!haveIGOResult ||
                isBetterResult(igoResult.summary, bestIGOResult, config.igoFeasibilityTol))
            {
                bestIGOResult = igoResult.summary;
                bestIGOSeed = seed;
                haveIGOResult = true;
            }
        }

        const gcopter::MetaTrajectoryPlanner metaPlanner;
        gcopter::GCOPTER_PolytopeSFC::SolverResult bestMetaResult;
        bool haveMetaResult = false;
        int bestMetaSeed = -1;
        if (config.runMetaOptimizer)
        {
            for (const int seed : config.metaOptimizerSeeds)
            {
                const gcopter::MetaTrajectoryPlanner::Result metaResult =
                    metaPlanner.solve(solver, initialGuess, makeMetaOptions(seed));
                appendRecord("META", seed, metaResult.summary, route, hPolys,
                             path_search_time_sec, corridor_generation_time_sec);

                if (!haveMetaResult ||
                    isBetterResult(metaResult.summary, bestMetaResult, config.igoFeasibilityTol))
                {
                    bestMetaResult = metaResult.summary;
                    bestMetaSeed = seed;
                    haveMetaResult = true;
                }
            }
        }

        gcopter::GCOPTER_PolytopeSFC::SolverResult bestXSpaceResult;
        bool haveXSpaceResult = false;
        int bestXSpaceSeed = -1;
        if (config.runIGOXSpaceBenchmark)
        {
            for (const int seed : config.igoSeeds)
            {
                const gcopter::GCOPTER_PolytopeSFC::SolverResult xspaceResult =
                    solver.solveIGOXSpaceBenchmark(initialGuess, makeIGOOptions(seed));
                appendRecord("IGO_XSPACE", seed, xspaceResult, route, hPolys,
                             path_search_time_sec, corridor_generation_time_sec);

                if (!haveXSpaceResult || isBetterResult(xspaceResult, bestXSpaceResult, config.igoFeasibilityTol))
                {
                    bestXSpaceResult = xspaceResult;
                    bestXSpaceSeed = seed;
                    haveXSpaceResult = true;
                }
            }
        }

        gcopter::GCOPTER_PolytopeSFC::SolverResult bestHeuristicResult;
        bool haveHeuristicResult = false;
        int bestHeuristicSeed = -1;
        if (config.runIGOHeuristicPlanner)
        {
            for (const int seed : config.igoSeeds)
            {
                const gcopter::GCOPTER_PolytopeSFC::SolverResult heuristicResult =
                    solver.solveIGOHeuristicPlanner(initialGuess, makeIGOOptions(seed));
                appendRecord("IGO_HEURISTIC", seed, heuristicResult, route, hPolys,
                             path_search_time_sec, corridor_generation_time_sec);

                if (!haveHeuristicResult || isBetterResult(heuristicResult, bestHeuristicResult, config.igoFeasibilityTol))
                {
                    bestHeuristicResult = heuristicResult;
                    bestHeuristicSeed = seed;
                    haveHeuristicResult = true;
                }
            }
        }

        lbfgsTraj.clear();
        igoTraj.clear();
        metaTraj.clear();
        minco::MINCO_S3NU lbfgsJerkOpt;
        minco::MINCO_S3NU igoJerkOpt;
        minco::MINCO_S3NU metaJerkOpt;
        bool haveLBFGSJerkOpt = false;
        bool haveIGOJerkOpt = false;
        bool haveMetaJerkOpt = false;
        if (lbfgsResult.has_solution)
        {
            haveLBFGSJerkOpt = solver.buildJerkOpt(lbfgsResult.best_x, lbfgsJerkOpt);
            if (haveLBFGSJerkOpt)
            {
                lbfgsJerkOpt.getTrajectory(lbfgsTraj);
                lbfgsVisualizer.visualizeTrajectory(lbfgsTraj);
            }
        }

        if (haveIGOResult && bestIGOResult.has_solution)
        {
            haveIGOJerkOpt = solver.buildJerkOpt(bestIGOResult.best_x, igoJerkOpt);
            if (haveIGOJerkOpt)
            {
                igoJerkOpt.getTrajectory(igoTraj);
                igoVisualizer.visualizeTrajectory(igoTraj);
            }
        }
        if (haveMetaResult && bestMetaResult.has_solution)
        {
            haveMetaJerkOpt =
                solver.buildMetaDirectJerkOpt(bestMetaResult, metaJerkOpt);
            if (!haveMetaJerkOpt)
            {
                haveMetaJerkOpt =
                    solver.buildJerkOpt(bestMetaResult.best_x, metaJerkOpt);
            }
            if (haveMetaJerkOpt)
            {
                metaJerkOpt.getTrajectory(metaTraj);
                metaVisualizer.visualizeTrajectory(metaTraj);
            }
        }

        const TrajectoryDiagnostics lbfgsDiagnostics =
            haveLBFGSJerkOpt ? computeTrajectoryDiagnostics(lbfgsJerkOpt)
                             : TrajectoryDiagnostics();
        const TrajectoryDiagnostics igoDiagnostics =
            haveIGOJerkOpt ? computeTrajectoryDiagnostics(igoJerkOpt)
                           : TrajectoryDiagnostics();
        const TrajectoryDiagnostics metaDiagnostics =
            haveMetaJerkOpt ? computeTrajectoryDiagnostics(metaJerkOpt)
                            : TrajectoryDiagnostics();
        appendDiagnostics("LBFGS", -1, lbfgsResult, lbfgsDiagnostics,
                          path_search_time_sec, corridor_generation_time_sec);
        if (haveIGOResult)
        {
            appendDiagnostics("IGO", bestIGOSeed, bestIGOResult, igoDiagnostics,
                              path_search_time_sec, corridor_generation_time_sec);
        }
        if (haveMetaResult)
        {
            appendDiagnostics("META", bestMetaSeed, bestMetaResult, metaDiagnostics,
                              path_search_time_sec, corridor_generation_time_sec);
        }
        writeTrajectorySamples(route, lbfgsDiagnostics, igoDiagnostics, metaDiagnostics);
        refreshPlotArtifacts();

        trajStamp = ros::Time::now().toSec();

        std::ostringstream info_stream;
        const std::string metrics_plot_path = getMetricsPlotPath();
        const std::string metrics_mean_csv_path = getMetricsMeanCsvPath();
        const std::string trajectory_plot_path = getTrajectoryPlotPath();
        const double frontend_time_sec = path_search_time_sec + corridor_generation_time_sec;
        info_stream << "Request " << requestId
                    << " LBFGS obj=" << lbfgsResult.objective
                    << " success=" << (lbfgsResult.has_solution &&
                                       lbfgsResult.violations.maxViolation() <= config.igoFeasibilityTol)
                    << " | IGO(best seed=" << bestIGOSeed
                    << ") obj=" << (haveIGOResult ? bestIGOResult.objective : std::numeric_limits<double>::infinity())
                    << " success=" << (haveIGOResult && bestIGOResult.has_solution &&
                                       bestIGOResult.violations.maxViolation() <= config.igoFeasibilityTol)
                    << " | META(best seed=" << bestMetaSeed
                    << ") obj=" << (haveMetaResult ? bestMetaResult.objective : std::numeric_limits<double>::infinity())
                    << " success=" << (haveMetaResult && bestMetaResult.has_solution &&
                                       bestMetaResult.violations.maxViolation() <= config.igoFeasibilityTol)
                    << " | opt_time LBFGS=" << lbfgsResult.wall_time
                    << " IGO=" << (haveIGOResult ? bestIGOResult.wall_time : std::numeric_limits<double>::infinity())
                    << " META=" << (haveMetaResult ? bestMetaResult.wall_time : std::numeric_limits<double>::infinity())
                    << " | frontend path=" << path_search_time_sec
                    << " corridor=" << corridor_generation_time_sec
                    << " total=" << frontend_time_sec
                    << " | benchmark_time LBFGS=" << (lbfgsResult.wall_time + frontend_time_sec)
                    << " IGO=" << (haveIGOResult ? bestIGOResult.wall_time + frontend_time_sec : std::numeric_limits<double>::infinity())
                    << " META=" << (haveMetaResult ? bestMetaResult.wall_time : std::numeric_limits<double>::infinity())
                    << " | LBFGS traj_len=" << lbfgsDiagnostics.trajectory_length
                    << " coll_len=" << lbfgsDiagnostics.collision_length
                    << " max_coll_dist=" << lbfgsDiagnostics.max_collision_distance
                    << " jerk_energy=" << lbfgsDiagnostics.jerk_energy
                    << " min_obs_dist=" << lbfgsDiagnostics.min_obstacle_distance
                    << " | IGO traj_len=" << igoDiagnostics.trajectory_length
                    << " coll_len=" << igoDiagnostics.collision_length
                    << " max_coll_dist=" << igoDiagnostics.max_collision_distance
                    << " jerk_energy=" << igoDiagnostics.jerk_energy
                    << " min_obs_dist=" << igoDiagnostics.min_obstacle_distance
                    << " | META traj_len=" << metaDiagnostics.trajectory_length
                    << " coll_len=" << metaDiagnostics.collision_length
                    << " max_coll_dist=" << metaDiagnostics.max_collision_distance
                    << " jerk_energy=" << metaDiagnostics.jerk_energy
                    << " min_obs_dist=" << metaDiagnostics.min_obstacle_distance;
        if (config.runIGOXSpaceBenchmark)
        {
            info_stream << " | IGO_XSPACE(best seed=" << bestXSpaceSeed
                        << ") obj=" << (haveXSpaceResult ? bestXSpaceResult.objective : std::numeric_limits<double>::infinity())
                        << " success=" << (haveXSpaceResult && bestXSpaceResult.has_solution &&
                                           bestXSpaceResult.violations.maxViolation() <= config.igoFeasibilityTol);
        }
        if (config.runIGOHeuristicPlanner)
        {
            info_stream << " | IGO_HEURISTIC(best seed=" << bestHeuristicSeed
                        << ") obj=" << (haveHeuristicResult ? bestHeuristicResult.objective : std::numeric_limits<double>::infinity())
                        << " success=" << (haveHeuristicResult && bestHeuristicResult.has_solution &&
                                           bestHeuristicResult.violations.maxViolation() <= config.igoFeasibilityTol);
        }
        if (!metrics_plot_path.empty())
        {
            info_stream << " | metrics_png=" << metrics_plot_path;
        }
        if (!metrics_mean_csv_path.empty())
        {
            info_stream << " | mean_csv=" << metrics_mean_csv_path;
        }
        if (!trajectory_plot_path.empty())
        {
            info_stream << " | traj_png=" << trajectory_plot_path;
        }

        ROS_INFO_STREAM(info_stream.str());

        ++requestId;
    }

public:
    inline void targetCallBack(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        if (!mapInitialized)
        {
            return;
        }

        if (startGoal.size() >= 2)
        {
            startGoal.clear();
            lbfgsTraj.clear();
            igoTraj.clear();
            metaTraj.clear();
            clearVisualizers();
        }

        const double zGoal = config.mapBound[4] + config.dilateRadius +
                             fabs(msg->pose.orientation.z) *
                                 (config.mapBound[5] - config.mapBound[4] - 2 * config.dilateRadius);
        const Eigen::Vector3d goal(msg->pose.position.x, msg->pose.position.y, zGoal);
        if (voxelMap.query(goal) == 0)
        {
            sharedVisualizer.visualizeStartGoal(goal, 0.5, startGoal.size());
            startGoal.emplace_back(goal);
            if (startGoal.size() == 2)
            {
                plan();
            }
        }
        else
        {
            ROS_WARN("Infeasible position selected.");
        }
    }

    inline void process()
    {
        if (trajStamp < 0.0)
        {
            return;
        }

        const double delta = ros::Time::now().toSec() - trajStamp;
        if (lbfgsTraj.getPieceNum() > 0 && delta >= 0.0 && delta < lbfgsTraj.getTotalDuration())
        {
            lbfgsVisualizer.visualizeSphere(lbfgsTraj.getPos(delta), 0.50);
        }
        if (igoTraj.getPieceNum() > 0 && delta >= 0.0 && delta < igoTraj.getTotalDuration())
        {
            igoVisualizer.visualizeSphere(igoTraj.getPos(delta), 0.50);
        }
        if (metaTraj.getPieceNum() > 0 && delta >= 0.0 && delta < metaTraj.getTotalDuration())
        {
            metaVisualizer.visualizeSphere(metaTraj.getPos(delta), 0.50);
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "compare_planning_node");
    ros::NodeHandle nh;

    ComparePlanner planner(Config(ros::NodeHandle("~")), nh);

    ros::Rate rate(100);
    while (ros::ok())
    {
        planner.process();
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
