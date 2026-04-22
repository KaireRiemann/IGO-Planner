#ifndef VISUALIZER_HPP
#define VISUALIZER_HPP

#include "gcopter/trajectory.hpp"
#include "gcopter/bspline_trajectory.hpp"
#include "gcopter/quickhull.hpp"
#include "gcopter/geo_utils.hpp"

#include <iostream>
#include <memory>
#include <chrono>
#include <cmath>

#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// Visualizer for the planner
class Visualizer
{
public:
    struct Color
    {
        double r = 1.0;
        double g = 1.0;
        double b = 1.0;
        double a = 1.0;
    };

    struct Style
    {
        std::string topicPrefix = "/visualizer";
        Color routeColor{1.0, 0.0, 0.0, 1.0};
        Color waypointsColor{1.0, 0.0, 0.0, 1.0};
        Color trajectoryColor{0.0, 0.5, 1.0, 1.0};
        Color meshColor{0.0, 0.0, 1.0, 0.15};
        Color edgeColor{0.0, 1.0, 1.0, 1.0};
        Color sphereColor{0.0, 0.0, 1.0, 1.0};
        Color startGoalColor{1.0, 0.0, 0.0, 1.0};
        double routeWidth = 0.1;
        double waypointScale = 0.35;
        double trajectoryWidth = 0.30;
        double edgeWidth = 0.02;
    };

private:
    // config contains the scale for some markers
    ros::NodeHandle nh;

    // These are publishers for path, waypoints on the trajectory,
    // the entire trajectory, the mesh of free-space polytopes,
    // the edge of free-space polytopes, and spheres for safety radius
    ros::Publisher routePub;
    ros::Publisher wayPointsPub;
    ros::Publisher trajectoryPub;
    ros::Publisher meshPub;
    ros::Publisher edgePub;
    ros::Publisher spherePub;
    Style style;

public:
    ros::Publisher speedPub;
    ros::Publisher thrPub;
    ros::Publisher tiltPub;
    ros::Publisher bdrPub;

public:
    Visualizer(ros::NodeHandle &nh_)
        : nh(nh_)
    {
        style = Style();
        initializePublishers();
    }

    Visualizer(ros::NodeHandle &nh_,
               const Style &style_)
        : nh(nh_),
          style(style_)
    {
        initializePublishers();
    }

private:
    inline void initializePublishers()
    {
        routePub = nh.advertise<visualization_msgs::Marker>(style.topicPrefix + "/route", 10);
        wayPointsPub = nh.advertise<visualization_msgs::Marker>(style.topicPrefix + "/waypoints", 10);
        trajectoryPub = nh.advertise<visualization_msgs::Marker>(style.topicPrefix + "/trajectory", 10);
        meshPub = nh.advertise<visualization_msgs::Marker>(style.topicPrefix + "/mesh", 1000);
        edgePub = nh.advertise<visualization_msgs::Marker>(style.topicPrefix + "/edge", 1000);
        spherePub = nh.advertise<visualization_msgs::Marker>(style.topicPrefix + "/spheres", 1000);
        speedPub = nh.advertise<std_msgs::Float64>(style.topicPrefix + "/speed", 1000);
        thrPub = nh.advertise<std_msgs::Float64>(style.topicPrefix + "/total_thrust", 1000);
        tiltPub = nh.advertise<std_msgs::Float64>(style.topicPrefix + "/tilt_angle", 1000);
        bdrPub = nh.advertise<std_msgs::Float64>(style.topicPrefix + "/body_rate", 1000);
    }

    inline void applyColor(visualization_msgs::Marker &marker,
                           const Color &color) const
    {
        marker.color.r = color.r;
        marker.color.g = color.g;
        marker.color.b = color.b;
        marker.color.a = color.a;
    }

public:
    inline void visualizeRoute(const std::vector<Eigen::Vector3d> &route)
    {
        visualization_msgs::Marker routeMarker;

        routeMarker.id = 0;
        routeMarker.type = visualization_msgs::Marker::LINE_LIST;
        routeMarker.header.stamp = ros::Time::now();
        routeMarker.header.frame_id = "odom";
        routeMarker.pose.orientation.w = 1.00;
        routeMarker.action = visualization_msgs::Marker::ADD;
        routeMarker.ns = "route";
        applyColor(routeMarker, style.routeColor);
        routeMarker.scale.x = style.routeWidth;

        if (route.size() > 0)
        {
            bool first = true;
            Eigen::Vector3d last;
            for (auto it : route)
            {
                if (first)
                {
                    first = false;
                    last = it;
                    continue;
                }
                geometry_msgs::Point point;

                point.x = last(0);
                point.y = last(1);
                point.z = last(2);
                routeMarker.points.push_back(point);
                point.x = it(0);
                point.y = it(1);
                point.z = it(2);
                routeMarker.points.push_back(point);
                last = it;
            }

            routePub.publish(routeMarker);
        }
    }

    inline void clear()
    {
        visualization_msgs::Marker marker;
        marker.header.stamp = ros::Time::now();
        marker.header.frame_id = "odom";
        marker.action = visualization_msgs::Marker::DELETEALL;
        routePub.publish(marker);
        wayPointsPub.publish(marker);
        trajectoryPub.publish(marker);
        meshPub.publish(marker);
        edgePub.publish(marker);
        spherePub.publish(marker);
    }

    template <int D>
    inline void visualizeTrajectory(const Trajectory<D> &traj)
    {
        visualization_msgs::Marker wayPointsMarker, trajMarker;

        wayPointsMarker.id = 0;
        wayPointsMarker.type = visualization_msgs::Marker::SPHERE_LIST;
        wayPointsMarker.header.stamp = ros::Time::now();
        wayPointsMarker.header.frame_id = "odom";
        wayPointsMarker.pose.orientation.w = 1.00;
        wayPointsMarker.action = visualization_msgs::Marker::ADD;
        wayPointsMarker.ns = "waypoints";
        applyColor(wayPointsMarker, style.waypointsColor);
        wayPointsMarker.scale.x = style.waypointScale;
        wayPointsMarker.scale.y = style.waypointScale;
        wayPointsMarker.scale.z = style.waypointScale;

        trajMarker = wayPointsMarker;
        trajMarker.type = visualization_msgs::Marker::LINE_LIST;
        trajMarker.ns = "trajectory";
        applyColor(trajMarker, style.trajectoryColor);
        trajMarker.scale.x = style.trajectoryWidth;

        if (traj.getPieceNum() > 0)
        {
            Eigen::MatrixXd wps = traj.getPositions();
            for (int i = 0; i < wps.cols(); i++)
            {
                geometry_msgs::Point point;
                point.x = wps.col(i)(0);
                point.y = wps.col(i)(1);
                point.z = wps.col(i)(2);
                wayPointsMarker.points.push_back(point);
            }

            wayPointsPub.publish(wayPointsMarker);
        }

        if (traj.getPieceNum() > 0)
        {
            double T = 0.01;
            Eigen::Vector3d lastX = traj.getPos(0.0);
            for (double t = T; t < traj.getTotalDuration(); t += T)
            {
                geometry_msgs::Point point;
                Eigen::Vector3d X = traj.getPos(t);
                point.x = lastX(0);
                point.y = lastX(1);
                point.z = lastX(2);
                trajMarker.points.push_back(point);
                point.x = X(0);
                point.y = X(1);
                point.z = X(2);
                trajMarker.points.push_back(point);
                lastX = X;
            }
            trajectoryPub.publish(trajMarker);
        }
    }

    inline void visualizeTrajectory(const bsplinetrajectory::NUBSTrajectory<3> &traj)
    {
        visualization_msgs::Marker controlPointsMarker, trajMarker;

        controlPointsMarker.id = 1;
        controlPointsMarker.type = visualization_msgs::Marker::SPHERE_LIST;
        controlPointsMarker.header.stamp = ros::Time::now();
        controlPointsMarker.header.frame_id = "odom";
        controlPointsMarker.pose.orientation.w = 1.00;
        controlPointsMarker.action = visualization_msgs::Marker::ADD;
        controlPointsMarker.ns = "nubs_control_points";
        applyColor(controlPointsMarker, Color{0.0, 0.65, 1.0, 1.0});
        controlPointsMarker.scale.x = style.waypointScale;
        controlPointsMarker.scale.y = style.waypointScale;
        controlPointsMarker.scale.z = style.waypointScale;

        trajMarker = controlPointsMarker;
        trajMarker.id = 1;
        trajMarker.type = visualization_msgs::Marker::LINE_LIST;
        trajMarker.ns = "nubs_trajectory";
        applyColor(trajMarker, Color{0.0, 0.85, 1.0, 1.0});
        trajMarker.scale.x = style.trajectoryWidth;

        if (traj.getPieceNum() > 0)
        {
            const auto &control_points = traj.getControlPoints();
            for (int i = 0; i < control_points.rows(); ++i)
            {
                geometry_msgs::Point point;
                point.x = control_points(i, 0);
                point.y = control_points(i, 1);
                point.z = control_points(i, 2);
                controlPointsMarker.points.push_back(point);
            }
            wayPointsPub.publish(controlPointsMarker);

            const double sample_dt = 0.01;
            Eigen::Vector3d lastX = traj.evaluate(0.0, 0);
            for (double t = sample_dt; t < traj.getTotalDuration(); t += sample_dt)
            {
                geometry_msgs::Point point;
                const Eigen::Vector3d X = traj.evaluate(t, 0);
                point.x = lastX(0);
                point.y = lastX(1);
                point.z = lastX(2);
                trajMarker.points.push_back(point);
                point.x = X(0);
                point.y = X(1);
                point.z = X(2);
                trajMarker.points.push_back(point);
                lastX = X;
            }
            trajectoryPub.publish(trajMarker);
        }
    }

    // Visualize the trajectory and its front-end path
    template <int D>
    inline void visualize(const Trajectory<D> &traj,
                          const std::vector<Eigen::Vector3d> &route)
    {
        visualizeRoute(route);
        visualizeTrajectory(traj);
    }

    inline void visualize(const bsplinetrajectory::NUBSTrajectory<3> &traj,
                          const std::vector<Eigen::Vector3d> &route)
    {
        visualizeRoute(route);
        visualizeTrajectory(traj);
    }

    // Visualize some polytopes in H-representation
    inline void visualizePolytope(const std::vector<Eigen::MatrixX4d> &hPolys)
    {

        // Due to the fact that H-representation cannot be directly visualized
        // We first conduct vertex enumeration of them, then apply quickhull
        // to obtain triangle meshs of polyhedra
        Eigen::Matrix3Xd mesh(3, 0), curTris(3, 0), oldTris(3, 0);
        for (size_t id = 0; id < hPolys.size(); id++)
        {
            oldTris = mesh;
            Eigen::Matrix<double, 3, -1, Eigen::ColMajor> vPoly;
            geo_utils::enumerateVs(hPolys[id], vPoly);

            quickhull::QuickHull<double> tinyQH;
            const auto polyHull = tinyQH.getConvexHull(vPoly.data(), vPoly.cols(), false, true);
            const auto &idxBuffer = polyHull.getIndexBuffer();
            int hNum = idxBuffer.size() / 3;

            curTris.resize(3, hNum * 3);
            for (int i = 0; i < hNum * 3; i++)
            {
                curTris.col(i) = vPoly.col(idxBuffer[i]);
            }
            mesh.resize(3, oldTris.cols() + curTris.cols());
            mesh.leftCols(oldTris.cols()) = oldTris;
            mesh.rightCols(curTris.cols()) = curTris;
        }

        // RVIZ support tris for visualization
        visualization_msgs::Marker meshMarker, edgeMarker;

        meshMarker.id = 0;
        meshMarker.header.stamp = ros::Time::now();
        meshMarker.header.frame_id = "odom";
        meshMarker.pose.orientation.w = 1.00;
        meshMarker.action = visualization_msgs::Marker::ADD;
        meshMarker.type = visualization_msgs::Marker::TRIANGLE_LIST;
        meshMarker.ns = "mesh";
        applyColor(meshMarker, style.meshColor);
        meshMarker.scale.x = 1.0;
        meshMarker.scale.y = 1.0;
        meshMarker.scale.z = 1.0;

        edgeMarker = meshMarker;
        edgeMarker.type = visualization_msgs::Marker::LINE_LIST;
        edgeMarker.ns = "edge";
        applyColor(edgeMarker, style.edgeColor);
        edgeMarker.scale.x = style.edgeWidth;

        geometry_msgs::Point point;

        int ptnum = mesh.cols();

        for (int i = 0; i < ptnum; i++)
        {
            point.x = mesh(0, i);
            point.y = mesh(1, i);
            point.z = mesh(2, i);
            meshMarker.points.push_back(point);
        }

        for (int i = 0; i < ptnum / 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                point.x = mesh(0, 3 * i + j);
                point.y = mesh(1, 3 * i + j);
                point.z = mesh(2, 3 * i + j);
                edgeMarker.points.push_back(point);
                point.x = mesh(0, 3 * i + (j + 1) % 3);
                point.y = mesh(1, 3 * i + (j + 1) % 3);
                point.z = mesh(2, 3 * i + (j + 1) % 3);
                edgeMarker.points.push_back(point);
            }
        }

        meshPub.publish(meshMarker);
        edgePub.publish(edgeMarker);

        return;
    }

    // Visualize all spheres with centers sphs and the same radius
    inline void visualizeSphere(const Eigen::Vector3d &center,
                                const double &radius)
    {
        visualization_msgs::Marker sphereMarkers, sphereDeleter;

        sphereMarkers.id = 0;
        sphereMarkers.type = visualization_msgs::Marker::SPHERE_LIST;
        sphereMarkers.header.stamp = ros::Time::now();
        sphereMarkers.header.frame_id = "odom";
        sphereMarkers.pose.orientation.w = 1.00;
        sphereMarkers.action = visualization_msgs::Marker::ADD;
        sphereMarkers.ns = "spheres";
        applyColor(sphereMarkers, style.sphereColor);
        sphereMarkers.scale.x = radius * 2.0;
        sphereMarkers.scale.y = radius * 2.0;
        sphereMarkers.scale.z = radius * 2.0;

        sphereDeleter = sphereMarkers;
        sphereDeleter.action = visualization_msgs::Marker::DELETE;

        geometry_msgs::Point point;
        point.x = center(0);
        point.y = center(1);
        point.z = center(2);
        sphereMarkers.points.push_back(point);

        spherePub.publish(sphereDeleter);
        spherePub.publish(sphereMarkers);
    }

    inline void visualizeNUBSSphere(const Eigen::Vector3d &center,
                                    const double &radius)
    {
        visualization_msgs::Marker sphereMarkers, sphereDeleter;

        sphereMarkers.id = 1;
        sphereMarkers.type = visualization_msgs::Marker::SPHERE_LIST;
        sphereMarkers.header.stamp = ros::Time::now();
        sphereMarkers.header.frame_id = "odom";
        sphereMarkers.pose.orientation.w = 1.00;
        sphereMarkers.action = visualization_msgs::Marker::ADD;
        sphereMarkers.ns = "nubs_spheres";
        applyColor(sphereMarkers, Color{0.0, 0.85, 1.0, 1.0});
        sphereMarkers.scale.x = radius * 2.0;
        sphereMarkers.scale.y = radius * 2.0;
        sphereMarkers.scale.z = radius * 2.0;

        sphereDeleter = sphereMarkers;
        sphereDeleter.action = visualization_msgs::Marker::DELETE;

        geometry_msgs::Point point;
        point.x = center(0);
        point.y = center(1);
        point.z = center(2);
        sphereMarkers.points.push_back(point);

        spherePub.publish(sphereDeleter);
        spherePub.publish(sphereMarkers);
    }

    inline void visualizeStartGoal(const Eigen::Vector3d &center,
                                   const double &radius,
                                   const int sg)
    {
        visualization_msgs::Marker sphereMarkers, sphereDeleter;

        sphereMarkers.id = sg;
        sphereMarkers.type = visualization_msgs::Marker::SPHERE_LIST;
        sphereMarkers.header.stamp = ros::Time::now();
        sphereMarkers.header.frame_id = "odom";
        sphereMarkers.pose.orientation.w = 1.00;
        sphereMarkers.action = visualization_msgs::Marker::ADD;
        sphereMarkers.ns = "StartGoal";
        applyColor(sphereMarkers, style.startGoalColor);
        sphereMarkers.scale.x = radius * 2.0;
        sphereMarkers.scale.y = radius * 2.0;
        sphereMarkers.scale.z = radius * 2.0;

        sphereDeleter = sphereMarkers;
        sphereDeleter.action = visualization_msgs::Marker::DELETEALL;

        geometry_msgs::Point point;
        point.x = center(0);
        point.y = center(1);
        point.z = center(2);
        sphereMarkers.points.push_back(point);

        if (sg == 0)
        {
            spherePub.publish(sphereDeleter);
            ros::Duration(1.0e-9).sleep();
            sphereMarkers.header.stamp = ros::Time::now();
        }
        spherePub.publish(sphereMarkers);
    }
};

#endif
