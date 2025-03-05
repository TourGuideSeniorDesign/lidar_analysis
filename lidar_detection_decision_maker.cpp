#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Core>
#include <limits>
#include <vector>
#include <mutex>
#include <sstream>
#include <cmath>
#include <cstdlib>

// this is how the output looks like: [INFO] [1741188660.063564512] [lidar_detection_decision_maker]: Steering angle: 0.00 deg, Speed limit: 2.00 mph; [INFO] [1741188660.063737862] [lidar_detection_decision_maker]: Processed cloud: Down:5646, Ground:2688, Obstacles:2936, Clusters:13, BBoxes:9

// Speed & LiDAR Constants 
// Speed constants (in mph)
const float SPEED_MAX     = 3.0f;  // Full speed
const float SPEED_CAUTION = 2.0f;  // Caution
const float SPEED_SLOW    = 1.0f;  // Slow
const float SPEED_STOP    = 0.0f;  // Stop

// LiDAR thresholds (meters) for determining speed limit
const float LIDAR_FULL_THRESHOLD    = 3.0f;
const float LIDAR_CAUTION_THRESHOLD = 1.0f;

// Data Structures 
struct BoundingBox {
  int cluster_id;
  Eigen::Vector3f min_pt;
  Eigen::Vector3f max_pt;
  float distance;   // Distance from LiDAR (origin)
  float angle_deg;  // Angle in degrees (0 = front, +left, -right)
};

//LiDAR Sensor Module for Speed Decision 
class LiDARSensor {
private:
  float obstacleDistance;  // Nearest obstacle distance (m)
  float speedLimit;        // Computed speed limit based on LiDAR reading

public:
  LiDARSensor() : obstacleDistance(100.0f), speedLimit(SPEED_MAX) { }

  // Update obstacle distance then compute speed limit accordingly.
  void updateDistance(float distance) {
    obstacleDistance = distance;
    if (obstacleDistance >= LIDAR_FULL_THRESHOLD) {
      speedLimit = SPEED_MAX;
    } else if (obstacleDistance >= LIDAR_CAUTION_THRESHOLD) {
      speedLimit = SPEED_CAUTION;
    } else {
      speedLimit = SPEED_SLOW;
    }
  }
    
  float getSpeedLimit() const {
    return speedLimit;
  }
};

//ROS2 Node: LiDAR Detection & Decision Maker 
class LidarDetectionDecisionMakerNode : public rclcpp::Node {
public:
  LidarDetectionDecisionMakerNode() : Node("lidar_detection_decision_maker") {
    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/livox/lidar", 10,
      std::bind(&LidarDetectionDecisionMakerNode::cloudCallback, this, std::placeholders::_1)
    );

    // Create a PCL Visualizer window
    viewer_ = std::make_shared<pcl::visualization::PCLVisualizer>("Lidar Decision Maker");
    viewer_->setBackgroundColor(0, 0, 0);

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(50),
      std::bind(&LidarDetectionDecisionMakerNode::updateViewer, this)
    );

    bbox_publisher_ = this->create_publisher<std_msgs::msg::String>("/lidar_bbox_data", 10);

    // Instantiate LiDARSensor module for speed decision
    lidar_sensor_ = std::make_shared<LiDARSensor>();
  }

private:
  // Decision 1: Compute steering angle based on bounding boxes.
  // If an obstacle is within a ±30° front cone and 2 m away,
  // steer left or right accordingly.
  float decideSteeringAngle(const std::vector<BoundingBox>& bboxes) {
    float frontConeDeg = 30.0f;
    float dangerDist   = 2.0f;
    float steerAngle   = 0.0f; // Default: go straight

    for (auto &box : bboxes) {
      if (box.distance < dangerDist && std::fabs(box.angle_deg) < frontConeDeg) {
        if (box.angle_deg > 0.0f) {
          // Obstacle on left, steer right
          steerAngle = -30.0f;
        } else {
          // Obstacle on right, steer left
          steerAngle = 30.0f;
        }
        break; // Once a danger is detected, set steering and exit
      }
    }
    return steerAngle;
  }

  // Main callback to process incoming LiDAR point clouds
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // Convert ROS PointCloud2 to a PCL point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud_in);

    if (cloud_in->empty()) {
      RCLCPP_WARN(this->get_logger(), "Received empty cloud, skipping processing.");
      return;
    }

    // Downsample the cloud using a VoxelGrid filter
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
    {
      pcl::VoxelGrid<pcl::PointXYZ> voxel;
      voxel.setInputCloud(cloud_in);
      voxel.setLeafSize(0.1f, 0.1f, 0.1f);
      voxel.filter(*cloud_downsampled);
    }

    // Segment the ground using RANSAC (plane model)
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.2f);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);
    seg.setInputCloud(cloud_downsampled);
    seg.segment(*inliers, *coeffs);

    // Separate ground and obstacles
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_obstacles(new pcl::PointCloud<pcl::PointXYZ>);
    {
      if (!inliers->indices.empty()) {
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_downsampled);
        extract.setIndices(inliers);

        extract.setNegative(false);
        extract.filter(*cloud_ground);

        extract.setNegative(true);
        extract.filter(*cloud_obstacles);
      } else {
        cloud_obstacles = cloud_downsampled;
      }
    }

    // Remove outliers from the obstacles cloud
    {
      pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
      sor.setInputCloud(cloud_obstacles);
      sor.setMeanK(20);
      sor.setStddevMulThresh(2.0);
      sor.filter(*cloud_obstacles);
    }

    // Cluster the obstacles using Euclidean Cluster Extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud_obstacles);
    std::vector<pcl::PointIndices> cluster_indices;
    {
      pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
      ec.setClusterTolerance(0.2); // 20cm
      ec.setMinClusterSize(30);    // Discard small clusters
      ec.setSearchMethod(tree);
      ec.setInputCloud(cloud_obstacles);
      ec.extract(cluster_indices);
    }

    // Build bounding boxes for each cluster
    std::vector<BoundingBox> boxes;
    boxes.reserve(cluster_indices.size());
    int cluster_id = 0;
    for (auto &indices : cluster_indices) {
      Eigen::Vector3f min_pt(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
      );
      Eigen::Vector3f max_pt(
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max()
      );
      for (int idx : indices.indices) {
        const auto &pt = (*cloud_obstacles)[idx];
        min_pt.x() = std::min(min_pt.x(), pt.x);
        min_pt.y() = std::min(min_pt.y(), pt.y);
        min_pt.z() = std::min(min_pt.z(), pt.z);
        max_pt.x() = std::max(max_pt.x(), pt.x);
        max_pt.y() = std::max(max_pt.y(), pt.y);
        max_pt.z() = std::max(max_pt.z(), pt.z);
      }
      Eigen::Vector3f center = (min_pt + max_pt) / 2.0f;
      float distance = center.norm();
      float angle_rad = std::atan2(center.y(), center.x());
      float angle_deg = angle_rad * 180.0f / M_PI;

      BoundingBox bb;
      bb.cluster_id = cluster_id;
      bb.min_pt = min_pt;
      bb.max_pt = max_pt;
      bb.distance = distance;
      bb.angle_deg = angle_deg;
      boxes.push_back(bb);
      cluster_id++;
    }

    // Filter bounding boxes by size (skip overly large or small ones)
    std::vector<BoundingBox> final_bboxes;
    for (auto &b : boxes) {
      Eigen::Vector3f size = b.max_pt - b.min_pt;
      float dx = size.x(), dy = size.y(), dz = size.z();
      bool pass_max = (dx < 3.0f && dy < 3.0f && dz < 3.0f);
      bool pass_min = (dx > 0.1f || dy > 0.1f || dz > 0.1f);
      if (pass_max && pass_min) {
        final_bboxes.push_back(b);
      }
    }

    // --- Decision Making --- 
    // 1. Steering: determine angle based on obstacles within a ±30° cone and 2m threshold.
    float steering_angle = decideSteeringAngle(final_bboxes);

    // 2. Speed: determine minimum obstacle distance and update LiDARSensor.
    float min_distance = std::numeric_limits<float>::max();
    for (auto &b : final_bboxes) {
      if (b.distance < min_distance)
        min_distance = b.distance;
    }
    // Use a default high distance if no obstacles were found.
    if (final_bboxes.empty()) {
      min_distance = 100.0f;
    }
    lidar_sensor_->updateDistance(min_distance);
    float speed_limit = lidar_sensor_->getSpeedLimit();

    RCLCPP_INFO(this->get_logger(), "Steering angle: %.2f deg, Speed limit: %.2f mph", 
                steering_angle, speed_limit);
    RCLCPP_INFO(this->get_logger(), "Processed cloud: Down:%lu, Ground:%lu, Obstacles:%lu, Clusters:%zu, BBoxes:%zu",
                cloud_downsampled->size(), cloud_ground->size(), cloud_obstacles->size(), 
                cluster_indices.size(), final_bboxes.size());

    // Publish bounding box information as a text message
    std_msgs::msg::String bbox_msg;
    std::stringstream ss;
    for (auto &b : final_bboxes) {
      ss << "Cluster " << b.cluster_id
         << " | Dist=" << b.distance << "m"
         << " | Angle=" << b.angle_deg << "deg"
         << " | min(" << b.min_pt.x() << "," << b.min_pt.y() << "," << b.min_pt.z() << ")"
         << " | max(" << b.max_pt.x() << "," << b.max_pt.y() << "," << b.max_pt.z() << ")\n";
    }
    bbox_msg.data = ss.str();
    bbox_publisher_->publish(bbox_msg);

    // Store data for visualization (protected by mutex)
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      ground_cloud_ = cloud_ground;
      obstacles_cloud_ = cloud_obstacles;
      cluster_indices_ = cluster_indices;
      final_bboxes_ = final_bboxes;
    }
  }

  // Timer callback to update the PCL Visualizer window
  void updateViewer() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground, obstacles;
    std::vector<pcl::PointIndices> clusters;
    std::vector<BoundingBox> bboxes;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      ground = ground_cloud_;
      obstacles = obstacles_cloud_;
      clusters = cluster_indices_;
      bboxes = final_bboxes_;
    }

    if (!ground || !obstacles) return;

    viewer_->removeAllPointClouds();
    viewer_->removeAllShapes();

    // Render the ground in green
    {
      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(ground, 0, 255, 0);
      viewer_->addPointCloud<pcl::PointXYZ>(ground, green, "ground_cloud");
    }

    // Render the obstacles in white
    {
      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> white(obstacles, 255, 255, 255);
      viewer_->addPointCloud<pcl::PointXYZ>(obstacles, white, "obstacles_cloud");
    }

    // Display each cluster with a random color and draw its bounding box with text.
    int i = 0;
    for (auto &indices : clusters) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      for (auto idx : indices.indices) {
        cluster_cloud->push_back((*obstacles)[idx]);
      }
      uint8_t rr = static_cast<uint8_t>(rand() % 256);
      uint8_t gg = static_cast<uint8_t>(rand() % 256);
      uint8_t bb = static_cast<uint8_t>(rand() % 256);
      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(cluster_cloud, rr, gg, bb);
      std::string cloud_name = "cluster_" + std::to_string(i);
      viewer_->addPointCloud<pcl::PointXYZ>(cluster_cloud, color, cloud_name);

      // Draw bounding box and add 3D text label
      for (auto &box : bboxes) {
        if (box.cluster_id == i) {
          viewer_->addCube(
            box.min_pt.x(), box.max_pt.x(),
            box.min_pt.y(), box.max_pt.y(),
            box.min_pt.z(), box.max_pt.z(),
            static_cast<double>(rr)/255.0,
            static_cast<double>(gg)/255.0,
            static_cast<double>(bb)/255.0,
            "bbox_" + std::to_string(i)
          );
          viewer_->setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
            "bbox_" + std::to_string(i)
          );
          Eigen::Vector3f box_center = (box.min_pt + box.max_pt) / 2.0f;
          std::string text_id = "text_" + std::to_string(i);
          char text_buf[100];
          snprintf(text_buf, 100, "Dist: %.2f m\nAngle: %.2f deg", box.distance, box.angle_deg);
          viewer_->addText3D<pcl::PointXYZ>(
            text_buf,
            pcl::PointXYZ(box_center.x(), box_center.y(), box.max_pt.z() + 0.3f),
            0.2f, 1.0f, 1.0f, 1.0f,
            text_id
          );
          break;
        }
      }
      i++;
    }
    viewer_->spinOnce(10);
  }

  // Member variables
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr bbox_publisher_;
  std::mutex data_mutex_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr obstacles_cloud_;
  std::vector<pcl::PointIndices> cluster_indices_;
  std::vector<BoundingBox> final_bboxes_;
  pcl::visualization::PCLVisualizer::Ptr viewer_;
  std::shared_ptr<LiDARSensor> lidar_sensor_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LidarDetectionDecisionMakerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
