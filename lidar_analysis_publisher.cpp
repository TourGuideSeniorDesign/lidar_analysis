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
#include <thread>
#include <sstream>
#include <cmath>

// A struct to hold bounding box info: min/max, distance, and angle
struct BoundingBox {
  int cluster_id;
  Eigen::Vector3f min_pt;
  Eigen::Vector3f max_pt;
  float distance;   // Distance from LiDAR (origin)
  float angle_deg;  // Angle in degrees (0=front, +left, -right)
};

class LidarDetectionNode : public rclcpp::Node
{
public:
  LidarDetectionNode()
  : Node("lidar_detection_streamer")
  {
    // Subscribe to LiDAR topic
    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/livox/lidar",  
      10,
      std::bind(&LidarDetectionNode::cloudCallback, this, std::placeholders::_1)
    );

    RCLCPP_INFO(this->get_logger(), 
                "Subscribed to /livox/lidar. Waiting for live LiDAR data...");

    // Create a PCL Visualizer 
    viewer_ = std::make_shared<pcl::visualization::PCLVisualizer>("Refined LiDAR Clusters");
    viewer_->setBackgroundColor(0, 0, 0);

    // Timer to refresh the PCL viewer ~20 times/s
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(50),
      std::bind(&LidarDetectionNode::updateViewer, this)
    );

    // Publisher for bounding-box data
    bbox_publisher_ = this->create_publisher<std_msgs::msg::String>(
      "/lidar_bbox_data",
      10
    );
  }

private:
  // Callback: runs when a new PointCloud2 arrives
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // Convert ROS2 PointCloud2 -> PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud_in);

    if (cloud_in->empty()) {
      RCLCPP_WARN(this->get_logger(), "Received an empty cloud. Skipping processing.");
      return;
    }

    // Downsample
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
    {
      pcl::VoxelGrid<pcl::PointXYZ> voxel;
      voxel.setInputCloud(cloud_in);
      voxel.setLeafSize(0.1f, 0.1f, 0.1f);
      voxel.filter(*cloud_downsampled);
    }

    // Segment ground (RANSAC plane)
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.2f);

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);
    seg.setInputCloud(cloud_downsampled);
    seg.segment(*inliers, *coeffs);

    // Separate ground & obstacles
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_obstacles(new pcl::PointCloud<pcl::PointXYZ>);
    {
      if (!inliers->indices.empty()) {
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_downsampled);
        extract.setIndices(inliers);

        // Extract ground
        extract.setNegative(false);
        extract.filter(*cloud_ground);

        // Extract obstacles
        extract.setNegative(true);
        extract.filter(*cloud_obstacles);
      } else {
        // If no plane found, treat all points as obstacles
        cloud_obstacles = cloud_downsampled;
      }
    }

    // Remove outliers (Statistical Outlier Removal)
    {
      pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
      sor.setInputCloud(cloud_obstacles);
      sor.setMeanK(20);
      sor.setStddevMulThresh(2.0);
      sor.filter(*cloud_obstacles);
    }

    // Cluster extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud_obstacles);

    std::vector<pcl::PointIndices> cluster_indices;
    {
      pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
      ec.setClusterTolerance(0.2); // 20cm
      ec.setMinClusterSize(30);    // discard small clusters
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

      // Center & distance
      Eigen::Vector3f center = (min_pt + max_pt) / 2.0f;
      float distance = center.norm();

      // Calculate angle (assuming x=front, y=left, z=up)
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

    // 6) Filter bounding boxes by size
    std::vector<BoundingBox> final_bboxes;
    for (auto &b : boxes) {
      Eigen::Vector3f size = b.max_pt - b.min_pt;
      float dx = size.x(), dy = size.y(), dz = size.z();

      // Skip very large or very small
      bool pass_max = (dx < 3.0f && dy < 3.0f && dz < 3.0f);
      bool pass_min = (dx > 0.1f || dy > 0.1f || dz > 0.1f);

      if (pass_max && pass_min) {
        final_bboxes.push_back(b);
      }
    }

    // 7) Store data for visualization (lock mutex)
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      ground_cloud_ = cloud_ground;
      obstacles_cloud_ = cloud_obstacles;
      cluster_indices_ = cluster_indices;
      final_bboxes_ = final_bboxes;
    }

    // 8) Logging
    RCLCPP_INFO(
      this->get_logger(), 
      "Live cloud => Down:%lu, Ground:%lu, Obstacles:%lu, Clusters:%zu, BBoxes:%zu",
      cloud_downsampled->size(), 
      cloud_ground->size(),
      cloud_obstacles->size(),
      cluster_indices.size(),
      final_bboxes.size()
    );

    // Print bounding box info (distance + coords + angle)
    for (auto & b : final_bboxes) {
      RCLCPP_INFO(
        this->get_logger(),
        " -> Cluster %d: dist= %.2f m, angle= %.2f deg, "
        "min(%.2f,%.2f,%.2f), max(%.2f,%.2f,%.2f)",
        b.cluster_id, b.distance, b.angle_deg,
        b.min_pt.x(), b.min_pt.y(), b.min_pt.z(),
        b.max_pt.x(), b.max_pt.y(), b.max_pt.z()
      );
    }

    // Publish bounding box info as text
    std_msgs::msg::String bbox_msg;
    std::stringstream ss;
    for (auto & b : final_bboxes) {
      ss << "Cluster " << b.cluster_id
         << " | Dist=" << b.distance << "m"
         << " | Angle=" << b.angle_deg << "deg"
         << " | min(" << b.min_pt.x() << "," << b.min_pt.y() << "," << b.min_pt.z() << ")"
         << " | max(" << b.max_pt.x() << "," << b.max_pt.y() << "," << b.max_pt.z() << ")\n";
    }
    bbox_msg.data = ss.str();
    bbox_publisher_->publish(bbox_msg);
  }

  // Timer callback: refresh PCL visualizer
  void updateViewer()
  {
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

    // If no data, do nothing
    if (!ground || !obstacles) {
      return;
    }

    // Clear old data
    viewer_->removeAllPointClouds();
    viewer_->removeAllShapes();

    // Show ground in green
    {
      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(ground, 0, 255, 0);
      viewer_->addPointCloud<pcl::PointXYZ>(ground, green, "ground_cloud");
    }

    // Show obstacles in white
    {
      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> white(obstacles, 255, 255, 255);
      viewer_->addPointCloud<pcl::PointXYZ>(obstacles, white, "obstacles_cloud");
    }

    // Color each cluster, draw bounding boxes, add distance/angle text
    int i = 0;
    for (auto & indices : clusters) {
      // Build cluster cloud
      pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      for (auto idx : indices.indices) {
        cluster_cloud->push_back((*obstacles)[idx]);
      }

      // Random color
      uint8_t rr = static_cast<uint8_t>(rand() % 256);
      uint8_t gg = static_cast<uint8_t>(rand() % 256);
      uint8_t bb = static_cast<uint8_t>(rand() % 256);
      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(cluster_cloud, rr, gg, bb);

      std::string cloud_name = "cluster_" + std::to_string(i);
      viewer_->addPointCloud<pcl::PointXYZ>(cluster_cloud, color, cloud_name);

      // Find corresponding bounding box
      for (auto & box : bboxes) {
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

          // 3D text label above bounding box with distance + angle
          {
            Eigen::Vector3f box_center = (box.min_pt + box.max_pt) / 2.0f;
            std::string text_id = "distance_text_" + std::to_string(i);
            char text_buf[100];
            snprintf(text_buf, 100, "Dist: %.2f m\nAngle: %.2f deg", box.distance, box.angle_deg);

            viewer_->addText3D<pcl::PointXYZ>(
              text_buf,
              pcl::PointXYZ(box_center.x(), box_center.y(), box.max_pt.z() + 0.3f),
              0.2f,  // text size
              1.0f, 1.0f, 1.0f,  // color (white)
              text_id
            );
          }
          break; // Found the bounding box for this cluster
        }
      }
      i++;
    }

    // viewer_->addCoordinateSystem(1.0); // optional
    // viewer_->resetCamera(); // optional
    viewer_->spinOnce(10);
  }

  // ---------------------------------------------------------------------------
  // Member variables
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr bbox_publisher_;

  // Data for visualization, protected by mutex
  std::mutex data_mutex_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr obstacles_cloud_;
  std::vector<pcl::PointIndices> cluster_indices_;
  std::vector<BoundingBox> final_bboxes_;

  // The PCL Visualizer
  pcl::visualization::PCLVisualizer::Ptr viewer_;
};

// Standard ROS 2 main
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LidarDetectionNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
