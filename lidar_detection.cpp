#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <regex>
#include <vector>
#include <stdexcept>
#include <cstring>      // for std::memcpy

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>

// A struct to hold each point's data, matching Python parse
struct PointData {
    float x, y, z;
    float intensity;
    uint8_t tag, line;
    double timestamp;
};


std::vector<PointData> parse_pointcloud2_file(const std::string& file_path)
{
    // 1) Read entire file
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file.");
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    // 2) Regex to find the data block: data=[...]
    std::regex re("data=\\[([^\\]]*)\\]");
    std::smatch match;
    if (!std::regex_search(content, match, re)) {
        throw std::runtime_error("Could not find 'data=[...]' block in file.");
    }

    // 3) Split by comma and convert to bytes
    std::vector<uint8_t> byte_vals;
    {
        std::stringstream ss(match[1].str());
        std::string val;
        while (std::getline(ss, val, ',')) {
            // Trim whitespace and convert
            int b = std::stoi(val);
            byte_vals.push_back(static_cast<uint8_t>(b));
        }
    }

    // 4) Each point is 26 bytes => 4 floats (x,y,z,intensity) + 2 uint8 (tag,line) + 1 double (timestamp)
    //    => same as the Python struct.unpack(fmt='<4f2Bd')
    const size_t STEP = 26;
    if (byte_vals.size() % STEP != 0) {
        throw std::runtime_error("Byte data is not a multiple of 26. Check file format!");
    }

    size_t n_points = byte_vals.size() / STEP;
    std::vector<PointData> points;
    points.reserve(n_points);

    for (size_t i = 0; i < n_points; ++i) {
        PointData p;
        // Copy x,y,z,intensity
        std::memcpy(&p.x,         &byte_vals[i*STEP +  0], sizeof(float));
        std::memcpy(&p.y,         &byte_vals[i*STEP +  4], sizeof(float));
        std::memcpy(&p.z,         &byte_vals[i*STEP +  8], sizeof(float));
        std::memcpy(&p.intensity, &byte_vals[i*STEP + 12], sizeof(float));

        // Copy tag, line (each 1 byte)
        p.tag  = byte_vals[i*STEP + 16];
        p.line = byte_vals[i*STEP + 17];

        // Copy timestamp (double => 8 bytes)
        std::memcpy(&p.timestamp, &byte_vals[i*STEP + 18], sizeof(double));

        points.push_back(p);
    }
    return points;
}

// A small helper to compute bounding box min/max for one cluster
struct BoundingBox {
    int cluster_id;
    Eigen::Vector3f min_pt;
    Eigen::Vector3f max_pt;
};

// Main pipeline
int main()
{
    // 1) Load data from text file:
    std::string file_path = "livox_data.txt";
    std::vector<PointData> raw_points = parse_pointcloud2_file(file_path);

    // Convert raw_points into a PCL PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
    cloud_in->reserve(raw_points.size());
    for (const auto& rp : raw_points) {
        pcl::PointXYZ pt(rp.x, rp.y, rp.z);
        cloud_in->push_back(pt);
    }
    std::cout << "Loaded " << cloud_in->size() << " points from file.\n";

    // 2) Downsample (voxel grid)
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
    {
        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        voxel.setInputCloud(cloud_in);
        voxel.setLeafSize(0.1f, 0.1f, 0.1f);
        voxel.filter(*cloud_downsampled);
    }
    std::cout << "After downsampling: " << cloud_downsampled->size() << " points.\n";

    // 3) Segment ground plane (RANSAC)
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.2);

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);
    seg.setInputCloud(cloud_downsampled);
    seg.segment(*inliers, *coeffs);

    // Separate ground vs obstacles
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_obstacles(new pcl::PointCloud<pcl::PointXYZ>);
    {
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_downsampled);
        extract.setIndices(inliers);

        // Ground
        extract.setNegative(false);
        extract.filter(*cloud_ground);

        // Obstacles
        extract.setNegative(true);
        extract.filter(*cloud_obstacles);
    }
    std::cout << "Ground inliers: " << cloud_ground->size()
              << ", Obstacles: "   << cloud_obstacles->size() << std::endl;

    // 4) Remove outliers from obstacles
    {
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud_obstacles);
        sor.setMeanK(20);
        sor.setStddevMulThresh(2.0);
        sor.filter(*cloud_obstacles);
    }
    std::cout << "Obstacles after outlier removal: " << cloud_obstacles->size() << std::endl;

    // 5) Cluster the above-ground points:
    
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud_obstacles);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.2);   // akin to DBSCAN eps
    ec.setMinClusterSize(15);      // akin to DBSCAN min_samples
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_obstacles);
    ec.extract(cluster_indices);

    int n_clusters = static_cast<int>(cluster_indices.size());
    std::cout << "Found " << n_clusters << " clusters.\n";

    // 6) Compute bounding boxes for each cluster
    std::vector<BoundingBox> boxes;
    boxes.reserve(n_clusters);

    int cluster_id = 0;
    for (auto & indices : cluster_indices) {
        // Gather cluster points
        Eigen::Vector3f min_pt( std::numeric_limits<float>::max(),
                                std::numeric_limits<float>::max(),
                                std::numeric_limits<float>::max());
        Eigen::Vector3f max_pt(-std::numeric_limits<float>::max(),
                               -std::numeric_limits<float>::max(),
                               -std::numeric_limits<float>::max());

        for (int idx : indices.indices) {
            const auto & pt = (*cloud_obstacles)[idx];
            min_pt.x() = std::min(min_pt.x(), pt.x);
            min_pt.y() = std::min(min_pt.y(), pt.y);
            min_pt.z() = std::min(min_pt.z(), pt.z);
            max_pt.x() = std::max(max_pt.x(), pt.x);
            max_pt.y() = std::max(max_pt.y(), pt.y);
            max_pt.z() = std::max(max_pt.z(), pt.z);
        }

        BoundingBox bb;
        bb.cluster_id = cluster_id;
        bb.min_pt     = min_pt;
        bb.max_pt     = max_pt;
        boxes.push_back(bb);

        cluster_id++;
    }

    // 7) Filter out bounding boxes that are too large or too small
    std::vector<BoundingBox> final_bboxes;
    for (auto & b : boxes) {
        Eigen::Vector3f size = b.max_pt - b.min_pt;
        float dx = size.x(), dy = size.y(), dz = size.z();

      
        bool pass_max = (dx < 3.0f && dy < 3.0f && dz < 3.0f);
        bool pass_min = (dx > 0.1f || dy > 0.1f || dz > 0.1f); 
        if (pass_max && pass_min) {
            final_bboxes.push_back(b);
        }
    }

    // Print final bounding box info
    std::cout << "==== Final Bounding Boxes ====\n";
    for (auto & box : final_bboxes) {
        Eigen::Vector3f mi = box.min_pt;
        Eigen::Vector3f ma = box.max_pt;
        std::cout << "Cluster " << box.cluster_id << " => "
                  << "min=(" << mi.x() << "," << mi.y() << "," << mi.z() << "), "
                  << "max=(" << ma.x() << "," << ma.y() << "," << ma.z() << ")"
                  << std::endl;
    }

    pcl::visualization::PCLVisualizer viewer("Refined LiDAR Clusters");
    viewer.setBackgroundColor(0, 0, 0);

    // Show the ground in green
    {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ground_color(cloud_ground, 0, 255, 0);
        viewer.addPointCloud<pcl::PointXYZ>(cloud_ground, ground_color, "ground");
    }
    // Show obstacle cloud overall (faint)
    {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> obstacle_color(cloud_obstacles, 255, 255, 255);
        viewer.addPointCloud<pcl::PointXYZ>(cloud_obstacles, obstacle_color, "obstacles");
    }

    // Now highlight each cluster in a random color
    int i = 0;
    for (auto & indices : cluster_indices) {
        // Check if the cluster_id is in final_bboxes
        bool is_in_final = false;
        for (auto & box : final_bboxes) {
            if (box.cluster_id == i) {
                is_in_final = true;
                break;
            }
        }
        // Create a new cloud for just that cluster
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (auto idx : indices.indices) {
            cluster_cloud->push_back((*cloud_obstacles)[idx]);
        }

        // Random color
        uint8_t r = static_cast<uint8_t>(rand() % 256);
        uint8_t g = static_cast<uint8_t>(rand() % 256);
        uint8_t b = static_cast<uint8_t>(rand() % 256);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cluster_color(cluster_cloud, r, g, b);

        std::string name = "cluster_" + std::to_string(i);
        viewer.addPointCloud<pcl::PointXYZ>(cluster_cloud, cluster_color, name);

        // If it's in final bboxes, draw a wireframe bounding box
        if (is_in_final) {
            // Find bounding box
            for (auto & box : final_bboxes) {
                if (box.cluster_id == i) {
                    // We'll draw a wireframe cube
                    viewer.addCube(box.min_pt.x(), box.max_pt.x(),
                                   box.min_pt.y(), box.max_pt.y(),
                                   box.min_pt.z(), box.max_pt.z(),
                                   static_cast<double>(r)/255.0,
                                   static_cast<double>(g)/255.0,
                                   static_cast<double>(b)/255.0,
                                   "bbox_" + std::to_string(i));
                    viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                                                       pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
                                                       "bbox_" + std::to_string(i));
                    break;
                }
            }
        }

        i++;
    }

    viewer.addCoordinateSystem(1.0);
    viewer.initCameraParameters();
    while (!viewer.wasStopped()) {
        viewer.spinOnce(10);
    }

    return 0;
}
