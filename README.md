Commands to run to start the sensors and the LiDAR

Sensors
1) ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyACM0 -b 115200
2) ros2 run my_lidar_pkg wheelchair [this runs the logic in main.cpp]

LiDAR
1) ros2 launch livox_ros_driver2 msg_MID360_launch.py
2) ros2 topic echo /livox/lidar
3) ros2 run my_lidar_pkg lidar_detection_with_distance


2025 README additions - Arturo Matlin

Purpose
This C++ ROS 2 package contains multiple programs for processing Livox Mid‑360 LiDAR point clouds. Its nodes and utilities perform ground segmentation, obstacle clustering, bounding‑box computation and basic decision making. Unlike the runtime nodes in wheelchair_code_module, these programs are focused on prototyping and visualising LiDAR processing rather than acting on sensor data. The repository also holds Jupyter notebooks and instructions for visualising point‑cloud logs.

Programs
1. lidar_analysis_publisher.cpp / lidar_detection_streamer

Description: ROS 2 node that subscribes to /livox/lidar, processes live point clouds and publishes bounding‑box data.

Behavior: Inside the callback it converts the incoming PointCloud2 message to a PCL cloud, downsamples via a voxel grid, segments the ground with RANSAC, removes outliers, performs Euclidean clustering and builds bounding boxes. It filters out very large or very small clusters before storing the results for visualisation and           publishes a summary of the clusters on the /lidar_bbox_data topic. It logs counts for the downsampled, ground and obstacle points as well as the number of clusters and      boxes. A PCL visualiser window displays the ground (green), obstacles (white) and each cluster in a unique colour with 3‑D bounding boxes.
   

2. lidar_detection.cpp

Description: Stand‑alone C++ tool for analysing offline LiDAR logs stored as byte arrays.

Behavior: Reads a text file containing raw Livox bytes, parses each 26‑byte packet into x,y,z,intensity,tag,line,timestamp structures and converts them into a PCL point cloud. It then applies the same pipeline as the live node—downsampling, ground segmentation, outlier removal, clustering and bounding‑box computation. The program prints cluster and bounding‑box information to the console and visualises the clusters with a PCL viewer.

3. lidar_detection_decision_maker.cpp

Description: ROS 2 node that couples obstacle detection with simple navigation heuristics.

Behavior: After the usual downsample–segment–cluster pipeline it computes bounding boxes and uses a helper class LiDARSensor to map the nearest obstacle distance to a speed limit (3 mph at ≥3 m, 2 mph at ≥1 m, 1 mph otherwise). A separate function decideSteeringAngle examines bounding boxes within ±30° and 2 m; if an obstacle is on the left it steers right and vice‑versa. The node logs the recommended steering angle and speed limit and publishes bounding‑box data for consumption by other nodes.

4. lidar_detection_distance_coordinates.cpp

Description: ROS 2 node emphasising distance measurement.

Behavior: Similar pipeline to the streamer but stores the centre of each bounding box and calculates its Euclidean distance from the LiDAR origin. It prints each cluster’s distance and coordinates to the ROS 2 log and visualises the clusters and their distances in a PCL viewer.

5. lidar_detection_script_2.cpp

Description: Simplified ROS 2 node for real‑time clustering.

Behavior: Downsamples the incoming point cloud, segments the ground, removes outliers, clusters obstacles and draws wire‑frame bounding boxes in a PCL viewer. Unlike the decision‑making node, it does not compute distances or speed commands.

Additional Info:

Visualization instructions – A visualizing.md document describes how to convert a ROS 2 bag (*.db3) to an MCAP file using ros2 bag convert and then view it in Foxglove Studio.

Notebooks – The Jupyter notebooks (lidar_detection_from_live_data_logs.ipynb, lidar_research.ipynb) contain exploratory LiDAR analyses.

Dependencies – All programs depend on ROS 2 Humble, the sensor_msgs and std_msgs message types, PCL (Point Cloud Library) and Eigen.

Running – To build the package, clone it into your ROS 2 workspace and run colcon build. Execute the programs using ros2 run lidar_analysis <executable> (e.g., ros2 run lidar_analysis lidar_detection_streamer). For the offline tool, compile using colcon build --packages-select lidar_analysis and run ./build/lidar_analysis/lidar_detection from the build directory.
