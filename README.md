Commands to run to start the sensors and the LiDAR

Sensors
1) ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyACM0 -b 115200
2) ros2 run my_lidar_pkg wheelchair [this runs the logic in main.cpp]

LiDAR
1) ros2 launch livox_ros_driver2 msg_MID360_launch.py
2) ros2 topic echo /livox/lidar
3) ros2 run my_lidar_pkg lidar_detection_with_distance
