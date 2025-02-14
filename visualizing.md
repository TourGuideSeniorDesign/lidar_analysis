# Visualizing LiDAR Data

There are multiple ways to visualize LiDAR data, depending on your setup and tools. Below are two methods:

## 1. Visualizing LiDAR Data on a Laptop (Without ROS)

If you want to visualize LiDAR data without requiring ROS, follow these steps:

### **Step 1: Convert the ROS2 Bag File**
Create a YAML configuration file (`convert.yaml`) to specify the output format:

```bash
cat << EOF > convert.yaml
output_bags:
  - uri: ros3_output
    storage_id: mcap
    all: true
EOF




![Image 2-14-25 at 5 15 PM](https://github.com/user-attachments/assets/2234df41-7978-4a13-8a70-fb9920661b46)

2. If we want to visualize on the 
