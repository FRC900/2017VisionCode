# Setup ROS for Local Development

source /opt/ros/kinetic/setup.bash
source ~/2017VisionCode/zebROS_ws/devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
export ROS_IP=$(hostname -I)
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

