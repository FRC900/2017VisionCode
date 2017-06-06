#!/usr/bin/env bash

# Setup ROS for Jetson Master
source /opt/ros/kinetic/setup.bash
source /home/ubuntu/2017VisionCode/zebROS_ws/install_isolated/setup.bash
export ROS_MASTER_URI=http://10.9.0.8:5802
export ROS_IP=10.9.0.8
export ROSLAUNCH_SSH_UNKOWN=1
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

exec "$@"
