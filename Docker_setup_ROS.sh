# ROS Setup install script for new jetsons
# Source: https://github.com/jetsonhacks/installROSTX1/blob/master/installROS.sh
# Setup Locale
# update-locale LANG=C LANGUAGE=C LC_ALL=C LC_MESSAGES=POSIX
# Setup sources.lst
sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# Setup keys
apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 0xB01FA116
# Installation
apt update
# Add Individual Packages here
# You can install a specific ROS package (replace underscores with dashes of the package name):
# apt-get install ros-kinetic-PACKAGE
# e.g.
# apt-get install ros-kinetic-navigation
# To find available packages:
# apt-cache search ros-kinetic
#
apt install -y python-rosdep python-rosinstall terminator ros-kinetic-rqt ros-kinetic-rqt-common-plugins ros-kinetic-tf2-ros \
 ros-kinetic-pcl-conversions ros-kinetic-cv-bridge ros-kinetic-tf ros-kinetic-map-server ros-kinetic-rviz ros-kinetic-hector-slam \
 ros-kinetic-hector-slam-launch ros-kinetic-rtabmap-ros ros-kinetic-robot-localization ros-kinetic-navigation ros-kinetic-robot-state-publisher \
 python-wstool ninja-build libsuitesparse-dev ros-kinetic-tf2-tools ros-kinetic-hardware-interface ros-kinetic-controller-manager ros-kinetic-control-msgs \
 ros-kinetic-joint-limits-interface ros-kinetic-transmission-interface ros-kinetic-control-toolbox libgoogle-glog-dev google-mock libcairo2-dev \
 protobuf-compiler liblua5.3-dev libopencv-dev ros-kinetic-rosparam-shortcuts
# Initialize rosdep
rosdep init

# Google Cartographer installation from https://google-cartographer-ros.readthedocs.io/en/latest/
# (Latest as of 4/27/2017, at least)
cd /ws/zebROS_ws
# Merge the cartographer_ros.rosinstall file and fetch code for dependencies.
wstool merge -t src https://raw.githubusercontent.com/googlecartographer/cartographer_ros/master/cartographer_ros.rosinstall
wstool update -t src

# Install deb dependencies.
# The command 'rosdep init' will print an error if you have already
# executed it since installing ROS. This error can be ignored.
rosdep update
rosdep install --from-paths src --ignore-src --rosdistro=${ROS_DISTRO} -y

rm -rf /ws/zebROS_ws/src/zed-ros-wrapper-master /ws/zebROS_ws/src/zed_covariance # Disable ZED stuff 
source /opt/ros/kinetic/setup.bash
catkin_make_isolated --install --use-ninja
source install_isolated/setup.bash
