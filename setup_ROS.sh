#!/bin/bash
# ROS Setup install script for new jetsons
# Source: https://github.com/jetsonhacks/installROSTX1/blob/master/installROS.sh

# Setup Locale
# sudo update-locale LANG=C LANGUAGE=C LC_ALL=C LC_MESSAGES=POSIX
# Setup sources.lst
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# Setup keys
sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 0xB01FA116
# Installation
sudo apt update
sudo apt install ros-kinetic-ros-base -y
# Add Individual Packages here
# You can install a specific ROS package (replace underscores with dashes of the package name):
# sudo apt-get install ros-kinetic-PACKAGE
# e.g.
# sudo apt-get install ros-kinetic-navigation
#
# To find available packages:
# apt-cache search ros-kinetic
# 
# Initialize rosdep
sudo apt-get install python-rosdep -y
# ssl certificates can get messed up on TX1 for some reason
sudo c_rehash /etc/ssl/certs
# Initialize rosdep
sudo rosdep init
# To find available packages, use:
rosdep update

source /opt/ros/kinetic/setup.bash

sudo apt install python-rosinstall -y
sudo apt install terminator
sudo apt install ros-kinetic-rqt ros-kinetic-rqt-common-plugins ros-kinetic-tf2-ros ros-kinetic-pcl-conversions ros-kinetic-cv-bridge ros-kinetic-tf ros-kinetic-map-server ros-kinetic-rviz ros-kinetic-hector-slam ros-kinetic-hector-slam-launch
