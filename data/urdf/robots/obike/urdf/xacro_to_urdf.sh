#!/bin/bash
source /opt/ros/foxy/setup.bash
#SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# add to ROS package path
#export ROS_PACKAGE_PATH=${SCRIPT_DIR}/../..:$ROS_PACKAGE_PATH
#echo "Using ROS_PACKAGE_PATH ${ROS_PACKAGE_PATH}"
ros2 run xacro xacro -o obike.urdf obike.xacro
