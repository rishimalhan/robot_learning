#!/bin/bash
Xvfb :1 -screen 0 1920x1200x24 &
x11vnc -display :1 -forever -nopw -listen localhost -xkb &
websockify --web=/usr/share/novnc/ 8080 localhost:5900 &
# Source the setup file
source /opt/ros/$ROS_DISTRO/setup.bash
sleep infinity