# Use an LXDE+VNC base image (includes noVNC at port 8080)
FROM dorowu/ubuntu-desktop-lxde-vnc:focal-arm64

USER root

ARG DEBIAN_FRONTEND=noninteractive

# Create workspace
WORKDIR /robot_learning
COPY . /robot_learning/

# Ensure essential packages are installed
RUN apt-get update && \
    apt-get install -y python3 python3-pip \
    software-properties-common \
    curl \
    wget \
    gnupg2 \
    build-essential \
    git \
    cmake \
    iproute2 \
    iputils-ping \
    net-tools \
    novnc \
    x11vnc \
    python3-websockify && \
    rm -rf /var/lib/apt/lists/*

# Allow pip to manage system-wide Python packages (Override PEP 668)
RUN pip config set global.break-system-packages true
# Install ROS1
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
RUN apt update
RUN apt install ros-noetic-desktop-full -y
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential -y
RUN apt install python3-rosdep -y
RUN rosdep init
RUN rosdep update
# Install catkin for ROS1 package building
RUN apt-get install -y python3-catkin-tools

# Install additional system packages needed (if any)
RUN apt-get update && xargs -a /robot_learning/docker/packages.txt apt-get install -y

# Install required Python packages
RUN pip install --no-cache-dir -r /robot_learning/docker/python-packages.txt

# Set up entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh

# Ensure the script is executable
RUN chmod +x /entrypoint.sh
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Default command
ENTRYPOINT ["/entrypoint.sh"]