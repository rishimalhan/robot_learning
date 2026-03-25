# Use an LXDE+VNC base image (includes noVNC at port 8080)
FROM dorowu/ubuntu-desktop-lxde-vnc:focal-arm64

USER root

ARG DEBIAN_FRONTEND=noninteractive

ENV HF_HUB_DISABLE_XET=1

# Create workspace
WORKDIR /robot_learning

# Install system dependencies, ROS, and clean apt cache in one layer
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      python3 python3-pip \
      software-properties-common \
      curl \
      wget \
      gnupg2 \
      lsb-release \
      build-essential \
      git \
      cmake \
      iproute2 \
      iputils-ping \
      net-tools \
      novnc \
      x11vnc \
      python3-websockify; \
    mkdir -p /etc/apt/keyrings; \
    curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
      | gpg --dearmor -o /etc/apt/keyrings/ros-archive-keyring.gpg; \
    echo "deb [signed-by=/etc/apt/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
      > /etc/apt/sources.list.d/ros-latest.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ros-noetic-desktop-full \
      python3-rosdep \
      python3-rosinstall \
      python3-rosinstall-generator \
      python3-wstool \
      python3-catkin-tools; \
    if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then rosdep init; fi; \
    rosdep update; \
    rm -rf /var/lib/apt/lists/*

# Allow pip to manage system-wide Python packages if the distro enforces PEP 668
RUN python3 -m pip config set global.break-system-packages true || true

# Copy dependency manifests early for better caching
COPY docker/packages.txt /robot_learning/docker/packages.txt
COPY docker/python-packages.txt /robot_learning/docker/python-packages.txt

# Install additional system packages needed (if any)
RUN set -eux; \
    apt-get update; \
    if [ -s /robot_learning/docker/packages.txt ]; then \
      xargs -a /robot_learning/docker/packages.txt apt-get install -y --no-install-recommends; \
    fi; \
    rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN python3 -m pip install --no-cache-dir -r /robot_learning/docker/python-packages.txt

# Set up entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh

# Ensure the script is executable
RUN chmod +x /entrypoint.sh
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Default command
ENTRYPOINT ["/entrypoint.sh"]
