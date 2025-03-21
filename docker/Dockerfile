# Use ROS1 Noetic base image
FROM --platform=linux/arm64 osrf/ros:noetic-desktop-full

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
    net-tools && \
    rm -rf /var/lib/apt/lists/*

# Make sure curl is available before downloading keys
RUN apt-get install -y curl

# Add ROS key (ONLY IF NOT ALREADY PRESENT)
RUN if [ ! -f "/usr/share/keyrings/ros-archive-keyring.gpg" ]; then \
        curl -sSL 'https://raw.githubusercontent.com/ros/rosdistro/master/ros.key' | apt-key add -; \
    fi

# Remove conflicting sources list if they exist
RUN rm -f /etc/apt/sources.list.d/ros-latest.list /etc/apt/sources.list.d/ros2.list /etc/apt/sources.list.d/ros1.list

# Update and upgrade system packages safely
RUN apt-get update && apt-get upgrade -y && apt-get autoremove -y

# Initialize rosdep
RUN rosdep init || true && rosdep update

# Allow pip to manage system-wide Python packages (Override PEP 668)
RUN pip config set global.break-system-packages true

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