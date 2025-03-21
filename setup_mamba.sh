mamba create -n robot_learning python=3.11
mamba activate robot_learning
conda config --env --add channels conda-forge
conda config --env --remove channels defaults
conda config --env --add channels robostack-noetic
mamba install ros-noetic-desktop-full
mamba deactivate
mamba activate robot_learning
mamba install compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep
pip install -r docker/python-packages.txt