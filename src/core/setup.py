#!/usr/bin/env python3

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=["core"], package_dir={"": "src"}, install_requires=["rospy"]
)

setup(**setup_args)
