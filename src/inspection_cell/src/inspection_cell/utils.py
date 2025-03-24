#! /usr/bin/env python3

import rospkg
import os


def resolve_package_path(pkg_uri):
    if not pkg_uri.startswith("package://"):
        raise ValueError("Not a package:// URI")

    # Strip the URI and get package + relative path
    path = pkg_uri[len("package://") :]
    pkg_name, rel_path = path.split("/", 1)

    # Get absolute path to the package
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(pkg_name)

    return os.path.join(pkg_path, rel_path)
