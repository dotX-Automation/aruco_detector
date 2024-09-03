"""
ArUco Detector launch file.

August 19, 2024
"""

# Copyright 2024 dotX Automation s.r.l.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    # Build config file path
    config = os.path.join(
        get_package_share_directory('aruco_detector'),
        'config',
        'aruco_detector.yaml'
    )

    # Create node launch description
    node = Node(
        package='aruco_detector',
        executable='aruco_detector_app',
        namespace='',
        emulate_tty=True,
        output='both',
        log_cmd=True,
        parameters=[config],
        remappings=[
            ('/aruco_detector/detections', '/aruco_detector/detections'),
            ('/aruco_detector/visual_targets', '/aruco_detector/visual_targets')
        ]
    )

    ld.add_action(node)

    return ld
