from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('quarry_drone_vlm')
    bridge_cfg = os.path.join(pkg_share, 'config', 'bridge_config.yaml')

    return LaunchDescription([
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='quarry_ros_gz_bridge',
            parameters=[{'config_file': bridge_cfg}],
            output='screen'
        ),
        Node(
            package='quarry_drone_vlm',
            executable='mission_node',
            name='quarry_mission_node',
            output='screen'
        ),
    ])
