from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='nodes',
            executable='node1',
            name='node1',
        ),
        Node(
            package='nodes',
            executable='node2',
            name='node2'
        )
    ])
