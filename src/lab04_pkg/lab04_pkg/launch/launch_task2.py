from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lab04_pkg',          # name of your ROS 2 package
            executable='task2_node',  # name of the entry point from setup.py
            name='task2_node',             # optional: node name override
            output='screen',
            
        )
    ])
