from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    xacro_file = PathJoinSubstitution([FindPackageShare('vx300s_description'), 'urdf', 'vx300s.xacro'])
    robot_description = Command([FindExecutable(name='xacro'), ' ', xacro_file])
    # 找到 RViz 配置文件（假设放在包的 config 目录里）
    rviz_config_file = PathJoinSubstitution([
        FindPackageShare('vx300s_description'),
        'config',
        'vx300s_display.rviz'
    ])

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            output='screen'
        ),
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],  # 指定配置文件
            output='screen'
        )
    ])