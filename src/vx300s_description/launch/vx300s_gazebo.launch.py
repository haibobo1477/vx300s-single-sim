from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # 1. Xacro -> robot_description
    xacro_file = PathJoinSubstitution([
        FindPackageShare('vx300s_description'),
        'urdf',
        'vx300s.xacro'
    ])
    robot_description = Command([FindExecutable(name='xacro'), ' ', xacro_file])

    # 2. RViz 配置
    rviz_config_file = PathJoinSubstitution([
        FindPackageShare('vx300s_description'),
        'config',
        'vx300s_display.rviz'
    ])

    # 3. Gazebo world 文件 (你保存的 world 放在 vx300s_description/worlds 目录里)
    world_file = PathJoinSubstitution([
        FindPackageShare('vx300s_description'),
        'world',
        'custom_room.world'   # ⚠️ 这里替换成你保存的 world 名字
    ])

    return LaunchDescription([
        # robot_state_publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            output='screen'
        ),

        # joint_state_publisher
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            output='screen'
        ),

        # RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            output='screen'
        ),

        # Gazebo (加载 world)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                FindPackageShare('gazebo_ros'),
                '/launch',
                '/gazebo.launch.py'
            ]),
            launch_arguments={'world': world_file}.items()
        ),

        # Spawn robot into Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-topic', 'robot_description',
                '-entity', 'vx300s'
            ],
            output='screen'
        )
    ])
