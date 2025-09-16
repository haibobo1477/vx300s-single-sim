import launch
import launch.event_handlers
import launch.launch_description_sources
import launch_ros
from ament_index_python.packages import get_package_share_directory
import os

import launch_ros.parameter_descriptions



from pathlib import Path

import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import launch_ros.parameter_descriptions



from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)



def generate_launch_description():
    # 包路径
    urdf_package_path = get_package_share_directory('vx300s_description')
    default_xacro_path = os.path.join(urdf_package_path, 'urdf', 'vx300s.urdf.xacro')
    default_gazebo_world_path = os.path.join(urdf_package_path, 'world', 'custom_room.world')

    # 参数: 机器人模型
    action_declare_arg_model_path = DeclareLaunchArgument(
        name='model',
        default_value=str(default_xacro_path),
        description='加载模型文件.'
    )

    # 用 xacro 生成 robot_description
    substitutions_command_result = launch.substitutions.Command(
        ['xacro ', LaunchConfiguration('model')]
    )
    robot_description_value = launch_ros.parameter_descriptions.ParameterValue(
        substitutions_command_result, value_type=str
    )

    # robot_state_publisher
    action_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description_value}]
    )

    # Gazebo 启动
    action_launch_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [get_package_share_directory('gazebo_ros'), '/launch', '/gazebo.launch.py']
        ),
        launch_arguments={
            'world': default_gazebo_world_path,
            'verbose': 'true'
        }.items()
    )

    # spawn 机器人到 Gazebo
    action_spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', '/robot_description', '-entity', 'vx300s'],
        output='screen'
    )
    
    action_load_joint_state_controller = launch.actions.ExecuteProcess(
        cmd='ros2 control load_controller joint_state_broadcaster --set-state active'.split(' '),
        output='screen'
    )
    
    
    action_load_effort_controller = launch.actions.ExecuteProcess(
        cmd='ros2 control load_controller arm_controller --set-state active'.split(' '),
        output='screen'
    )

    # Gazebo 模型路径
    # gz_resource_path_env_var = SetEnvironmentVariable(
    #     name='GAZEBO_MODEL_PATH',
    #     value=[
    #         EnvironmentVariable('GAZEBO_MODEL_PATH', default_value=''),
    #         '/usr/share/gazebo-11/models:',
    #         str(Path(FindPackageShare('vx300s_description').find('vx300s_description')).parent.resolve())
    #     ]
    # )
 

  
    return LaunchDescription([
        action_declare_arg_model_path,
        # gz_resource_path_env_var,
        action_robot_state_publisher,
        action_launch_gazebo,
        action_spawn_entity,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=action_spawn_entity,
                on_exit=[action_load_joint_state_controller],
            )
        ),
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=action_spawn_entity,
                on_exit=[action_load_effort_controller],
            )
        ),
    ])
