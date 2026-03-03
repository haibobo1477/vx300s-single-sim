import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def generate_launch_description():

    # =======================
    # Package paths
    # =======================
    
    moveit_share = get_package_share_directory("vx300s_moveit_config")
    desc_share = get_package_share_directory("vx300s_description")

    xacro_file = os.path.join(
        desc_share, "urdf", "vx300s_moveit.urdf.xacro"
    )

    move_group_launch = os.path.join(
        moveit_share, "launch", "move_group.launch.py"
    )

    default_rviz_config = os.path.join(
        moveit_share, "config", "moveit.rviz"
    )

    # =======================
    # Launch arguments
    # =======================
    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use sim time (false for non-Gazebo)",
    )

    declare_rviz = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="Start RViz2",
    )

    declare_rvizconfig = DeclareLaunchArgument(
        "rvizconfig",
        default_value=default_rviz_config,
        description="RViz config file",
    )

    # =======================
    # robot_description (xacro)
    # =======================
    robot_description_content = Command(
        ["xacro ", xacro_file]
    )

    robot_description = {
        "robot_description": robot_description_content
    }

    # =======================
    # robot_state_publisher
    # =======================
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[
            robot_description,
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    # =======================
    # ros2_control controller_manager
    # =======================
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            robot_description,
            os.path.join(
                moveit_share,
                "config",
                "ros2_controllers.yaml",
            ),
        ],
        output="screen",
    )

    # =======================
    # Controllers spawner
    # =======================
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "arm_controller",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    # 如果你有 gripper controller，可以打开
    gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "gripper_controller",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    # =======================
    # MoveIt move_group
    # =======================
    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(move_group_launch),
        launch_arguments={
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }.items(),
    )

    # =======================
    # RViz
    # =======================
    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", LaunchConfiguration("rvizconfig")],
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
        output="screen",
        condition=IfCondition(LaunchConfiguration("rviz")),
    )
    
    
    # =======================
    # LaunchDescription
    # =======================
    return LaunchDescription(
        [
            declare_use_sim_time,
            declare_rviz,
            declare_rvizconfig,

            robot_state_publisher,
            ros2_control_node,

            joint_state_broadcaster_spawner,
            arm_controller_spawner,
            gripper_controller_spawner,

            move_group,
            rviz2,
        ]
    )

