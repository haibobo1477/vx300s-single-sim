#!/usr/bin/env python3
import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import pinocchio as pin
import numpy as np

# ====== Pinocchio 模型构建 ======
urdf_model_path = "/home/haibo/vx300s_ws/src/vx300s_description/urdf/vx300s.urdf"
mesh_dir = "/home/haibo/vx300s_ws/src/vx300s_description/vx300s_meshes/"

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path,
    mesh_dir,
    root_joint=None
)
data = model.createData()

# 9 个关节（包括手爪）
target_joints = [
    "waist", "shoulder", "elbow",
    "forearm_roll", "wrist_angle", "wrist_rotate",
    "gripper", "left_finger", "right_finger"
]

def extract_joint_positions(msg, joint_list):
    """
    从 JointState 消息中提取指定顺序的关节位置
    - msg: JointState
    - joint_list: 需要提取的关节名顺序
    """
    name_to_pos = dict(zip(msg.name, msg.position))
    q_extracted = []
    for jname in joint_list:
        if jname in name_to_pos:
            q_extracted.append(name_to_pos[jname])
        else:
            q_extracted.append(0.0)  # 如果缺失该关节，补 0
    return np.array(q_extracted)


def joint_state_callback(msg, node, publisher):
    # 提取 9 个关节位置
    q_9 = extract_joint_positions(msg, target_joints)
    
    q_10 = np.insert(q_9, 0, 0.0)

    # 计算重力补偿力矩
    G = pin.computeGeneralizedGravity(model, data, q_10)

    # 打印关节信息
    node.get_logger().info("=== Joint States ===")
    for jname, qval, gval in zip(target_joints, q_9, G):
        node.get_logger().info(
            f"Joint: {jname:15s} | pos: {qval:.4f} rad | torque(G): {gval:.4f} Nm"
        )

    # 发布结果
    effort_msg = Float64MultiArray()
    effort_msg.data = G[:6].tolist()
    publisher.publish(effort_msg)


def main(args=None):
    rclpy.init(args=args)

    # 创建节点
    node = rclpy.create_node("gravity_comp_node")

    # 发布者
    effort_pub = node.create_publisher(Float64MultiArray, "/arm_controller/commands", 10)

    # 订阅者
    node.create_subscription(
        JointState,
        "/joint_states",
        lambda msg: joint_state_callback(msg, node, effort_pub),
        10
    )

    node.get_logger().info("Gravity compensation node started. Listening to /joint_states ...")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()































# #!/usr/bin/env python3
# import rclpy
# from sensor_msgs.msg import JointState
# from std_msgs.msg import Float64MultiArray

# import pinocchio as pin
# import numpy as np

# # ====== Pinocchio 模型构建 ======
# # urdf_model_path = "/home/yc/vx300s-single-sim/src/vx300s_description/urdf/vx300s.urdf"
# # mesh_dir = "/home/yc/vx300s-single-sim/src/vx300s_description/vx300s_meshes/"

# urdf_model_path = "/home/haibo/vx300s_ws/src/vx300s_description/urdf/vx300s.urdf"
# mesh_dir = "/home/haibo/vx300s_ws/src/vx300s_description/vx300s_meshes/"

# model, collision_model, visual_model = pin.buildModelsFromUrdf(
#     urdf_model_path,
#     mesh_dir,
#     root_joint=None
# )
# data = model.createData()


# # 6 个主要关节
# target_joints = [
#     "waist", "shoulder", "elbow",
#     "forearm_roll", "wrist_angle", "wrist_rotate",
#     "gripper", "left_finger", "right_finger"
# ]


# def extract_joint_positions(msg, joint_list):
#     """
#     从 JointState 消息中提取指定顺序的关节位置
#     - msg: JointState
#     - joint_list: 需要提取的关节名顺序
#     """
#     name_to_pos = dict(zip(msg.name, msg.position))
#     q_extracted = []
#     for jname in joint_list:
#         if jname in name_to_pos:
#             q_extracted.append(name_to_pos[jname])
#         else:
#             q_extracted.append(0.0)  # 如果缺失该关节，补 0
#     return np.array(q_extracted)


# def joint_state_callback(msg, node):
#     # 提取 9 个关节位置
#     q_9 = extract_joint_positions(msg, target_joints)

#     # 打印结果
#     node.get_logger().info("=== Extracted Joint Positions ===")
#     for jname, qval in zip(target_joints, q_9):
#         node.get_logger().info(f"Joint: {jname:15s} | position: {qval:.4f} rad")


# def main(args=None):
#     rclpy.init(args=args)

#     # 创建节点
#     node = rclpy.create_node("extract_joint_positions_node")

#     # 创建订阅者
#     node.create_subscription(
#         JointState,
#         "/joint_states",
#         lambda msg: joint_state_callback(msg, node),
#         10
#     )

#     node.get_logger().info("Joint position extractor node started. Listening to /joint_states ...")
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main()





















