#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import pinocchio as pin
import numpy as np
import math
from time import monotonic


# urdf_model_path = "/home/haibo/vx300s_ws/src/vx300s_description/urdf/vx300s_fix.urdf"
# mesh_dir = "/home/haibo/vx300s_ws/src/vx300s_description/vx300s_meshes/"


urdf_model_path = "/home/yc/vx300s-single-sim/src/vx300s_description/urdf/vx300s_fix.urdf"
mesh_dir = "/home/yc/vx300s-single-sim/src/vx300s_description/vx300s_meshes/"

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path,
    package_dirs=[mesh_dir]
)


custom_gravity = pin.Motion(np.array([0, 0, -9.80, 0, 0, 0]))
model.gravity = custom_gravity
data = model.createData()


target_joints = ["waist", "shoulder", "elbow",
                 "forearm_roll", "wrist_angle", "wrist_rotate"]

assert model.nv == len(target_joints), \
    f"model.nv={model.nv} 与 target_joints 数量 {len(target_joints)} 不一致，请检查。"

def extract_by_name(msg_names, msg_vals, wanted_names):
    m = dict(zip(msg_names, msg_vals)) if msg_vals is not None else {}
    out = []
    for n in wanted_names:
        out.append(m.get(n, 0.0))
    return np.array(out, dtype=float)



class GravityCompNode(Node):
    def __init__(self):
        super().__init__("gravity_comp_node")

        self.damping = np.array([0.6, 0.8, 0.6, 0.15, 0.12, 0.12], dtype=float)

     
        self.create_subscription(JointState, "/joint_states",
                                 self.joint_state_callback, 50)
        self.tau_pub = self.create_publisher(Float64MultiArray,
                                             "/arm_controller/commands", 10)

        self.get_logger().info("Gravity + Damping compensation started.")

    def joint_state_callback(self, msg: JointState):
       
        q = extract_by_name(msg.name, msg.position, target_joints)
        # print(q)
        
        dq = extract_by_name(msg.name, msg.velocity, target_joints)
        # print(dq)


       
        tau_g = pin.rnea(model, data, q,
                         np.zeros(model.nv),  # dq=0
                         np.zeros(model.nv))  # ddq=0

        tau = tau_g - self.damping * dq


        print(tau_g)
        # --------- publisher ---------
        msg_out = Float64MultiArray()
        msg_out.data = tau.tolist()
        self.tau_pub.publish(msg_out)

def main(args=None):
    rclpy.init(args=args)
    node = GravityCompNode()
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
# from rclpy.node import Node

# from sensor_msgs.msg import JointState
# from std_msgs.msg import Float64MultiArray

# import pinocchio as pin
# import numpy as np

# # ====== Pinocchio 模型构建 ======
# # # urdf_model_path = "/home/yc/vx300s-single-sim/src/vx300s_description/urdf/vx300s.urdf"
# # # mesh_dir = "/home/yc/vx300s-single-sim/src/vx300s_description/vx300s_meshes/"

# urdf_model_path = "/home/haibo/vx300s_ws/src/vx300s_description/urdf/vx300s_fix.urdf"
# mesh_dir = "/home/haibo/vx300s_ws/src/vx300s_description/vx300s_meshes/"

# model, collision_model, visual_model = pin.buildModelsFromUrdf(
#     urdf_model_path,
#     package_dirs=[mesh_dir]
# )

# custom_gravity = pin.Motion(np.array([0, 0, -9.80, 0, 0, 0])) # Example for Mars gravity
# model.gravity = custom_gravity

# data = model.createData()

# # 9 个主要关节（ROS 里顺序）
# target_joints = [
#     "waist", "shoulder", "elbow",
#     "forearm_roll", "wrist_angle", "wrist_rotate"
# ]




# def extract_joint_positions(msg, joint_list):
#     """从 JointState 消息中提取 9 个关节值"""
#     name_to_pos = dict(zip(msg.name, msg.position))
#     q_extracted = []
#     for jname in joint_list:
#         if jname in name_to_pos:
#             q_extracted.append(name_to_pos[jname])
#         else:
#             q_extracted.append(0.0)  # 缺失补 0
#     return np.array(q_extracted)




# class GravityCompNode(Node):
#     def __init__(self):
#         super().__init__("gravity_comp_node")

#         # 订阅 joint_states
#         self.create_subscription(
#             JointState,
#             "/joint_states",
#             self.joint_state_callback,
#             10
#         )

#         # 发布重力补偿力矩
#         self.tau_pub = self.create_publisher(Float64MultiArray, "/arm_controller/commands", 10)

#         # self.get_logger().info("Gravity compensation node started ✅")

#     def joint_state_callback(self, msg):
#         # 提取 6 维
#         q_6 = extract_joint_positions(msg, target_joints)
#         print(q_6)

    
#         # 计算重力补偿项
#         # G = pin.computeGeneralizedGravity(model, data, q_6)
#         tau_g = pin.rnea(model, data, q_6, 
#                  np.zeros(model.nv),   # dq=0
#                  np.zeros(model.nv))   # ddq=0
        
        
#         # # 发布
#         tau_msg = Float64MultiArray()
#         tau_msg.data = tau_g.tolist()
#         self.tau_pub.publish(tau_msg)

#         # self.get_logger().info(f"G(q) = {np.round(G, 3)}")


# def main(args=None):
#     rclpy.init(args=args)
#     node = GravityCompNode()

#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main()









# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node

# from sensor_msgs.msg import JointState
# from std_msgs.msg import Float64MultiArray

# import pinocchio as pin
# import numpy as np

# # ====== Pinocchio 模型构建 ======
# # # urdf_model_path = "/home/yc/vx300s-single-sim/src/vx300s_description/urdf/vx300s.urdf"
# # # mesh_dir = "/home/yc/vx300s-single-sim/src/vx300s_description/vx300s_meshes/"

# urdf_model_path = "/home/haibo/vx300s_ws/src/vx300s_description/urdf/vx300s.urdf"
# mesh_dir = "/home/haibo/vx300s_ws/src/vx300s_description/vx300s_meshes/"

# model, collision_model, visual_model = pin.buildModelsFromUrdf(
#     urdf_model_path,
#     package_dirs=[mesh_dir]
# )


# data = model.createData()

# # 9 个主要关节（ROS 里顺序）
# target_joints = [
#     "waist", "shoulder", "elbow",
#     "forearm_roll", "wrist_angle", "wrist_rotate",
#     "gripper", "left_finger", "right_finger"
# ]


# def extract_joint_positions(msg, joint_list):
#     """从 JointState 消息中提取 9 个关节值"""
#     name_to_pos = dict(zip(msg.name, msg.position))
#     q_extracted = []
#     for jname in joint_list:
#         if jname in name_to_pos:
#             q_extracted.append(name_to_pos[jname])
#         else:
#             q_extracted.append(0.0)  # 缺失补 0
#     return np.array(q_extracted)


# def ros_to_pinocchio_q(q_ros, model):
#     """
#     将 ROS 的 9x1 转换为 Pinocchio 的 10x1
#     - 前 6 个关节直接复制
#     - gripper: 用 SO(2) [cosθ, sinθ]
#     - left/right finger: 直接复制
#     """
#     q_pin = pin.neutral(model).copy()

#     q_pin[0:6] = q_ros[0:6]

#     theta = q_ros[6]   # gripper
#     q_pin[6] = np.cos(theta)
#     q_pin[7] = np.sin(theta)

#     q_pin[8] = q_ros[7]   # left_finger
#     q_pin[9] = q_ros[8]   # right_finger

#     q_pin = pin.normalize(model, q_pin)
#     return q_pin


# class GravityCompNode(Node):
#     def __init__(self):
#         super().__init__("gravity_comp_node")

#         # 订阅 joint_states
#         self.create_subscription(
#             JointState,
#             "/joint_states",
#             self.joint_state_callback,
#             10
#         )

#         # 发布重力补偿力矩
#         self.tau_pub = self.create_publisher(Float64MultiArray, "/arm_controller/commands", 10)

#         # self.get_logger().info("Gravity compensation node started ✅")

#     def joint_state_callback(self, msg):
#         # 提取 9 维
#         q_9 = extract_joint_positions(msg, target_joints)

#         # 转换成 10 维
#         q_10 = ros_to_pinocchio_q(q_9, model)
#         # print(q_10)

#         # 计算重力补偿项
#         G = pin.computeGeneralizedGravity(model, data, q_10)
#         # tau_g = pin.rnea(model, data, q_10, 
#         #          np.zeros(model.nv),   # dq=0
#         #          np.zeros(model.nv))   # ddq=0
        
#         G6 = G[0:6]

#         # 发布
#         tau_msg = Float64MultiArray()
#         tau_msg.data = G6.tolist()
#         self.tau_pub.publish(tau_msg)

#         self.get_logger().info(f"G(q) = {np.round(G6, 3)}")


# def main(args=None):
#     rclpy.init(args=args)
#     node = GravityCompNode()

#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main()












