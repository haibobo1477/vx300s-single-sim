#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import pinocchio as pin
import numpy as np
import math
from time import monotonic

# ====== Pinocchio 模型构建 ======
urdf_model_path = "/home/haibo/vx300s_ws/src/vx300s_description/urdf/vx300s_fix.urdf"
mesh_dir = "/home/haibo/vx300s_ws/src/vx300s_description/vx300s_meshes/"

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path,
    package_dirs=[mesh_dir]
)

# 自定义重力（保持和你原来一致）
custom_gravity = pin.Motion(np.array([0, 0, -9.80, 0, 0, 0]))
model.gravity = custom_gravity
data = model.createData()

# 6 个主要关节（顺序要与控制器/JointState一致）
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

        # --------------- 可调参数 ---------------
        # 关节粘性阻尼系数（Nm/(rad/s)），按顺序对应 target_joints
        self.damping = np.array([0.6, 0.8, 0.6, 0.15, 0.12, 0.12], dtype=float)
        # 速度一阶低通滤波截止频率（Hz），抑制编码器噪声
        self.vel_cutoff_hz = 8.0
        # 每个关节的扭矩限幅（Nm），按需调整；若不清楚，先给个保守值
        self.torque_limits = np.array([6.0, 8.0, 6.0, 1.5, 1.0, 1.0], dtype=float)

        # 速度滤波状态
        self.v_filt = np.zeros(model.nv)
        self.last_stamp_sec = None  # 从 JointState.header 计算 dt
        self.fallback_monotonic = monotonic()

        # 订阅与发布
        self.create_subscription(JointState, "/joint_states",
                                 self.joint_state_callback, 50)
        self.tau_pub = self.create_publisher(Float64MultiArray,
                                             "/arm_controller/commands", 10)

        self.get_logger().info("Gravity + Damping compensation started.")

    def joint_state_callback(self, msg: JointState):
        # --------- 提取 q, dq ---------
        q = extract_by_name(msg.name, msg.position, target_joints)
        # 有些驱动不会发布 velocity，就用 0 或自行差分
        dq_raw = extract_by_name(msg.name, msg.velocity, target_joints)

        # --------- 计算 dt（用于滤波） ---------
        if msg.header.stamp.sec or msg.header.stamp.nanosec:
            stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if self.last_stamp_sec is None:
                dt = 0.0
            else:
                dt = max(1e-4, min(0.1, stamp_sec - self.last_stamp_sec))
            self.last_stamp_sec = stamp_sec
        else:
            # 万一 header 没时间戳，用 monotonic 做一个
            now = monotonic()
            dt = max(1e-4, min(0.1, now - self.fallback_monotonic))
            self.fallback_monotonic = now

        # --------- 速度低通滤波（单极点一阶） ---------
        # y[k] = y[k-1] + alpha*(x[k]-y[k-1]), alpha = 1 - exp(-2π fc dt)
        if dt > 0.0:
            alpha = 1.0 - math.exp(-2.0 * math.pi * self.vel_cutoff_hz * dt)
        else:
            alpha = 1.0
        self.v_filt = self.v_filt + alpha * (dq_raw - self.v_filt)
        dq = self.v_filt

        # --------- 重力项 ---------
        tau_g = pin.rnea(model, data, q,
                         np.zeros(model.nv),  # dq=0
                         np.zeros(model.nv))  # ddq=0

        # --------- 阻尼项（粘性阻尼） ---------
        tau_damp = self.damping * dq

        # 总扭矩：重力补偿 - 阻尼（阻尼是“耗能”，与速度同向时减小输出）
        tau = tau_g - tau_damp
        print(tau)

        # --------- 限幅与数值健壮性 ---------
        # tau = np.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
        # tau = np.clip(tau, -self.torque_limits, self.torque_limits)

        # --------- 发布 ---------
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












