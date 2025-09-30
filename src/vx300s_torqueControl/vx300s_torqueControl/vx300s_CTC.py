#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import pinocchio as pin
import numpy as np

# ====== Pinocchio model ======
urdf_model_path = "/home/haibo/vx300s_ws/src/vx300s_description/urdf/vx300s_fix.urdf"
mesh_dir = "/home/haibo/vx300s_ws/src/vx300s_description/vx300s_meshes/"

# urdf_model_path = "/home/yc/vx300s-single-sim/src/vx300s_description/urdf/vx300s_fix.urdf"
# mesh_dir = "/home/yc/vx300s-single-sim/src/vx300s_description/vx300s_meshes/"



model, _, _ = pin.buildModelsFromUrdf(urdf_model_path, package_dirs=[mesh_dir])

# set gravity
model.gravity = pin.Motion.Zero()
model.gravity.linear = np.array([0.0, 0.0, -9.81])

data = model.createData()

# 6 joints
target_joints = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
assert model.nv == len(target_joints), f"model.nv={model.nv} 与目标关节数 {len(target_joints)} 不一致"

def extract_by_name(msg_names, msg_vals, wanted_names):
    m = dict(zip(msg_names, msg_vals)) if msg_vals is not None else {}
    return np.array([m.get(n, 0.0) for n in wanted_names], dtype=float)

class GravityCompNode(Node):
    def __init__(self):
        super().__init__("gravity_comp_node")
        
        
        self.start_time = self.get_clock().now().nanoseconds * 1e-9


        # ---- parameter----
        self.D_visc = np.array([0.6, 0.8, 0.6, 0.15, 0.12, 0.12], dtype=float)
        # tau limitation
        self.tau_limit = np.array([10, 10, 10, 5, 4, 4], dtype=float)

        # PD gain
        self.Kp = np.diag([50, 50, 40, 20, 10, 5])
        self.Kd = np.diag([5,  5,  4,  2,  1, 0.5])

        # reference
        self.q_des   = np.zeros(6)
        self.qd_des  = np.zeros(6)
        self.qdd_des = np.zeros(6)

        # pub/sub
        self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 50)
        self.tau_pub = self.create_publisher(Float64MultiArray, "/arm_controller/commands", 10)

    def joint_state_callback(self, msg: JointState):
        
        # time now
        t = self.get_clock().now().nanoseconds * 1e-9 - self.start_time

    # --------- ref trajtory ---------
        
        # --------- ref trajectory ---------
        A = [0.0, 0.0, 0.0, 0.0, 0.0, 90.5]  
        w = [0.5, 0.5, 0.5, 0.5, 0.5, 5.5]   

        q_des   = np.zeros(6)
        qd_des  = np.zeros(6)
        qdd_des = np.zeros(6)

        for i in range(6):
            q_des[i]   = A[i] * np.sin(w[i] * t)
            qd_des[i]  = A[i] * w[i] * np.cos(w[i] * t)
            qdd_des[i] = - A[i] * (w[i]**2) * np.sin(w[i] * t)

        self.q_des   = q_des
        self.qd_des  = qd_des
        self.qdd_des = qdd_des
        
        # current joint states
        q  = extract_by_name(msg.name, msg.position, target_joints)
        dq = extract_by_name(msg.name, msg.velocity, target_joints) 

        # error
        e    = self.q_des  - q
        edot = self.qd_des - dq

        # reference acc
        qdd_ref = self.qdd_des + self.Kp.dot(e) + self.Kd.dot(edot)

        #  M(q)qdd_ref + C(q,dq)dq + G(q)
        tau = pin.rnea(model, data, q, dq, qdd_ref)

        tau -= self.D_visc * dq

        tau = np.clip(tau, -self.tau_limit, self.tau_limit)
        
        self.get_logger().info(f"t={t:.2f}, q_des={self.q_des}, q={q}")

        # pub
        out = Float64MultiArray()
        out.data = tau.tolist()
        self.tau_pub.publish(out)

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


































# import pinocchio as pin
# from pinocchio.robot_wrapper import RobotWrapper
# from pinocchio.visualize import MeshcatVisualizer
# import numpy as np
# import time
# import meshcat
# from pinocchio import FrameType



# import pinocchio as pin
# import numpy as np
# from pinocchio import FrameType



# # 你的 URDF 路径
# # urdf_model_path = "/home/yc/vx300s-single-sim/src/vx300s_description/urdf/vx300s.urdf"
# # mesh_dir = "/home/yc/vx300s-single-sim/src/vx300s_description/vx300s_meshes/"


# urdf_model_path = "/home/haibo/vx300s_ws/src/vx300s_description/urdf/vx300s_fix.urdf"
# mesh_dir = "/home/haibo/vx300s_ws/src/vx300s_description/vx300s_meshes/"

# # 构建模型
# # robot = RobotWrapper.BuildFromURDF(urdf_model_path, mesh_dir)
# model, collision_model, visual_model = pin.buildModelsFromUrdf(
#     urdf_model_path,
#     mesh_dir
# )


# print(model.gravity)

# data = model.createData()
# print(data)



# # q = pin.randomConfiguration(model)   # 随机一个姿态
# # q = np.array([0, 0, 0, 0, 0, 0, 0.8, 0, 0, 0]) 
# # print(q)
# q = pin.neutral(model)
# # print(q)
# v = np.random.randn(model.nv)        # 随机速度
# # print(v)



# masses = {}
# for fid, frame in enumerate(model.frames):
#     if frame.type == FrameType.BODY:
#         jidx = frame.parent            # 该BODY隶属的joint索引
#         I = model.inertias[jidx]       # 该link的惯性
#         masses[frame.name] = float(I.mass)

# # 打印
# total_mass = 0.0
# for name, m in masses.items():
#     print(f"{name:30s}  mass = {m:.6f} kg")
#     total_mass += m
# print(f"\nTotal mass = {total_mass:.6f} kg")


# # 可视化器
# # viz = MeshcatVisualizer(model, collision_model, visual_model)

# # viz.initViewer(open=True)       # 打开浏览器窗口
# # viz.loadViewerModel()           # 加载模型
# # viz.display(q) 
# # viz.viewer.open()


# M = pin.crba(model, data, q)   # Composite Rigid Body Algorithm
# M = (M + M.T) / 2.0            # 数值对称化（避免小误差）

# G = pin.computeGeneralizedGravity(model, data, q)

# C = pin.computeCoriolisMatrix(model, data, q, v)
# # print(M.shape)
# # print(C.shape)
# # print(G)
# # print(q)
# print(model)