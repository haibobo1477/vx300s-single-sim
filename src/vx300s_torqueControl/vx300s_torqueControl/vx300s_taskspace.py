# # 1. 获取当前状态
# q, dq = joint_states()

# # 2. 正运动学 & Jacobian
# X = f(q)
# J = computeJacobian(q)
# Jdot_dq = computeJdotTimesV(q, dq)

# # 3. 生成末端参考轨迹
# Xd, Xd_dot, Xd_ddot = trajectory_generator(t)

# # 4. 任务空间控制律
# aX = Xd_ddot + Kp @ (Xd - X) + Kd @ (Xd_dot - J@dq)

# # 5. 转换到关节空间
# aq = J_pinv @ (aX - Jdot_dq)

# # 6. Inverse Dynamics
# tau = M(q) @ aq + C(q,dq)@dq + G(q)

# # 7. 发力矩命令
# publish(tau)


#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import pinocchio as pin
import numpy as np
import modern_robotics as mr
import time


# ----------------------------------
Slist = np.array([
    [0.0, 0.0, 1.0,  0.0,      0.0,     0.0],
    [0.0, 1.0, 0.0, -0.12705,  0.0,     0.0],
    [0.0, 1.0, 0.0, -0.42705,  0.0,     0.05955],
    [1.0, 0.0, 0.0,  0.0,      0.42705, 0.0],
    [0.0, 1.0, 0.0, -0.42705,  0.0,     0.35955],
    [1.0, 0.0, 0.0,  0.0,      0.42705, 0.0]
]).T

M = np.array([
    [1.0, 0.0, 0.0, 0.536494],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.42705],
    [0.0, 0.0, 0.0, 1.0]
])


# ----------------- Pinocchio model -----------------
urdf_model_path = "/home/haibo/vx300s_ws/src/vx300s_description/urdf/vx300s_fix.urdf"
mesh_dir = "/home/haibo/vx300s_ws/src/vx300s_description/vx300s_meshes/"


# urdf_model_path = "/home/yc/vx300s-single-sim/src/vx300s_description/urdf/vx300s_fix.urdf"
# mesh_dir = "/home/yc/vx300s-single-sim/src/vx300s_description/vx300s_meshes/"

model, _, _ = pin.buildModelsFromUrdf(urdf_model_path, package_dirs=[mesh_dir])

# 
model.gravity = pin.Motion.Zero()
model.gravity.linear = np.array([0.0, 0.0, -9.81])
data = model.createData()

# 
target_joints = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
assert model.nv == len(target_joints), f"model.nv={model.nv} 与目标关节数 {len(target_joints)} 不一致"


# -----------------  -----------------
def extract_by_name(msg_names, msg_vals, wanted_names):
    m = dict(zip(msg_names, msg_vals)) if msg_vals is not None else {}
    return np.array([m.get(n, 0.0) for n in wanted_names], dtype=float)




def trajectory_generator(t, T_total=10.0):
    # 
    X_start = np.array([0.2, 0.4, 0.4])
    X_end   = np.array([0.6, 0.4, 0.4])

    # 
    t_clip = np.clip(t, 0, T_total)
    tau = t_clip / T_total

    # ----  ----
    s      = 10*tau**3 - 15*tau**4 + 6*tau**5
    s_dot  = (30*tau**2 - 60*tau**3 + 30*tau**4) / T_total
    s_ddot = (60*tau - 180*tau**2 + 120*tau**3) / (T_total**2)

    # ----  ----
    pos = (1 - s) * X_start + s * X_end
    vel = s_dot  * (X_end - X_start)
    acc = s_ddot * (X_end - X_start)

    # ----  ----
    rot_vec = np.zeros(3)
    omega_const  = np.zeros(3)
    domega_const = np.zeros(3)

    # ---- 6D ----
    Xd    = np.hstack([rot_vec, pos])
    Xd_d  = np.hstack([omega_const, vel])
    Xd_dd = np.hstack([domega_const, acc])

    return Xd, Xd_d, Xd_dd



# ----------------- ROS2 -----------------
class GravityCompNode(Node):
    def __init__(self):
        super().__init__("gravity_comp_node")
        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        # pub/sub
        self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 50)
        self.tau_pub = self.create_publisher(Float64MultiArray, "/arm_controller/commands", 10)

        # 
        self.Kp = np.diag([60.0, 60.0, 60.0, 40.0, 30.0, 20.0])
        self.Kd = np.diag([50.0,   50.0,   50.0,   10.0,  10.0,  10.0])

        self.tau_limit = np.array([10, 10, 10, 5, 4, 4], dtype=float)

        # URDF in frame 
        self.ee_name = "vx300s/ee_gripper_link"
        self.ee_id = model.getFrameId(self.ee_name)

    def joint_state_callback(self, msg: JointState):
       
        t = self.get_clock().now().nanoseconds * 1e-9 - self.start_time

        q = extract_by_name(msg.name, msg.position, target_joints)
        dq = extract_by_name(msg.name, msg.velocity, target_joints)
        a_zero = np.zeros_like(dq)

       
        pin.forwardKinematics(model, data, q, dq, a_zero)
        pin.updateFramePlacements(model, data)
        oMf = data.oMf[self.ee_id]
        Xp = oMf.translation
        X = np.hstack([np.zeros(3), Xp])


       
        J = mr.JacobianSpace(Slist, q)   # [w v]
        # Jv = J[3:6, :]

        a_cl = pin.getFrameClassicalAcceleration(model, data, self.ee_id, pin.LOCAL_WORLD_ALIGNED)
        Jdot_qdot = np.hstack([a_cl.angular, a_cl.linear])

        Xd, Xd_d, Xd_dd = trajectory_generator(t)

       
        e = Xd - X
        v = J @ dq
        edot = Xd_d - v
        aX = Xd_dd + self.Kp @ e + self.Kd @ edot

       

        J_pinv = np.linalg.pinv(J)
        aq = J_pinv @ (aX - Jdot_qdot)

        
        tau = pin.rnea(model, data, q, dq, aq)
        
        time.sleep(0.05)
        

        
        out = Float64MultiArray()
        out.data = tau.tolist()
        self.tau_pub.publish(out)

        
        self.get_logger().info(f"t={t:.2f}, pos={X}, Xd={Xd}, e={e}")


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
# import modern_robotics as mr



# Slist = np.array([[0.0, 0.0, 1.0,  0.0,     0.0,     0.0],
#                   [0.0, 1.0, 0.0, -0.12705, 0.0,     0.0],
#                   [0.0, 1.0, 0.0, -0.42705, 0.0,     0.05955],
#                   [1.0, 0.0, 0.0,  0.0,     0.42705, 0.0],
#                   [0.0, 1.0, 0.0, -0.42705, 0.0,     0.35955],
#                   [1.0, 0.0, 0.0,  0.0,     0.42705, 0.0]]).T


# M = np.array([[1.0, 0.0, 0.0, 0.536494],
#               [0.0, 1.0, 0.0, 0.0],
#               [0.0, 0.0, 1.0, 0.42705],
#               [0.0, 0.0, 0.0, 1.0]])



# # ====== Pinocchio model ======
# urdf_model_path = "/home/haibo/vx300s_ws/src/vx300s_description/urdf/vx300s_fix.urdf"
# mesh_dir = "/home/haibo/vx300s_ws/src/vx300s_description/vx300s_meshes/"

# # urdf_model_path = "/home/yc/vx300s-single-sim/src/vx300s_description/urdf/vx300s_fix.urdf"
# # mesh_dir = "/home/yc/vx300s-single-sim/src/vx300s_description/vx300s_meshes/"



# model, _, _ = pin.buildModelsFromUrdf(urdf_model_path, package_dirs=[mesh_dir])

# # set gravity
# model.gravity = pin.Motion.Zero()
# model.gravity.linear = np.array([0.0, 0.0, -9.81])

# data = model.createData()

# # 6 joints
# target_joints = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
# assert model.nv == len(target_joints), f"model.nv={model.nv} 与目标关节数 {len(target_joints)} 不一致"

# def extract_by_name(msg_names, msg_vals, wanted_names):
#     m = dict(zip(msg_names, msg_vals)) if msg_vals is not None else {}
#     return np.array([m.get(n, 0.0) for n in wanted_names], dtype=float)


# def trajectory_generator(t):
#     # 直线
#     X_start = np.array([0.2, -0.4, 0.4])
#     X_end   = np.array([0.2,  0.4, 0.4])
#     w = 0.5
#     s = 0.5 * (1 - np.cos(w*t))
#     s_dot = 0.5 * w * np.sin(w*t)
#     s_ddot = 0.5 * w**2 * np.cos(w*t)
#     Xd   = (1 - s) * X_start + s * X_end
#     Xd_d = s_dot  * (X_end - X_start)
#     Xd_dd= s_ddot * (X_end - X_start)

#     # 姿态：绕 z 轴旋转
#     theta = 0.2 * np.sin(0.5*t)         # 角度
#     omega_d = np.array([0, 0, 0.1*np.cos(0.5*t)])  # 角速度
#     alpha_d = np.array([0, 0, -0.05*np.sin(0.5*t)]) # 角加速度

#     return Xd, Xd_d, Xd_dd, omega_d, alpha_d



# class GravityCompNode(Node):
#     def __init__(self):
#         super().__init__("gravity_comp_node")
        
        
#         self.start_time = self.get_clock().now().nanoseconds * 1e-9

#         # pub/sub
#         self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 50)
#         self.tau_pub = self.create_publisher(Float64MultiArray, "/arm_controller/commands", 10)
        
        
#         # self.Kp = np.diag([50.0, 50.0, 50.0])
#         # self.Kd = np.diag([10.0, 10.0, 10.0])
        
#         self.Kp = np.block([[200*np.eye(3), np.zeros((3,3))],
#                             [np.zeros((3,3)), 200*np.eye(3)]
#                             ])
#         self.Kd = np.block([[50*np.eye(3), np.zeros((3,3))],
#                             [np.zeros((3,3)), 50*np.eye(3)]
#                             ])
        
        
#         self.ee_name = "vx300s/ee_gripper_link"   # URDF 里定义的末端 frame 名字
#         self.ee_id = model.getFrameId(self.ee_name)

#     def joint_state_callback(self, msg: JointState):
        
#         # time now
#         t = self.get_clock().now().nanoseconds * 1e-9 - self.start_time

#         q  = extract_by_name(msg.name, msg.position, target_joints)
#         dq = extract_by_name(msg.name, msg.velocity, target_joints) 
        
#         a_zero = np.zeros_like(dq)
        
#         pin.forwardKinematics(model, data, q, dq, a_zero)
#         pin.updateFramePlacements(model, data)
        
#         oMf = data.oMf[self.ee_id]
#         X = oMf.translation 
#         R = oMf.rotation
        
        
#         J = mr.JacobianSpace(Slist, q)
#         Jv = J[3:6, :]
#         Jw = J[0:3, :]
        
#         a_cl = pin.getFrameClassicalAcceleration(model, data, self.ee_id,
#                                                  pin.LOCAL_WORLD_ALIGNED)
        
#         Jdot_qdot = np.hstack([a_cl.angular, a_cl.linear])
        
        
#         Xd, Xd_d, Xd_dd, omega_d, alpha_d = trajectory_generator(t)
        
        
#         # print(a_cl.linear)
#         # print(a_cl.angular)
#         # Jdot_qdot = np.hstack([a_cl.angular, a_cl.linear])
#         # print(Jdot_qdot)   # [w v]
#         # print(a_cl)
        
#         v = Jv @ dq
#         w = Jw @ dq
        
#         e_p = Xd - X
#         e_o = omega_d - w
        
#         e = np.hstack([e_o, e_p])
        
        
#         edot = np.hstack([alpha_d - (J[0:3,:] @ dq),
#                           Xd_d   - v])
        

#         # aX = Xd_dd + self.Kp @ e + self.Kd @ edot
        
#         aX = np.hstack([alpha_d, Xd_dd]) + self.Kp @ e + self.Kd @ edot
        
        
#         # Jdot*dq
        
        
        
#         # aq = J_pinv @ (aX - Jdot_dq)
#         # lam = 1e-3
#         # JvJJ = Jv @ Jv.T
#         # Jv_pinv = Jv.T @ np.linalg.inv(JvJJ + lam**2*np.eye(3))
#         # aq = Jv_pinv @ (aX - Jvdot_qdot)
        
#         lam = 1e-3
#         J_pinv = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(6))
#         aq = J_pinv @ (aX - Jdot_qdot)

#         #  M(q)qdd_ref + C(q,dq)dq + G(q)
#         tau = pin.rnea(model, data, q, dq, aq)


#         # # pub
#         out = Float64MultiArray()
#         out.data = tau.tolist()
#         self.tau_pub.publish(out)
        
#         self.get_logger().info(f"t={t:.2f}, pos={X}, Xd={Xd}, e={e}")



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
    
    
    
    