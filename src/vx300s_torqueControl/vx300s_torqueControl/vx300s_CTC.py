#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import pinocchio as pin
from std_msgs.msg import Float64MultiArray

from IK import get_angles  # 你的逆运动学函数
import modern_robotics as mr

# ====== 目标关节名称 ======
target_joints = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper", "left_finger", "right_finger"]


Slist = np.array([
    [0.0, 0.0, 1.0,  0.0,      0.0,     0.0],
    [0.0, 1.0, 0.0, -0.12705,  0.0,     0.0],
    [0.0, 1.0, 0.0, -0.42705,  0.0,     0.05955],
    [1.0, 0.0, 0.0,  0.0,      0.42705, 0.0],
    [0.0, 1.0, 0.0, -0.42705,  0.0,     0.35955],
    [1.0, 0.0, 0.0,  0.0,      0.42705, 0.0]
]).T



def extract_by_name(msg_names, msg_vals, wanted_names):
    """从JointState消息中提取指定关节顺序的角度或速度"""
    m = dict(zip(msg_names, msg_vals)) if msg_vals is not None else {}
    return np.array([m.get(n, 0.0) for n in wanted_names], dtype=float)

# ===== 五次多项式函数 =====
def quintic_coeffs(s0, v0, a0, sf, vf, af, T):
    A = np.array([
        [T**3,   T**4,    T**5],
        [3*T**2, 4*T**3,  5*T**4],
        [6*T,   12*T**2, 20*T**3]
    ], dtype=float)
    b = np.array([
        sf - (s0 + v0*T + 0.5*a0*T**2),
        vf - (v0 + a0*T),
        af - a0
    ], dtype=float)
    a3, a4, a5 = np.linalg.solve(A, b)
    return np.array([s0, v0, a0, a3, a4, a5])

def quintic_eval(coeffs, t):
    s0, v0, a0, a3, a4, a5 = coeffs
    s   = s0 + v0*t + 0.5*a0*t**2 + a3*t**3 + a4*t**4 + a5*t**5
    ds  = v0 + a0*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
    dds = a0 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3
    return s, ds, dds


# ===== 直线轨迹生成函数 =====
def generate_line_trajectory(t, T_total=6.0):
    # 起点与终点
    P0 = np.array([0.4, 0.2, 0.3])
    Pf = np.array([0.4, -0.2, 0.3])
    roll, pitch, yaw = 0.0, np.pi/4, 0.0

    # 路径方向
    dP = Pf - P0
    d = np.linalg.norm(dP)
    dP_dir = dP / d

    # 周期化时间（实现来回运动）
    cycle_time = 2 * T_total
    t_mod = t % cycle_time  # 当前所在周期时间

    # 判断当前是“去程”还是“回程”
    if t_mod <= T_total:
        # 正向（P0 -> Pf）
        s0, sf = 0, d
        direction = 1.0
        t_local = t_mod
    else:
        # 反向（Pf -> P0）
        s0, sf = 0, d
        direction = -1.0
        t_local = t_mod - T_total  # 回程时间相对起点

    # 计算五次多项式（每个半周期重新使用）
    s_coeffs = quintic_coeffs(s0, 0, 0, sf, 0, 0, T_total)
    s, ds, dds = quintic_eval(s_coeffs, np.clip(t_local, 0, T_total))

    # 根据方向决定末端位置
    if direction > 0:
        Pd   = P0 + s  * dP_dir
        dPd  =      ds * dP_dir
        ddPd =     dds * dP_dir
    else:
        Pd   = Pf - s  * dP_dir
        dPd  = -    ds * dP_dir
        ddPd = -   dds * dP_dir

    pos = np.concatenate(([roll, pitch, yaw], Pd))
    vel = np.concatenate(([0, 0, 0], dPd))
    acc = np.concatenate(([0, 0, 0], ddPd))
    return pos, vel, acc



def generate_ellipse_trajectory(t, T_total=6.0):
    """
    使用五次多项式生成平滑椭圆轨迹：
    - 椭圆中心: Pc
    - 长轴 a 沿 Y 方向
    - 短轴 b 沿 Z 方向
    - 椭圆角度 θ(t) 用五次多项式平滑生成，保证速度与加速度连续
    """

    # === 椭圆参数 ===
    Pc = np.array([0.3, 0.0, 0.3])  # 椭圆中心
    a = 0.15                        # y方向半径
    b = 0.05                        # z方向半径
    roll, pitch, yaw = 0.0, np.pi/4, 0.0

    # === 角度的五次多项式 θ(t) ===
    #   θ(0)=0, θ'(0)=0, θ''(0)=0
    #   θ(T)=2π, θ'(T)=0, θ''(T)=0
    theta_coeffs = quintic_coeffs(0, 0, 0, 2*np.pi, 0, 0, T_total)
    theta, dtheta, ddtheta = quintic_eval(theta_coeffs, t % T_total)

    # === 位置 ===
    x = Pc[0] + a * np.cos(theta)
    y = Pc[1] + b * np.sin(theta)
    z = Pc[2] 

    # === 一阶导（速度） ===
    dx = 0.0
    dy = -a * np.sin(theta) * dtheta
    dz =  b * np.cos(theta) * dtheta

    # === 二阶导（加速度） ===
    ddx = 0.0
    ddy = -a * (np.cos(theta) * dtheta**2 + np.sin(theta) * ddtheta)
    ddz =  b * (-np.sin(theta) * dtheta**2 + np.cos(theta) * ddtheta)

    # === 合并 ===
    pos = np.array([roll, pitch, yaw, x, y, z])
    vel = np.array([0, 0, 0, dx, dy, dz])
    acc = np.array([0, 0, 0, ddx, ddy, ddz])

    return pos, vel, acc

class TrajectorySimNode(Node):
    def __init__(self):
        super().__init__("trajectory_sim_node")

        # === Pinocchio 模型 ===
        urdf_model_path = "/home/yc/vx300s-single-sim/src/vx300s_description/urdf/vx300s.urdf"
        mesh_dir = "/home/yc/vx300s-single-sim/src/vx300s_description/vx300s_meshes/"
        
        # urdf_model_path = "/home/haibo/vx300s_ws/src/vx300s_description/urdf/vx300s.urdf"
        # mesh_dir = "/home/haibo/vx300s_ws/src/vx300s_description/vx300s_meshes/"
        self.model, _, _ = pin.buildModelsFromUrdf(urdf_model_path, package_dirs=[mesh_dir])
        self.data = self.model.createData()

        self.ee_name = "vx300s/ee_gripper_link"
        self.ee_id = self.model.getFrameId(self.ee_name)

        # === 控制参数 ===
        self.Kp = np.diag([460, 260, 260, 1000, 2500, 2000])
        self.Kd = np.diag([15, 15, 14, 13, 22, 25])

        self.joint_state_msg = None
        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        # === 订阅Joint States ===
        self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 10)
        self.tau_pub = self.create_publisher(Float64MultiArray, "/arm_controller/commands", 10)
        self.pub_e   = self.create_publisher(Float64MultiArray, "/vx300s/e", 10)  # <--- 新增：发布误差 e


        # === 定时器(50Hz) ===
        self.timer = self.create_timer(0.01, self.timer_callback)

    def joint_state_callback(self, msg: JointState):
        self.joint_state_msg = msg

    def timer_callback(self):
        if self.joint_state_msg is None:
            return

        t = self.get_clock().now().nanoseconds * 1e-9 - self.start_time
        q = extract_by_name(self.joint_state_msg.name, self.joint_state_msg.position, target_joints)
        dq = extract_by_name(self.joint_state_msg.name, self.joint_state_msg.velocity, target_joints)
        # print(q[:6])
        
        # print(t)

        # # === 五次多项式直线轨迹 ===
        pos, vel, acc = generate_ellipse_trajectory(t)
        roll, pitch, yaw, x, y, z = pos

        # # === 数值逆运动学 ===
        j1, j2, j3, j4, j5, j6 = get_angles(x, y, z, roll, pitch, yaw)
        q_des = np.array([j1, j2, j3, j4, j5, j6], dtype=float)

        # # === 雅可比伪逆 ===
        J = mr.JacobianSpace(Slist, q_des) 
        J_pinv = np.linalg.pinv(J)
        # print(J_pinv.shape)
        
        a_cl = pin.getFrameClassicalAcceleration(self.model, self.data, self.ee_id, pin.LOCAL_WORLD_ALIGNED)
        Jdot_qdot = np.hstack([a_cl.angular, a_cl.linear])
        # print(Jdot_qdot)
        
        qd_des  = J_pinv @ vel
        qdd_des = J_pinv @ (acc - Jdot_qdot)
        
        # # print(q_des)

        print(q)
        
        # # === 误差 ===
        e = q_des - q[:6]
        edot = qd_des - dq[:6]
        qdd_ref = qdd_des + self.Kp @ e + self.Kd @ edot
        qdd_ref_full = np.concatenate([qdd_ref, np.zeros(3)])
        q_full = np.concatenate([q, np.zeros(1)])
        
        # print(e)
        e_msg = Float64MultiArray()
        e_msg.data = e.tolist()
        self.pub_e.publish(e_msg)

        # # === 计算动力学力矩 ===
        tau = pin.rnea(self.model, self.data, q_full, dq, qdd_ref_full)
        # print(tau)
        

        msg_out = Float64MultiArray()
        msg_out.data = tau[:6].tolist() 
        self.tau_pub.publish(msg_out)

        # === 打印仿真信息 ===
        # self.get_logger().info(
        #         f"t={t:.2f}s | pos={np.round(pos[:3],3)} | q_des={np.round(q_des,3)} | tau={np.round(tau,2)}"
        #     )


def main(args=None):
    rclpy.init(args=args)
    node = TrajectorySimNode()
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
# import numpy as np
# import pinocchio as pin
# from std_msgs.msg import Float64MultiArray

# from IK import get_angles  # 你的逆运动学函数

# # ====== 目标关节名称 ======
# target_joints = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]

# def extract_by_name(msg_names, msg_vals, wanted_names):
#     """从JointState消息中提取指定关节顺序的角度或速度"""
#     m = dict(zip(msg_names, msg_vals)) if msg_vals is not None else {}
#     return np.array([m.get(n, 0.0) for n in wanted_names], dtype=float)

# # ===== 椭圆轨迹生成函数 =====
# def generate_ellipse_trajectory(t, T_total=10.0):
#     """
#     生成一个椭圆轨迹：
#     x(t) = cx + a*cos(w*t)
#     y(t) = cy + b*sin(w*t)
#     z(t) = cz
#     """
#     # 椭圆中心与长短轴
#     cx, cy, cz = 0.3, 0.0, 0.3
#     a, b = 0.15, 0.10  # 长轴和短轴
#     w = 2 * np.pi / T_total  # 角速度（保证一个周期内走完椭圆）

#     roll, pitch, yaw = 0.0, np.pi/2, 0.0

#     # 位置
#     x = cx + a * np.cos(w * t)
#     y = cy + b * np.sin(w * t)
#     z = cz

#     # 一阶导（速度）
#     dx = -a * w * np.sin(w * t)
#     dy =  b * w * np.cos(w * t)
#     dz = 0.0

#     # 二阶导（加速度）
#     ddx = -a * w**2 * np.cos(w * t)
#     ddy = -b * w**2 * np.sin(w * t)
#     ddz = 0.0

#     pos = np.array([x, y, z, roll, pitch, yaw])
#     vel = np.array([dx, dy, dz, 0, 0, 0])
#     acc = np.array([ddx, ddy, ddz, 0, 0, 0])

#     return pos, vel, acc


# class TrajectorySimNode(Node):
#     def __init__(self):
#         super().__init__("trajectory_sim_node")

#         # === Pinocchio 模型 ===
#         urdf_model_path = "/home/yc/vx300s-single-sim/src/vx300s_description/urdf/vx300s_fix.urdf"
#         mesh_dir = "/home/yc/vx300s-single-sim/src/vx300s_description/vx300s_meshes/"
#         self.model, _, _ = pin.buildModelsFromUrdf(urdf_model_path, package_dirs=[mesh_dir])
#         self.data = self.model.createData()

#         self.ee_name = "vx300s/ee_gripper_link"
#         self.ee_id = self.model.getFrameId(self.ee_name)

#         # === 控制参数 ===
#         self.Kp = np.diag([60, 60, 60, 50, 50, 30])
#         self.Kd = np.diag([5, 5, 4, 3, 2, 2])

#         self.joint_state_msg = None
#         self.start_time = self.get_clock().now().nanoseconds * 1e-9

#         # === 订阅Joint States ===
#         self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 10)
#         self.tau_pub = self.create_publisher(Float64MultiArray,
#                                              "/arm_controller/commands", 10)

#         # === 定时器(100Hz) ===
#         self.timer = self.create_timer(0.02, self.timer_callback)

#     def joint_state_callback(self, msg: JointState):
#         self.joint_state_msg = msg

#     def timer_callback(self):
#         if self.joint_state_msg is None:
#             return

#         q = extract_by_name(self.joint_state_msg.name, self.joint_state_msg.position, target_joints)
#         dq = extract_by_name(self.joint_state_msg.name, self.joint_state_msg.velocity, target_joints)
#         t = self.get_clock().now().nanoseconds * 1e-9 - self.start_time

#         # === 椭圆轨迹 ===
#         pos, vel, acc = generate_ellipse_trajectory(t)
#         x, y, z, roll, pitch, yaw = pos

#         # === 数值逆运动学 ===
#         j1, j2, j3, j4, j5, j6 = get_angles(x, y, z, roll, pitch, yaw)
#         q_des = np.array([j1, j2, j3, j4, j5, j6], dtype=float)

#         # === 雅可比阻尼伪逆 ===
#         J = pin.computeFrameJacobian(self.model, self.data, q_des, self.ee_id, pin.LOCAL_WORLD_ALIGNED)
#         J_pinv = np.linalg.inv(J)
#         qd_des  = J_pinv @ vel
#         qdd_des = J_pinv @ acc

#         # === 误差与控制 ===
#         e = q_des - q
#         edot = qd_des - dq
#         qdd_ref = qdd_des + self.Kp @ e + self.Kd @ edot

#         # === 动力学力矩 ===
#         tau = pin.rnea(self.model, self.data, q, dq, qdd_ref)

#         # === 发布到仿真控制器 ===
#         msg_out = Float64MultiArray()
#         msg_out.data = tau.tolist()
#         self.tau_pub.publish(msg_out)

#         # === 打印仿真信息 ===
#         self.get_logger().info(
#                 f"t={t:.2f}s | pos={np.round(pos[:3],3)} | q_des={np.round(q_des,3)} | tau={np.round(tau,2)}"
#             )


# def main(args=None):
#     rclpy.init(args=args)
#     node = TrajectorySimNode()
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
# import numpy as np
# import pinocchio as pin
# from std_msgs.msg import Float64MultiArray

# from IK import get_angles  # 你的逆运动学函数

# # ====== 目标关节名称 ======
# target_joints = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]

# def extract_by_name(msg_names, msg_vals, wanted_names):
#     """从JointState消息中提取指定关节顺序的角度或速度"""
#     m = dict(zip(msg_names, msg_vals)) if msg_vals is not None else {}
#     return np.array([m.get(n, 0.0) for n in wanted_names], dtype=float)

# # ===== 五次多项式函数 =====
# def quintic_coeffs(s0, v0, a0, sf, vf, af, T):
#     A = np.array([
#         [T**3,   T**4,    T**5],
#         [3*T**2, 4*T**3,  5*T**4],
#         [6*T,   12*T**2, 20*T**3]
#     ], dtype=float)
#     b = np.array([
#         sf - (s0 + v0*T + 0.5*a0*T**2),
#         vf - (v0 + a0*T),
#         af - a0
#     ], dtype=float)
#     a3, a4, a5 = np.linalg.solve(A, b)
#     return np.array([s0, v0, a0, a3, a4, a5])

# def quintic_eval(coeffs, t):
#     s0, v0, a0, a3, a4, a5 = coeffs
#     s   = s0 + v0*t + 0.5*a0*t**2 + a3*t**3 + a4*t**4 + a5*t**5
#     ds  = v0 + a0*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
#     dds = a0 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3
#     return s, ds, dds


# # ===== 直线轨迹生成函数 =====
# def generate_line_trajectory(t, T_total=6.0):
#     # 起点与终点
#     P0 = np.array([0.4, 0.2, 0.3])
#     Pf = np.array([0.4, -0.2, 0.3])
#     roll, pitch, yaw = 0.0, np.pi/4, 0.0

#     # 路径方向
#     dP = Pf - P0
#     d = np.linalg.norm(dP)
#     dP_dir = dP / d

#     # 周期化时间（实现来回运动）
#     cycle_time = 2 * T_total
#     t_mod = t % cycle_time  # 当前所在周期时间

#     # 判断当前是“去程”还是“回程”
#     if t_mod <= T_total:
#         # 正向（P0 -> Pf）
#         s0, sf = 0, d
#         direction = 1.0
#         t_local = t_mod
#     else:
#         # 反向（Pf -> P0）
#         s0, sf = 0, d
#         direction = -1.0
#         t_local = t_mod - T_total  # 回程时间相对起点

#     # 计算五次多项式（每个半周期重新使用）
#     s_coeffs = quintic_coeffs(s0, 0, 0, sf, 0, 0, T_total)
#     s, ds, dds = quintic_eval(s_coeffs, np.clip(t_local, 0, T_total))

#     # 根据方向决定末端位置
#     if direction > 0:
#         Pd   = P0 + s  * dP_dir
#         dPd  =      ds * dP_dir
#         ddPd =     dds * dP_dir
#     else:
#         Pd   = Pf - s  * dP_dir
#         dPd  = -    ds * dP_dir
#         ddPd = -   dds * dP_dir

#     pos = np.concatenate((Pd, [roll, pitch, yaw]))
#     vel = np.concatenate((dPd, [0, 0, 0]))
#     acc = np.concatenate((ddPd, [0, 0, 0]))
#     return pos, vel, acc


# class TrajectorySimNode(Node):
#     def __init__(self):
#         super().__init__("trajectory_sim_node")

#         # === Pinocchio 模型 ===
#         # urdf_model_path = "/home/yc/vx300s-single-sim/src/vx300s_description/urdf/vx300s_fix.urdf"
#         # mesh_dir = "/home/yc/vx300s-single-sim/src/vx300s_description/vx300s_meshes/"
        
#         urdf_model_path = "/home/haibo/vx300s_ws/src/vx300s_description/urdf/vx300s_fix.urdf"
#         mesh_dir = "/home/haibo/vx300s_ws/src/vx300s_description/vx300s_meshes/"
#         self.model, _, _ = pin.buildModelsFromUrdf(urdf_model_path, package_dirs=[mesh_dir])
#         self.data = self.model.createData()

#         self.ee_name = "vx300s/ee_gripper_link"
#         self.ee_id = self.model.getFrameId(self.ee_name)

#         # === 控制参数 ===
#         self.Kp = np.diag([260, 260, 260, 250, 250, 3000])
#         self.Kd = np.diag([55, 55, 54, 43, 32, 45])

#         self.joint_state_msg = None
#         self.start_time = self.get_clock().now().nanoseconds * 1e-9

#         # === 订阅Joint States ===
#         self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 10)
#         self.tau_pub = self.create_publisher(Float64MultiArray, "/arm_controller/commands", 10)

#         # === 定时器(50Hz) ===
#         self.timer = self.create_timer(0.01, self.timer_callback)

#     def joint_state_callback(self, msg: JointState):
#         self.joint_state_msg = msg

#     def timer_callback(self):
#         if self.joint_state_msg is None:
#             return

#         t = self.get_clock().now().nanoseconds * 1e-9 - self.start_time
#         q = extract_by_name(self.joint_state_msg.name, self.joint_state_msg.position, target_joints)
#         dq = extract_by_name(self.joint_state_msg.name, self.joint_state_msg.velocity, target_joints)
        
#         # print(t)

#         # === 五次多项式直线轨迹 ===
#         pos, vel, acc = generate_line_trajectory(t)
#         x, y, z, roll, pitch, yaw = pos

#         # === 数值逆运动学 ===
#         j1, j2, j3, j4, j5, j6 = get_angles(x, y, z, roll, pitch, yaw)
#         q_des = np.array([j1, j2, j3, j4, j5, j6], dtype=float)

#         # === 雅可比伪逆 ===
#         J = pin.computeFrameJacobian(self.model, self.data, q_des, self.ee_id, pin.LOCAL_WORLD_ALIGNED)
#         J_pinv = np.linalg.pinv(J)
#         print(J_pinv.shape)
        
#         a_cl = pin.getFrameClassicalAcceleration(self.model, self.data, self.ee_id, pin.LOCAL_WORLD_ALIGNED)
#         Jdot_qdot = np.hstack([a_cl.linear, a_cl.angular])
        
#         qd_des  = J_pinv @ vel
#         qdd_des = J_pinv @ (acc - Jdot_qdot)
        
#         # print(q_des)
        
        
        
        
#         # print(q)
        
#         # === 误差 ===
#         e = q_des - q
#         edot = qd_des - dq
#         qdd_ref = qdd_des + self.Kp @ e + self.Kd @ edot
        
#         print(e)

#         # === 计算动力学力矩 ===
#         tau = pin.rnea(self.model, self.data, q, dq, qdd_ref)


#         msg_out = Float64MultiArray()
#         msg_out.data = tau.tolist()
#         self.tau_pub.publish(msg_out)

#         # === 打印仿真信息 ===
#         # self.get_logger().info(
#         #         f"t={t:.2f}s | pos={np.round(pos[:3],3)} | q_des={np.round(q_des,3)} | tau={np.round(tau,2)}"
#         #     )


# def main(args=None):
#     rclpy.init(args=args)
#     node = TrajectorySimNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main()










