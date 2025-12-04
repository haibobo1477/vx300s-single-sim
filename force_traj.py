#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import pinocchio as pin

from interbotix_common_modules.common_robot.robot import robot_startup, robot_shutdown
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointGroupCommand

from IK import get_angles
import modern_robotics as mr

# ====== 目标关节 ======
target_joints = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]

Slist = np.array([
    [0.0, 0.0, 1.0,  0.0,      0.0,     0.0],
    [0.0, 1.0, 0.0, -0.12705,  0.0,     0.0],
    [0.0, 1.0, 0.0, -0.42705,  0.0,     0.05955],
    [1.0, 0.0, 0.0,  0.0,      0.42705, 0.0],
    [0.0, 1.0, 0.0, -0.42705,  0.0,     0.35955],
    [1.0, 0.0, 0.0,  0.0,      0.42705, 0.0]
]).T


def extract_by_name(msg_names, msg_vals, wanted_names):
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


# ===== 椭圆轨迹生成函数 =====
def generate_line_trajectory(t, T_total=13.0):
    # 起点与终点
    P0 = np.array([0.4, 0.2, 0.3])
    Pf = np.array([0.4, -0.2, 0.3])
    roll, pitch, yaw = 0.0, np.pi/4, 0.0

    dP = Pf - P0
    d = np.linalg.norm(dP)
    dP_dir = dP / d

    cycle_time = 2 * T_total
    t_mod = t % cycle_time 

    if t_mod <= T_total:
        s0, sf = 0, d
        direction = 1.0
        t_local = t_mod
    else:
        s0, sf = 0, d
        direction = -1.0
        t_local = t_mod - T_total  # 回程时间相对起点

    s_coeffs = quintic_coeffs(s0, 0, 0, sf, 0, 0, T_total)
    s, ds, dds = quintic_eval(s_coeffs, np.clip(t_local, 0, T_total))

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



def generate_ellipse_trajectory(t, T_total=13.0):
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


class JointStateTimerNode(Node):
    def __init__(self):
        super().__init__("joint_state_timer_node")

        # === Pinocchio ===
        urdf_model_path = "/home/yc/vx300s-single-sim/src/vx300s_description/urdf/vx300s_fix.urdf"
        mesh_dir = "/home/yc/vx300s-single-sim/src/vx300s_description/vx300s_meshes/"
        self.model, _, _ = pin.buildModelsFromUrdf(urdf_model_path, package_dirs=[mesh_dir])
        self.model.gravity = pin.Motion.Zero()
        self.model.gravity.linear = np.array([0.0, 0.0, -9.81])
        self.data = self.model.createData()

        self.ee_name = "vx300s/ee_gripper_link"
        self.ee_id = self.model.getFrameId(self.ee_name)

        # === PID gain ===
        self.Kp = np.diag([20, 25, 24, 10, 12, 10])
        # self.Kd = np.diag([15, 15, 14, 13, 22, 25])
        self.Ki = np.diag([2.5, 3.5, 3.5, 1.5, 2.0, 3.5])

        self.n = 6
        self.Omega = 0.02                         #
        self.gamma = 3.0                          # I/gamma^2 
        self.Gamma = np.array([0.02, 0.030, 0.026, 0.025, 0.010, 0.025], dtype=float)  
        self.Khat_diag = np.zeros(6, dtype=float) # 
        # self.Khat_max  = np.array([400, 400, 400, 300, 250, 250], dtype=float)
        self.Khat = np.diag(self.Khat_diag)

        self.e_int     = np.zeros(6)
        # self.eint_max  = np.array([0.4, 0.4, 0.4, 0.25, 0.25, 0.25], dtype=float)
        self.edot_filt = np.zeros(6)


        # === torque to cuurent ===
        self.torque_constants = np.array([2.15, 2.15, 2.15, 2.15, 2.15, 1.793], dtype=float)
        self.current_units    = np.array([0.00269]*6, dtype=float)
        self.cmd_last = np.zeros(6)

        # JointState 
        self.joint_state_msg = None
        now = self.get_clock().now().nanoseconds * 1e-9
        self.start_time = now
        self.prev_time  = now

        # === Joint States ===
        self.create_subscription(JointState, "/vx300s/joint_states", self.joint_state_callback, 50)

        # === Interbotix ===
        self.bot = InterbotixManipulatorXS(robot_model='vx300s')
        robot_startup()
        self.bot.core.robot_set_operating_modes('group', 'arm', 'current')

        # === pub ===
        self.pub = self.create_publisher(JointGroupCommand, "/vx300s/commands/joint_group", 10)
        self.pub_e   = self.create_publisher(Float64MultiArray, "/vx300s/e", 10)  

        # === timer 200Hz ===
        self.timer = self.create_timer(0.022, self.timer_callback)

    def joint_state_callback(self, msg: JointState):
        self.joint_state_msg = msg

    def timer_callback(self):
        if self.joint_state_msg is None:
            return

        
        t_now = self.get_clock().now().nanoseconds * 1e-9
        dt = t_now - self.prev_time
        if dt <= 0.0: 
            dt = 1e-4
        self.prev_time = t_now
        t = t_now - self.start_time
        
        # === traj ===
        pos, vel, acc = generate_line_trajectory(t)
        # pos, vel, acc = generate_ellipse_trajectory(t)
        
        roll, pitch, yaw, x, y, z = pos

        # === inverse kin ===
        j1, j2, j3, j4, j5, j6 = get_angles(x, y, z, roll, pitch, yaw)
        q_des = np.array([j1, j2, j3, j4, j5, j6], dtype=float)

        # === jacobian ===
        # J = pin.computeFrameJacobian(self.model, self.data, q_des, self.ee_id, pin.LOCAL_WORLD_ALIGNED)
        J = mr.JacobianSpace(Slist, q_des) 

        lamda = 0.05  # damping factor，0.01~0.1
        J_pinv = J.T @ np.linalg.inv(J @ J.T + (lamda ** 2) * np.eye(J.shape[0]))
        # J_pinv = np.linalg.pinv(J)

        a_cl = pin.getFrameClassicalAcceleration(self.model, self.data, self.ee_id, pin.LOCAL_WORLD_ALIGNED)
        Jdot_qdot = np.hstack([a_cl.angular, a_cl.linear])

        qd_des  = J_pinv @ vel
        qdd_des = J_pinv @ (acc - Jdot_qdot)

        q = extract_by_name(self.joint_state_msg.name, self.joint_state_msg.position, target_joints)
        dq = extract_by_name(self.joint_state_msg.name, self.joint_state_msg.velocity, target_joints)
        


        # === error ===
        e    = q_des - q

        # self.edot_filt = 0.9 * self.edot_filt + 0.1 * (qd_des - dq)
        # edot = self.edot_filt

        edot = qd_des - dq
        print(e)

        self.e_int += e * dt
        # self.e_int = np.clip(self.e_int, -self.eint_max, self.eint_max)

        # print(e)
        e_msg = Float64MultiArray()
        e_msg.data = e.tolist()
        self.pub_e.publish(e_msg)

        # === acc reference ===
        # qdd_ref = qdd_des + self.Kp @ e + self.Kd @ edot
        s = edot + self.Kp @ e + self.Ki @ self.e_int

        thresh = self.Omega / np.sqrt(2.0 * self.n)
        for i in range(self.n):
            if abs(s[i]) > thresh:
                self.Khat_diag[i] += self.Gamma[i] * (s[i]**2) * dt

        # self.Khat_diag = np.clip(self.Khat_diag, 0.0, self.Khat_max)
        self.Khat = np.diag(self.Khat_diag)

        qdd_ref = qdd_des + (self.Khat + (1.0 / (self.gamma**2)) * np.eye(self.n)) @ s

        # === inverse dynamic ===
        tau = pin.rnea(self.model, self.data, q, dq, qdd_ref)

        # === torque to cuurent ===

        I = tau / self.torque_constants

        cmd = I / self.current_units
 
        self.cmd_last = cmd
        
        # print(cmd)


        # === pub ===
        msg = JointGroupCommand()
        msg.name = "arm"
        msg.cmd = self.cmd_last.tolist()
        self.pub.publish(msg)

        # self.get_logger().info(f"t={t:.2f}s, pos={np.round(pos[:3],3)}, tau={np.round(tau,2)}")

    def destroy_node(self):
        msg = JointGroupCommand()
        msg.name = "arm"
        msg.cmd = [0.0]*6
        self.pub.publish(msg)
        robot_shutdown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = JointStateTimerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()









