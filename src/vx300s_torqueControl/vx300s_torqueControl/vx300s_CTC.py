#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import numpy as np
import pinocchio as pin

from IK import get_angles
import modern_robotics as mr


# ====== Pinocchio model ======
# urdf_model_path = "/home/haibo/vx300s_ws/src/vx300s_description/urdf/vx300s_fix.urdf"
# mesh_dir = "/home/haibo/vx300s_ws/src/vx300s_description/vx300s_meshes/"


urdf_model_path = "/home/yc/vx300s-single-sim/src/vx300s_description/urdf/vx300s_fix.urdf"
mesh_dir = "/home/yc/vx300s-single-sim/src/vx300s_description/vx300s_meshes/"

model, _, _ = pin.buildModelsFromUrdf(urdf_model_path, package_dirs=[mesh_dir])
model.gravity = pin.Motion.Zero()
model.gravity.linear = np.array([0.0, 0.0, -9.81])
data = model.createData()

target_joints = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
assert model.nv == len(target_joints)


Slist = np.array([
    [0.0, 0.0, 1.0,  0.0,      0.0,     0.0],
    [0.0, 1.0, 0.0, -0.12705,  0.0,     0.0],
    [0.0, 1.0, 0.0, -0.42705,  0.0,     0.05955],
    [1.0, 0.0, 0.0,  0.0,      0.42705, 0.0],
    [0.0, 1.0, 0.0, -0.42705,  0.0,     0.35955],
    [1.0, 0.0, 0.0,  0.0,      0.42705, 0.0]
]).T

# ====== Utils ======
def extract_by_name(msg_names, msg_vals, wanted_names):
    m = dict(zip(msg_names, msg_vals)) if msg_vals is not None else {}
    return np.array([m.get(n, 0.0) for n in wanted_names], dtype=float)

# 五次多项式轨迹函数
def quintic_trajectory(q0, qf, T, t):
    tau = np.clip(t / T, 0.0, 1.0)
    tau2, tau3, tau4, tau5 = tau**2, tau**3, tau**4, tau**5
    s     = 10*tau3 - 15*tau4 + 6*tau5
    s_dot = (30*tau2 - 60*tau3 + 30*tau4) / T
    s_ddot= (60*tau - 180*tau2 + 120*tau3) / (T**2)
    q   = (1-s)*q0 + s*qf
    qd  = s_dot*(qf - q0)
    qdd = s_ddot*(qf - q0)
    return q, qd, qdd

# ====== Trajectory Generator ======
class TrajectoryGenerator:
    def __init__(self, mode="sine"):
        self.mode = mode
        self.ee_name = "vx300s/ee_gripper_link"
        self.ee_id = model.getFrameId(self.ee_name)

        # for quintic
        self.q0 = np.zeros(6)
        self.qf = np.array([0.5, -0.5, 0.3, 0.0, 0.0, 0.0])
        self.T  = 8.0  # seconds

    def generate(self, t, q_current):
        if self.mode == "sine":
            A = np.array([0.3, 0.2, 0.2, 0.1, 0.0, 0.0])   # 振幅 (rad)
            w = np.array([0.5, 0.5, 0.8, 0.6, 0.0, 0.0])   # 频率 (rad/s)
            q   = A * np.sin(w * t)
            qd  = A * w * np.cos(w * t)
            qdd = -A * (w**2) * np.sin(w * t)
            return q, qd, qdd

        elif self.mode == "quintic":
            return quintic_trajectory(self.q0, self.qf, self.T, t)

        elif self.mode == "circle":
            # 任务空间圆轨迹
            roll, pitch, yaw = np.pi/2, 0.0, np.pi/2
            center = np.array([0.3, 0.3, 0.3])
            r, w = 0.1, 1.0
            pos = np.array([
                center[0] + r * np.cos(w * t),
                center[1] + r * np.sin(w * t),
                center[2],
                roll,
                pitch,
                yaw
            ])
            vel = np.array([
                -r * w * np.sin(w * t),
                 r * w * np.cos(w * t),
                 0.0,
                 0.0,
                 0.0,
                 0.0,
            ])
            acc = np.array([
                -r * w**2 * np.cos(w * t),
                -r * w**2 * np.sin(w * t),
                 0.0,
                 0.0,
                 0.0,
                 0.0,
            ])

             
            j1, j2, j3, j4, j5, j6 = get_angles(pos[0], pos[1], pos[2], roll, pitch, yaw)
            q_des = np.array([j1, j2, j3, j4, j5, j6], dtype=float)

            # 速度 & 加速度用伪逆算
            J = pin.computeFrameJacobian(model, data, q_des, self.ee_id, pin.LOCAL_WORLD_ALIGNED)
            # qd_des  = np.linalg.pinv(J[:3, :]) @ vel
            # qdd_des = np.linalg.pinv(J[:3, :]) @ (acc - pin.getFrameClassicalAcceleration(model, data, self.ee_id, pin.LOCAL_WORLD_ALIGNED).linear)

            # J = mr.JacobianSpace(Slist, q_des)

            qd_des  = np.linalg.pinv(J) @ vel
            qdd_des = np.linalg.pinv(J) @ (acc - pin.getFrameClassicalAcceleration(model, data, self.ee_id, pin.LOCAL_WORLD_ALIGNED))
            return q_des, qd_des, qdd_des

        else:
            raise ValueError("Unknown trajectory mode")


# ====== ROS2 Node ======
class JointSpaceController(Node):
    def __init__(self, mode):
        super().__init__("joint_space_controller")
        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        # controller params
        self.Kp = np.diag([50, 50, 40, 40, 50, 100])
        self.Kd = np.diag([5,  5,  4,  20,  20, 50.5])
        self.D_visc = np.array([0.6, 0.8, 0.6, 0.15, 0.12, 0.12])
        self.tau_limit = np.array([10, 10, 10, 5, 4, 4])

        # trajectory generator
        self.traj_gen = TrajectoryGenerator(mode=mode)

        # pub/sub
        self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 50)
        self.tau_pub = self.create_publisher(Float64MultiArray, "/arm_controller/commands", 10)

    def joint_state_callback(self, msg: JointState):
        t = self.get_clock().now().nanoseconds * 1e-9 - self.start_time
        q  = extract_by_name(msg.name, msg.position, target_joints)
        dq = extract_by_name(msg.name, msg.velocity, target_joints)

        # reference trajectory
        q_des, qd_des, qdd_des = self.traj_gen.generate(t, q)

        # error
        e    = q_des - q
        edot = qd_des - dq

        # reference acc
        qdd_ref = qdd_des + self.Kp @ e + self.Kd @ edot

        # dynamics
        tau = pin.rnea(model, data, q, dq, qdd_ref)
        print(tau)
        # tau -= self.D_visc * dq
        # tau = np.clip(tau, -self.tau_limit, self.tau_limit)

        # log
        # self.get_logger().info(f"t={t:.2f}, q_des={q_des}, q={q}")

        # pub
        out = Float64MultiArray()
        out.data = tau.tolist()
        self.tau_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = JointSpaceController(mode="circle")  # "sine" / "quintic" / "circle"
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()









'''


#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import pinocchio as pin

from interbotix_common_modules.common_robot.robot import robot_startup, robot_shutdown
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointGroupCommand

from IK import get_angles

# ====== 目标关节 ======
target_joints = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]

def extract_by_name(msg_names, msg_vals, wanted_names):
    m = dict(zip(msg_names, msg_vals)) if msg_vals is not None else {}
    return np.array([m.get(n, 0.0) for n in wanted_names], dtype=float)

# ===== 圆轨迹生成函数 =====
def generate_circle_trajectory(t):
    roll, pitch, yaw = 0.0, 0.0, 0.0
    center = np.array([0.3, 0.3, 0.3])  # 圆心
    r, w = 0.1, 0.2  # 半径, 角速度

    pos = np.array([
        center[0] + r * np.cos(w * t),
        center[1] + r * np.sin(w * t),
        center[2],
        roll,
        pitch,
        yaw
    ])
    vel = np.array([
        -r * w * np.sin(w * t),
         r * w * np.cos(w * t),
         0.0, 0.0, 0.0, 0.0
    ])
    acc = np.array([
        -r * w**2 * np.cos(w * t),
        -r * w**2 * np.sin(w * t),
         0.0, 0.0, 0.0, 0.0
    ])
    return pos, vel, acc


class JointStateTimerNode(Node):
    def __init__(self):
        super().__init__("joint_state_timer_node")

        # Pinocchio 模型
        urdf_model_path = "/home/yc/vx300s-single-sim/src/vx300s_description/urdf/vx300s_fix.urdf"
        mesh_dir = "/home/yc/vx300s-single-sim/src/vx300s_description/vx300s_meshes/"
        self.model, _, _ = pin.buildModelsFromUrdf(urdf_model_path, package_dirs=[mesh_dir])
        self.model.gravity = pin.Motion.Zero()
        self.model.gravity.linear = np.array([0.0, 0.0, -9.81])
        self.data = self.model.createData()

        self.ee_name = "vx300s/ee_gripper_link"
        self.ee_id = self.model.getFrameId(self.ee_name)

        # 控制参数
        self.Kp = np.diag([50, 50, 40, 30, 20, 10])
        self.Kd = np.diag([5,  5,  4,  3,  2,  2])

        # 力矩转电流参数
        self.torque_constants = np.array([2.15, 2.15, 2.15, 2.15, 2.15, 1.793], dtype=float)
        self.current_units    = np.array([0.00269]*6, dtype=float)
        self.current_limit_cmd = np.array([500, 500, 500, 400, 300, 300], dtype=float)  # 命令限幅

        # 电流平滑
        self.cmd_last = np.zeros(6)

        # JointState 缓存
        self.joint_state_msg = None
        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        # 订阅关节状态
        self.create_subscription(JointState, "/vx300s/joint_states", self.joint_state_callback, 50)

        # Interbotix 初始化
        self.bot = InterbotixManipulatorXS(robot_model='vx300s')
        robot_startup()
        self.bot.core.robot_set_operating_modes('group', 'arm', 'current')

        # 发布器
        self.pub = self.create_publisher(JointGroupCommand, "/vx300s/commands/joint_group", 10)

        # 控制定时器 100Hz
        self.timer = self.create_timer(0.02, self.timer_callback)

    def joint_state_callback(self, msg: JointState):
        self.joint_state_msg = msg

    def timer_callback(self):
        if self.joint_state_msg is None:
            return

        q = extract_by_name(self.joint_state_msg.name, self.joint_state_msg.position, target_joints)
        dq = extract_by_name(self.joint_state_msg.name, self.joint_state_msg.velocity, target_joints)
        t = self.get_clock().now().nanoseconds * 1e-9 - self.start_time

        # 生成圆轨迹
        pos, vel, acc = generate_circle_trajectory(t)
        x, y, z, roll, pitch, yaw = pos

        # IK
        j1, j2, j3, j4, j5, j6 = get_angles(x, y, z, roll, pitch, yaw)
        q_des = np.array([j1, j2, j3, j4, j5, j6], dtype=float)

        # 雅可比 (阻尼伪逆)
        J = pin.computeFrameJacobian(self.model, self.data, q_des, self.ee_id, pin.LOCAL_WORLD_ALIGNED)
        lam = 1e-3
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(J.shape[0]))
        qd_des  = J_pinv @ vel
        qdd_des = J_pinv @ acc

        # 误差
        e    = q_des - q
        edot = qd_des - dq

        # 参考加速度
        qdd_ref = qdd_des + self.Kp @ e + self.Kd @ edot

        # 动力学 (加阻尼补偿)
        tau = pin.rnea(self.model, self.data, q, dq, qdd_ref)
        # tau -= 0.2 * dq

        # 力矩 -> 电流命令
        I = tau / self.torque_constants
        cmd = I / self.current_units

        # 限幅 + 平滑
        cmd = np.clip(cmd, -self.current_limit_cmd, self.current_limit_cmd)
        self.cmd_last = 0.9 * self.cmd_last + 0.1 * cmd

        # 发布
        msg = JointGroupCommand()
        msg.name = "arm"
        msg.cmd = self.cmd_last.tolist()
        self.pub.publish(msg)

        # 打印日志（每 0.5s 打印一次）
        self.get_logger().info(f"t={t:.2f}, tau={np.round(tau,2)}, cmd={np.round(self.cmd_last,1)}")

    def destroy_node(self):
        # 停止前清零命令
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


'''












