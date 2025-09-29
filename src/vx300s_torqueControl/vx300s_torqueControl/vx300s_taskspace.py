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



Slist = np.array([[0.0, 0.0, 1.0,  0.0,     0.0,     0.0],
                  [0.0, 1.0, 0.0, -0.12705, 0.0,     0.0],
                  [0.0, 1.0, 0.0, -0.42705, 0.0,     0.05955],
                  [1.0, 0.0, 0.0,  0.0,     0.42705, 0.0],
                  [0.0, 1.0, 0.0, -0.42705, 0.0,     0.35955],
                  [1.0, 0.0, 0.0,  0.0,     0.42705, 0.0]]).T


M = np.array([[1.0, 0.0, 0.0, 0.536494],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.42705],
              [0.0, 0.0, 0.0, 1.0]])



# ====== Pinocchio model ======
# urdf_model_path = "/home/haibo/vx300s_ws/src/vx300s_description/urdf/vx300s_fix.urdf"
# mesh_dir = "/home/haibo/vx300s_ws/src/vx300s_description/vx300s_meshes/"

urdf_model_path = "/home/yc/vx300s-single-sim/src/vx300s_description/urdf/vx300s_fix.urdf"
mesh_dir = "/home/yc/vx300s-single-sim/src/vx300s_description/vx300s_meshes/"



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

        # pub/sub
        self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 50)
        self.tau_pub = self.create_publisher(Float64MultiArray, "/arm_controller/commands", 10)

    def joint_state_callback(self, msg: JointState):
        
        # time now
        t = self.get_clock().now().nanoseconds * 1e-9 - self.start_time

        q  = extract_by_name(msg.name, msg.position, target_joints)
        dq = extract_by_name(msg.name, msg.velocity, target_joints) 


        X = mr.FKinSpace(M, Slist, q)
        J = mr.JacobianSpace(Slist, q)
        pin.computeJointJacobiansTimeVariation(model, data, q, dq)
        # Jdot_dq = computeJdotTimesV(q, dq)


        # Xd, Xd_dot, Xd_ddot = trajectory_generator(t)

        # aX = Xd_ddot + Kp @ (Xd - X) + Kd @ (Xd_dot - J@dq)
        # aq = J_pinv @ (aX - Jdot_dq)

        #  M(q)qdd_ref + C(q,dq)dq + G(q)
        # tau = pin.rnea(model, data, q, dq, aq)

        
        # self.get_logger().info(f"t={t:.2f}, q_des={self.q_des}, q={q}")

        # # pub
        # out = Float64MultiArray()
        # out.data = tau.tolist()
        # self.tau_pub.publish(out)



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