#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
from modern_robotics import GravityForces


class GravityCompNode(Node):
    def __init__(self):
        super().__init__('gravity_comp_node')

        # ---------------- Mlist ----------------
        self.M01 = np.array([[1, 0, 0,      0],
                             [0, 1, 0,      0],
                             [0, 0, 1,  0.079],
                             [0, 0, 0,      1]])
        self.M12 = np.array([[1, 0, 0,      0],
                             [0, 1, 0,      0],
                             [0, 0, 1,  0.048],
                             [0, 0, 0,      1]])
        self.M23 = np.array([[1, 0, 0,   0.06],
                             [0, 1, 0,      0],
                             [0, 0, 1,    0.3],
                             [0, 0, 0,      1]])
        self.M34 = np.array([[1, 0, 0,   0.2],
                             [0, 1, 0,     0],
                             [0, 0, 1,     0],
                             [0, 0, 0,     1]])
        self.M45 = np.array([[1, 0, 0,   0.1],
                             [0, 1, 0,     0],
                             [0, 0, 1,     0],
                             [0, 0, 0,     1]])
        self.M56 = np.array([[1, 0, 0,   0.07],
                             [0, 1, 0,      0],
                             [0, 0, 1,      0],
                             [0, 0, 0,      1]])
        self.M67 = np.array([[1, 0, 0,  0.107],
                             [0, 1, 0,      0],
                             [0, 0, 1,      0],
                             [0, 0, 0,      1]])
        self.Mlist = np.array([self.M01, self.M12, self.M23, self.M34,
                               self.M45, self.M56, self.M67])

        # ---------------- Glist ----------------
        self.G1 = np.diag([0.006024, 0.001700, 0.007162, 0.969034, 0.969034, 0.969034])
        self.G2 = np.diag([0.0009388, 0.001138, 0.001201, 0.798614, 0.798614, 0.798614])
        self.G3 = np.diag([0.008925, 0.008937, 0.0009357, 0.792592, 0.792592, 0.792592])
        self.G4 = np.diag([0.0001524, 0.001342, 0.001441, 0.322228, 0.322228, 0.322228])
        self.G5 = np.diag([0.0001753, 0.0005269, 0.0005911, 0.414823, 0.414823, 0.414823])
        self.G6 = np.diag([4.631e-5, 4.514e-5, 5.27e-5, 0.344642, 0.344642, 0.344642])
        self.Glist = np.array([self.G1, self.G2, self.G3, self.G4, self.G5, self.G6])

        # ---------------- Slist ----------------
        self.Slist = np.array([[0.0, 0.0, 1.0,  0.0,     0.0,     0.0],
                               [0.0, 1.0, 0.0, -0.12705, 0.0,     0.0],
                               [0.0, 1.0, 0.0, -0.42705, 0.0,  0.05955],
                               [1.0, 0.0, 0.0,  0.0,     0.42705, 0.0],
                               [0.0, 1.0, 0.0, -0.42705, 0.0,  0.35955],
                               [1.0, 0.0, 0.0,  0.0,     0.42705, 0.0]]).T

        # ---------------- 参数 ----------------
        self.g = np.array([0, 0, -9.81])  # 重力方向

        self.desired = ['waist','shoulder','elbow','forearm_roll','wrist_angle','wrist_rotate']

        # 订阅关节状态
        self.subscription = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        # 发布到控制话题
        self.pub = self.create_publisher(Float64MultiArray, '/arm_controller/commands', 10)

        # 定时器 10Hz
        self.timer = self.create_timer(0.01, self.timer_callback)

        self.latest_msg = None

    def joint_state_callback(self, msg: JointState):
        self.latest_msg = msg

    def timer_callback(self):
        if self.latest_msg is None:
            return

        # name -> index
        name2idx = {n: i for i, n in enumerate(self.latest_msg.name)}
        pos = np.zeros(6)
        for k, jname in enumerate(self.desired):
            i = name2idx.get(jname, None)
            if i is not None and i < len(self.latest_msg.position):
                pos[k] = self.latest_msg.position[i]

        # # 重力补偿
        tau = GravityForces(pos, self.g, self.Mlist, self.Glist, self.Slist)

        # 发布控制命令
        msg_out = Float64MultiArray()
        msg_out.data = tau.tolist()
        self.pub.publish(msg_out)

        # 打印
        self.get_logger().info(f'关节角度: {np.round(pos,3)}')
        self.get_logger().info(f'发布重力补偿扭矩: {np.round(tau,3)}')


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


if __name__ == '__main__':
    main()














# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import JointState
# import numpy as np
# import modern_robotics as mr

# class JointStateListener(Node):
    
#     def __init__(self):
#         super().__init__('joint_state_listener')

#         self.M01 = np.array([[1, 0, 0,      0],
#                              [0, 1, 0,      0],
#                              [0, 0, 1,  0.079],
#                              [0, 0, 0,      1]])
        
#         self.M12 = np.array([[ 1, 0, 0,      0],
#                              [ 0, 1, 0,      0],
#                              [ 0, 0, 1,  0.048],
#                              [ 0, 0, 0,      1]])
        
#         self.M23 = np.array([[1, 0, 0,   0.06],
#                              [0, 1, 0,      0],
#                              [0, 0, 1,    0.3],
#                              [0, 0, 0,      1]])
       
#         self.M34 = np.array([[1, 0, 0,   0.2],
#                              [0, 1, 0,     0],
#                              [0, 0, 1,     0],
#                              [0, 0, 0,     1]])

#         self.M45 = np.array([[1, 0, 0,   0.1],
#                              [0, 1, 0,     0],
#                              [0, 0, 1,     0],
#                              [0, 0, 0,     1]])

#         self.M56 = np.array([[1, 0, 0,    0.07],
#                              [0, 1, 0,       0],
#                              [0, 0, 1,       0],
#                              [0, 0, 0,       1]])

#         self.M67 = np.array([[1, 0, 0,    0.107],
#                              [0, 1, 0,        0],
#                              [0, 0, 1,        0],
#                              [0, 0, 0,        1]])
        
#         self.Mlist = np.array([self.M01, self.M12, self.M23, self.M34, self.M45, self.M56, self.M67])

#         # 1. waist link
#         self.G1 = np.diag([0.006024, 0.001700, 0.007162,
#               0.969034, 0.969034, 0.969034])

#         # 2. shoulder link
#         self.G2 = np.diag([0.0009388, 0.001138, 0.001201,
#               0.798614, 0.798614, 0.798614])

#         # 3. upper arm link
#         self.G3 = np.diag([0.008925, 0.008937, 0.0009357,
#               0.792592, 0.792592, 0.792592])

#         # 4. upper forearm link
#         self.G4 = np.diag([0.0001524, 0.001342, 0.001441,
#               0.322228, 0.322228, 0.322228])

#         # 5. lower forearm link
#         self.G5 = np.diag([0.0001753, 0.0005269, 0.0005911,
#               0.414823, 0.414823, 0.414823])

#         # 6. wrist link
#         self.G6 = np.diag([4.631e-5, 4.514e-5, 5.27e-5,
#               0.115395, 0.115395, 0.115395])

#         # Collect into Glist
#         self.Glist = np.array([self.G1, self.G2, self.G3, self.G4, self.G5, self.G6])


#         self.Slist = np.array([[0.0, 0.0, 1.0,  0.0,     0.0,     0.0],
#                   [0.0, 1.0, 0.0, -0.12705, 0.0,     0.0],
#                   [0.0, 1.0, 0.0, -0.42705, 0.0,     0.05955],
#                   [1.0, 0.0, 0.0,  0.0,     0.42705, 0.0],
#                   [0.0, 1.0, 0.0, -0.42705, 0.0,     0.35955],
#                   [1.0, 0.0, 0.0,  0.0,     0.42705, 0.0]]).T

#         # 期望的关节顺序（也可以通过 ros2 param 设置）
#         self.declare_parameter(
#             'desired_joint_order',
#             ['waist','shoulder','elbow','forearm_roll','wrist_angle','wrist_rotate']
#         )
#         self.desired = list(self.get_parameter('desired_joint_order').value)

#         self.subscription = self.create_subscription(
#             JointState, '/joint_states', self.joint_state_callback, 10
#         )
#         # 定时打印频率：10 Hz
#         self.timer = self.create_timer(0.1, self.timer_callback)

#         self.latest_msg = None

#     def joint_state_callback(self, msg: JointState):
#         self.latest_msg = msg

#     def timer_callback(self):
#         if self.latest_msg is None:
#             return

#         # name -> index 映射（来自消息原始顺序）
#         name2idx = {n: i for i, n in enumerate(self.latest_msg.name)}

#         # 只取期望顺序中的前6个（如果期望里本就6个，就等价）
#         wanted = self.desired[:6]

#         # 预分配并按期望顺序填充
#         names = []
#         pos = np.zeros(len(wanted))
#         vel = np.zeros(len(wanted))
#         eff = np.zeros(len(wanted))

#         for k, jname in enumerate(wanted):
#             if jname not in name2idx:
#                 # 找不到就提示 + 填 0（也可换成 np.nan）
#                 self.get_logger().warn_once(f'Joint "{jname}" not found in /joint_states; check naming.')
#                 names.append(jname)
#                 pos[k] = 0.0
#                 vel[k] = 0.0
#                 eff[k] = 0.0
#             else:
#                 i = name2idx[jname]
#                 names.append(jname)
#                 # 防止缺字段：用条件取值
#                 pos[k] = self.latest_msg.position[i] if i < len(self.latest_msg.position) else 0.0
#                 vel[k] = self.latest_msg.velocity[i] if i < len(self.latest_msg.velocity) else 0.0
#                 eff[k] = self.latest_msg.effort[i]   if i < len(self.latest_msg.effort)   else 0.0

#         # 保留三位小数
#         pos = np.round(pos, 3)
#         vel = np.round(vel, 3)
#         eff = np.round(eff, 3)

#         # 打印
#         self.get_logger().info(f'关节(前6,按期望顺序): {names}')
#         self.get_logger().info(f'位置: {pos}')
#         self.get_logger().info(f'速度: {vel}')
#         self.get_logger().info(f'力矩: {eff}')
#         self.get_logger().info('---')


# def main(args=None):
#     rclpy.init(args=args)
#     node = JointStateListener()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()
