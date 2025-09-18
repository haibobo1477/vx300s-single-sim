#!/usr/bin/env python3
import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import pinocchio as pin
import numpy as np

# ====== Pinocchio 模型构建 ======
urdf_model_path = "/home/yc/vx300s-single-sim/src/vx300s_description/urdf/vx300s.urdf"
mesh_dir = "/home/yc/vx300s-single-sim/src/vx300s_description/vx300s_meshes/"

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path,
    mesh_dir
)
data = model.createData()

# 只取 6 个主要关节（跳过 gripper 和 fingers）
# 注意：这里按你的 URDF 结构 Joint 1~6 对应 waist ~ wrist_rotate
controlled_joints = ["waist", "shoulder", "elbow",
                     "forearm_roll", "wrist_angle", "wrist_rotate"]

# ====== 回调函数：订阅 joint_states ======
def joint_state_callback(msg, node, publisher):
    # msg.position 顺序必须和 model.names 对应
    name_to_pos = dict(zip(msg.name, msg.position))

    # 构造完整的 q（长度 = model.nq）
    q = pin.neutral(model)   # 先用默认值填充
    for jname, jpos in name_to_pos.items():
        if jname in model.names:
            idx = model.getJointId(jname)
            if idx < model.nq:
                q[idx - 1] = jpos   # idx-1 因为 joint 0 是 universe

    # 计算重力补偿
    G = pin.computeGeneralizedGravity(model, data, q)

    # 只取前 6 个关节的力矩
    effort_msg = Float64MultiArray()
    effort_msg.data = [G[model.getJointId(j)-1] for j in controlled_joints]

    publisher.publish(effort_msg)
    node.get_logger().info(f"Published gravity torques: {effort_msg.data}")


def main(args=None):
    rclpy.init(args=args)

    # 创建节点
    node = rclpy.create_node("gravity_comp_node")

    # 创建发布者
    effort_pub = node.create_publisher(Float64MultiArray, "/arm_controller/commands", 10)

    # 创建订阅者
    node.create_subscription(
        JointState,
        "/joint_states",
        lambda msg: joint_state_callback(msg, node, effort_pub),
        10
    )

    node.get_logger().info("Gravity compensation node started.")
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

# # 你的 URDF 路径
# urdf_model_path = "/home/yc/vx300s-single-sim/src/vx300s_description/urdf/vx300s.urdf"
# mesh_dir = "/home/yc/vx300s-single-sim/src/vx300s_description/vx300s_meshes/"

# # 构建模型
# # robot = RobotWrapper.BuildFromURDF(urdf_model_path, mesh_dir)
# model, collision_model, visual_model = pin.buildModelsFromUrdf(
#     urdf_model_path,
#     mesh_dir
# )

# # 可视化器
# # viz = MeshcatVisualizer(model, collision_model, visual_model)

# # viz.initViewer(open=True)       # 打开浏览器窗口
# # viz.loadViewerModel()           # 加载模型
# # viz.display(pin.neutral(model)) 
# # viz.viewer.open()

# data = model.createData()
# '''
# Nb joints = 10 (nq=10,nv=9)
#   Joint 0 universe: parent=0
#   Joint 1 waist: parent=0
#   Joint 2 shoulder: parent=1
#   Joint 3 elbow: parent=2
#   Joint 4 forearm_roll: parent=3
#   Joint 5 wrist_angle: parent=4
#   Joint 6 wrist_rotate: parent=5
#   Joint 7 gripper: parent=6
#   Joint 8 left_finger: parent=6
#   Joint 9 right_finger: parent=6
# '''

# q = pin.randomConfiguration(model)   # 随机一个姿态
# v = np.random.randn(model.nv)        # 随机速度

# G = pin.computeGeneralizedGravity(model, data, q)


# print("关节位置 q =", q)
# print("重力补偿力矩 G(q) =", G)