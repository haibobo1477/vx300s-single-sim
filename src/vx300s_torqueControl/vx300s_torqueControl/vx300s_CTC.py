import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import time

# 你的 URDF 路径
# urdf_model_path = "/home/yc/vx300s-single-sim/src/vx300s_description/urdf/vx300s.urdf"
# mesh_dir = "/home/yc/vx300s-single-sim/src/vx300s_description/vx300s_meshes/"


urdf_model_path = "/home/haibo/vx300s_ws/src/vx300s_description/urdf/vx300s.urdf"
mesh_dir = "/home/haibo/vx300s_ws/src/vx300s_description/vx300s_meshes/"

# 构建模型
# robot = RobotWrapper.BuildFromURDF(urdf_model_path, mesh_dir)
model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path,
    mesh_dir,
    root_joint_name="waist"
)

# 可视化器
# viz = MeshcatVisualizer(model, collision_model, visual_model)

# viz.initViewer(open=True)       # 打开浏览器窗口
# viz.loadViewerModel()           # 加载模型
# viz.display(pin.neutral(model)) 
# viz.viewer.open()


data = model.createData()
# q = pin.randomConfiguration(model)   # 随机一个姿态
q = pin.neutral(model)
v = np.random.randn(model.nv)        # 随机速度

M = pin.crba(model, data, q)   # Composite Rigid Body Algorithm
M = (M + M.T) / 2.0            # 数值对称化（避免小误差）

G = pin.computeGeneralizedGravity(model, data, q)

C = pin.computeCoriolisMatrix(model, data, q, v)
# print(M.shape)
# print(C.shape)
print(G.shape)
print(q.shape)
print(model)