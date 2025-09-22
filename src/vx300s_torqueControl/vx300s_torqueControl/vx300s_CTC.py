import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import time
import meshcat
from pinocchio import FrameType



import pinocchio as pin
import numpy as np
from pinocchio import FrameType



# 你的 URDF 路径
# urdf_model_path = "/home/yc/vx300s-single-sim/src/vx300s_description/urdf/vx300s.urdf"
# mesh_dir = "/home/yc/vx300s-single-sim/src/vx300s_description/vx300s_meshes/"


urdf_model_path = "/home/haibo/vx300s_ws/src/vx300s_description/urdf/vx300s_fix.urdf"
mesh_dir = "/home/haibo/vx300s_ws/src/vx300s_description/vx300s_meshes/"

# 构建模型
# robot = RobotWrapper.BuildFromURDF(urdf_model_path, mesh_dir)
model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path,
    mesh_dir
)


print(model.gravity)

data = model.createData()
print(data)



# q = pin.randomConfiguration(model)   # 随机一个姿态
# q = np.array([0, 0, 0, 0, 0, 0, 0.8, 0, 0, 0]) 
# print(q)
q = pin.neutral(model)
# print(q)
v = np.random.randn(model.nv)        # 随机速度
# print(v)



masses = {}
for fid, frame in enumerate(model.frames):
    if frame.type == FrameType.BODY:
        jidx = frame.parent            # 该BODY隶属的joint索引
        I = model.inertias[jidx]       # 该link的惯性
        masses[frame.name] = float(I.mass)

# 打印
total_mass = 0.0
for name, m in masses.items():
    print(f"{name:30s}  mass = {m:.6f} kg")
    total_mass += m
print(f"\nTotal mass = {total_mass:.6f} kg")


# 可视化器
# viz = MeshcatVisualizer(model, collision_model, visual_model)

# viz.initViewer(open=True)       # 打开浏览器窗口
# viz.loadViewerModel()           # 加载模型
# viz.display(q) 
# viz.viewer.open()


M = pin.crba(model, data, q)   # Composite Rigid Body Algorithm
M = (M + M.T) / 2.0            # 数值对称化（避免小误差）

G = pin.computeGeneralizedGravity(model, data, q)

C = pin.computeCoriolisMatrix(model, data, q, v)
# print(M.shape)
# print(C.shape)
# print(G)
# print(q)
print(model)