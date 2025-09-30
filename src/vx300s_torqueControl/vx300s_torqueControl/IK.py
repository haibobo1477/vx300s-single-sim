from sympy import symbols, sin, cos, pi, atan2, sqrt
from sympy.matrices import Matrix
import numpy as np

# ----------------- 工具函数 -----------------
def get_hypotenuse(a, b):
    return sqrt(a*a + b*b)

def get_cosine_law_angle(a, b, c):    
    cos_gamma = (a*a + b*b - c*c) / (2*a*b)
    sin_gamma = sqrt(1 - cos_gamma * cos_gamma)
    return atan2(sin_gamma, cos_gamma)

def get_wrist_center(gripper_point, R0g, dg=0.1387):
    xu, yu, zu = gripper_point 
    nx, ny, nz = R0g[0, 2], R0g[1, 2], R0g[2, 2]
    xw = xu - dg * nx
    yw = yu - dg * ny
    zw = zu - dg * nz 
    return xw, yw, zw

def get_first_three_angles(wrist_center):
    x, y, z = wrist_center
    Lm, L2, L3 = 0.06, 0.3, 0.3
    d1 = 0.12675
    r = get_hypotenuse(x, y)
    h = z - d1 
    Lr = get_hypotenuse(L2, Lm)
    C  = get_hypotenuse(r, h)

    beta  = atan2(Lm, L2)
    psi   = pi/2 - beta
    phi   = get_cosine_law_angle(Lr, L3, C)
    gamma = atan2(h, r)
    alpha = get_cosine_law_angle(Lr, C, L3)

    q1 = atan2(y, x)
    q2 = pi/2 - beta - alpha - gamma
    q3 = pi - psi - phi
    return q1, q2, q3 

def get_last_three_angles(R):
    sin_q4, cos_q4 = R[2, 2], -R[0, 2]
    sin_q5, cos_q5 = sqrt(R[0, 2]**2 + R[2, 2]**2), R[1, 2]
    sin_q6, cos_q6 = -R[1, 1], R[1, 0]
    q4 = atan2(sin_q4, cos_q4)
    q5 = atan2(sin_q5, cos_q5)
    q6 = atan2(sin_q6, cos_q6)
    return q4, q5, q6

# ----------------- 逆运动学主函数 -----------------
def get_angles(x, y, z, roll, pitch, yaw):
    gripper_point = (x, y, z)
    q1, q2, q3, q4, q5, q6 = symbols('q1:7')

    # 基于欧拉角计算旋转矩阵 R0u
    alpha, beta, gamma = symbols('alpha beta gamma')
    R0u = Matrix([
        [cos(alpha)*cos(beta), -sin(alpha)*cos(gamma)+cos(alpha)*sin(beta)*sin(gamma), sin(alpha)*sin(gamma)+cos(alpha)*sin(beta)*cos(gamma)],
        [sin(alpha)*cos(beta),  cos(alpha)*cos(gamma)+sin(alpha)*sin(beta)*sin(gamma),-cos(alpha)*sin(gamma)+sin(alpha)*sin(beta)*cos(gamma)],
        [-sin(beta),            cos(beta)*sin(gamma),                                   cos(beta)*cos(gamma)]
    ])

    RguT = Matrix([[0,0,1],[0,-1,0],[1,0,0]])  # URDF → DH 坐标变换
    R0u_eval = R0u.evalf(subs={alpha:yaw, beta:pitch, gamma:roll})
    R0g_eval = R0u_eval * RguT

    # wrist center
    wrist_center = get_wrist_center(gripper_point, R0g_eval)
    j1, j2, j3 = get_first_three_angles(wrist_center)

    # R03 转置
    R03T = Matrix([
        [sin(q2+q3)*cos(q1), sin(q2+q3)*sin(q1), cos(q2+q3)],
        [cos(q2+q3)*cos(q1), cos(q2+q3)*sin(q1),-sin(q2+q3)],
        [-sin(q1),           cos(q1),            0]
    ])
    R03T_eval = R03T.evalf(subs={q1:j1.evalf(), q2:j2.evalf(), q3:j3.evalf()})

    # R36
    R36_eval = R03T_eval * R0g_eval
    j4, j5, j6 = get_last_three_angles(R36_eval)

    return float(j1), float(j2), float(j3), float(j4), float(j5), float(j6)

# # ----------------- Demo -----------------
# if __name__ == "__main__":
#     px, py, pz = 0.6, 0.06, 0.059
#     roll, pitch, yaw = 2.9, 1.2, 1.5
#     q = get_angles(px, py, pz, roll, pitch, yaw)
#     print("Inverse kinematics solution:", q)
