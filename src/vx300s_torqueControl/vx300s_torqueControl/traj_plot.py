#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import tf2_ros
import tf_transformations
import matplotlib.pyplot as plt
from std_msgs.msg import Float64MultiArray


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


def generate_line_trajectory(t, T_total=6.0):
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
        t_local = t_mod - T_total

    s_coeffs = quintic_coeffs(s0, 0, 0, sf, 0, 0, T_total)
    s, ds, dds = quintic_eval(s_coeffs, np.clip(t_local, 0, T_total))

    if direction > 0:
        Pd = P0 + s * dP_dir
    else:
        Pd = Pf - s * dP_dir

    pos = np.concatenate(([roll, pitch, yaw], Pd))
    return pos


def generate_ellipse_trajectory(t, T_total=6.0):
    Pc = np.array([0.3, 0.0, 0.3])  # 
    a = 0.15  # 
    b = 0.05  # 
    roll, pitch, yaw = 0.0, np.pi/4, 0.0

    theta_coeffs = quintic_coeffs(0, 0, 0, 2*np.pi, 0, 0, T_total)
    theta, dtheta, ddtheta = quintic_eval(theta_coeffs, t % T_total)

    
    x = Pc[0] + a * np.cos(theta)
    y = Pc[1] + b * np.sin(theta)
    z = Pc[2]

    pos = np.array([roll, pitch, yaw, x, y, z])
    return pos


class TFEEPlotNode(Node):
    def __init__(self):
        super().__init__('tf_ee_plot_node')

        # === TF ===
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.source_frame = 'vx300s/base_link'
        self.target_frame = 'vx300s/ee_gripper_link'

        # === 'line' or 'ellipse'）===
        self.traj_type = 'line'   
        # self.traj_type = 'ellipse'

        self.start_time = self.get_clock().now().nanoseconds * 1e-9
        self.x_data, self.y_data, self.z_data = [], [], []

        # ===  ===
        plt.ion()
        self.fig = plt.figure("End-Effector Trajectory (TF + Reference)", figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # 
        self.tf_line, = self.ax.plot([], [], [], 'b-', label='Actual (TF)')
        self.tf_point, = self.ax.plot([], [], [], 'ro', label='Current Pose')

        # 
        self.plot_reference_trajectory()

        # 
        self.ax.set_xlim(0.2, 0.5)
        self.ax.set_ylim(-0.3, 0.3)
        self.ax.set_zlim(0.0, 0.6)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.legend()
        self.ax.grid(True)

        # 
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.logged_once = False

        self.get_logger().info(f"✅ TF EE Plot Node Started — Listening [{self.source_frame} → {self.target_frame}]")

    def plot_reference_trajectory(self):
        """"""
        t_list = np.linspace(0, 6, 300)

        if self.traj_type == 'line':
            ref_points = np.array([generate_line_trajectory(t)[3:] for t in t_list])
            label = 'Reference Trajectory'
            color = 'r'
        else:
            ref_points = np.array([generate_ellipse_trajectory(t)[3:] for t in t_list])
            label = 'Reference Trajectory'
            color = 'r'

        x_ref, y_ref, z_ref = ref_points[:, 0], ref_points[:, 1], ref_points[:, 2]
        self.ax.plot(x_ref, y_ref, z_ref, '--', color=color, linewidth=1.8, label=label)

    def timer_callback(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.source_frame, self.target_frame, rclpy.time.Time()
            )
            t = trans.transform.translation
            x, y, z = t.x, t.y, t.z

            # 
            self.x_data.append(x)
            self.y_data.append(y)
            self.z_data.append(z)

            # 
            self.tf_line.set_data(self.x_data, self.y_data)
            self.tf_line.set_3d_properties(self.z_data)
            self.tf_point.set_data([x], [y])
            self.tf_point.set_3d_properties([z])

            self.ax.set_title(f"End-Effector TF vs Reference\nx={x:.3f}, y={y:.3f}, z={z:.3f}")
            plt.pause(0.001)

            if not self.logged_once:
                self.get_logger().info(f"✅ Plotting EE trajectory with {self.traj_type} reference.")
                self.logged_once = True

        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException):
            pass


def main(args=None):
    rclpy.init(args=args)
    node = TFEEPlotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        plt.ioff()
        plt.show()
        rclpy.shutdown()


if __name__ == '__main__':
    main()



