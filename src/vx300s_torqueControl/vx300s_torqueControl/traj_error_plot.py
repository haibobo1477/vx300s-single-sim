#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import tf2_ros
import tf_transformations
import matplotlib.pyplot as plt


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
    a = 0.15  # y
    b = 0.05  # z
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

        # === TF init ===
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.source_frame = 'vx300s/base_link'
        self.target_frame = 'vx300s/ee_gripper_link'

        # === type ===
        self.traj_type = 'line'   # or 'ellipse'

        # self.traj_type = 'ellipse'

        self.start_time = self.get_clock().now().nanoseconds * 1e-9
        self.t_data, self.x_data, self.y_data, self.z_data = [], [], [], []

        # ===  ===
        plt.ion()
        self.fig = plt.figure("End-Effector Trajectory Visualization", figsize=(10, 8))

        # 1：3D
        self.ax3d = self.fig.add_subplot(2, 1, 1, projection='3d')
        self.tf_line, = self.ax3d.plot([], [], [], 'b-', label='Actual (TF)')
        self.tf_point, = self.ax3d.plot([], [], [], 'ro', label='Current Pose')
        self.plot_reference_trajectory(self.ax3d)

        self.ax3d.set_xlim(0.2, 0.5)
        self.ax3d.set_ylim(-0.3, 0.3)
        self.ax3d.set_zlim(0.0, 0.6)
        self.ax3d.set_xlabel('X (m)')
        self.ax3d.set_ylabel('Y (m)')
        self.ax3d.set_zlabel('Z (m)')
        self.ax3d.legend()
        self.ax3d.grid(True)

        # 2：X/Y/Z 
        self.ax_xyz = self.fig.add_subplot(2, 1, 2)
        self.line_x, = self.ax_xyz.plot([], [], 'r-', label='X (m)')
        self.line_y, = self.ax_xyz.plot([], [], 'g-', label='Y (m)')
        self.line_z, = self.ax_xyz.plot([], [], 'b-', label='Z (m)')
        self.ax_xyz.set_xlabel("Time (s)")
        self.ax_xyz.set_ylabel("Position (m)")
        self.ax_xyz.set_title("End-Effector Position over Time")
        self.ax_xyz.grid(True)
        self.ax_xyz.legend(loc='upper right')

        # timer
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.logged_once = False

        self.get_logger().info(f"✅ TF EE Plot Node Started — Listening [{self.source_frame} → {self.target_frame}]")

    def plot_reference_trajectory(self, ax):
        """"""
        t_list = np.linspace(0, 6, 300)
        if self.traj_type == 'line':
            ref_points = np.array([generate_line_trajectory(t)[3:] for t in t_list])
        else:
            ref_points = np.array([generate_ellipse_trajectory(t)[3:] for t in t_list])
        ax.plot(ref_points[:,0], ref_points[:,1], ref_points[:,2], '--r', linewidth=1.8, label='Reference Trajectory')

    def timer_callback(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.source_frame, self.target_frame, rclpy.time.Time()
            )
            t_now = self.get_clock().now().nanoseconds * 1e-9 - self.start_time
            t = trans.transform.translation
            x, y, z = t.x, t.y, t.z

            # 
            self.t_data.append(t_now)
            self.x_data.append(x)
            self.y_data.append(y)
            self.z_data.append(z)

            # ===  3D ===
            self.tf_line.set_data(self.x_data, self.y_data)
            self.tf_line.set_3d_properties(self.z_data)
            self.tf_point.set_data([x], [y])
            self.tf_point.set_3d_properties([z])

            # ===  X/Y/Z  ===
            self.line_x.set_data(self.t_data, self.x_data)
            self.line_y.set_data(self.t_data, self.y_data)
            self.line_z.set_data(self.t_data, self.z_data)
            self.ax_xyz.set_xlim(0, max(self.t_data) + 0.5)

            # 
            xyz_all = np.array([self.x_data, self.y_data, self.z_data])
            ymin, ymax = np.min(xyz_all), np.max(xyz_all)
            margin = 0.05 * (ymax - ymin + 1e-6)
            self.ax_xyz.set_ylim(ymin - margin, ymax + margin)

            # 
            self.ax3d.set_title(
                f"End-Effector TF vs Reference\nx={x:.3f}, y={y:.3f}, z={z:.3f}"
            )

            plt.pause(0.001)

            if not self.logged_once:
                self.get_logger().info(f"✅ Plotting EE trajectory ({self.traj_type}) with XYZ time curves.")
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

def main(args=None):
    rclpy.init(args=args)
    node = TFEEPlotNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        plt.ioff()
        plt.show()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

