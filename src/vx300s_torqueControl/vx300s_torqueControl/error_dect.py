#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import matplotlib.pyplot as plt

class JointErrorPlotter(Node):
    def __init__(self):
        super().__init__('joint_error_plotter')

        # ===  ===
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/vx300s/e',
            self.error_callback,
            10
        )

        # ===  ===
        self.n_joints = 6
        self.time_data = []                        # 
        self.error_data = [[] for _ in range(self.n_joints)]

        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        # === ===
        plt.ion()
        self.fig, self.axs = plt.subplots(self.n_joints, 1, figsize=(8, 10), sharex=True)
        self.lines = []
        joint_names = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]

        for i in range(self.n_joints):
            line, = self.axs[i].plot([], [], label=f"Joint {i+1} ({joint_names[i]})")
            self.axs[i].set_ylabel("Error (rad)")
            self.axs[i].grid(True)
            self.axs[i].legend(loc='upper right')
            self.lines.append(line)
        self.axs[-1].set_xlabel("Time (s)")

        # === timer(10Hz) ===
        self.timer = self.create_timer(0.1, self.update_plot)
        self.get_logger().info("✅ Joint Error Plotter Node Started — Subscribed to /vx300s/e")

    def error_callback(self, msg: Float64MultiArray):
        """"""
        t_now = self.get_clock().now().nanoseconds * 1e-9 - self.start_time
        self.time_data.append(t_now)

        e = np.array(msg.data, dtype=float)
        if len(e) != self.n_joints:
            self.get_logger().warn(f"Error length mismatch: {len(e)} (expected {self.n_joints})")
            return

        for i in range(self.n_joints):
            self.error_data[i].append(e[i])

    def update_plot(self):
        """"""
        if len(self.time_data) < 5:
            return  # 

        t = np.array(self.time_data)
        for i in range(self.n_joints):
            e_i = np.array(self.error_data[i])
            self.lines[i].set_data(t, e_i)
            # 
            self.axs[i].set_xlim(0, max(t))
            # 
            if len(e_i) > 10 and len(t) % 20 == 0:
                ymin, ymax = np.min(e_i), np.max(e_i)
                margin = 0.1 * (ymax - ymin + 1e-6)
                self.axs[i].set_ylim(ymin - margin, ymax + margin)

        plt.pause(0.001)
        plt.draw()


def main(args=None):
    rclpy.init(args=args)
    node = JointErrorPlotter()
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







