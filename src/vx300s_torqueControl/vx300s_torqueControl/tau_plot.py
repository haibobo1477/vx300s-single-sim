#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ======  ======
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#8c564b"]

class TauPlotNode(Node):
    def __init__(self):
        super().__init__('tau_plot_node')
        self.get_logger().info("✅ Tau Plot Node started. Subscribing to /arm_controller/commands")

        # ===  ===
        self.sub = self.create_subscription(Float64MultiArray, "/arm_controller/commands", self.tau_callback, 10)

        # ===  ===
        self.time_data = []
        self.tau_data = defaultdict(list)
        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        # ===  ===
        plt.ion()
        self.fig, self.ax = plt.subplots(3, 2, figsize=(10, 7))
        self.fig.suptitle("Joint Torques (τ) - Full History", fontsize=14)

        self.lines = []
        for i, axis in enumerate(self.ax.flat):
            line, = axis.plot([], [], lw=2, label=JOINT_NAMES[i], color=COLORS[i])
            axis.set_title(JOINT_NAMES[i])
            axis.set_xlabel("Time [s]")
            axis.set_ylabel("Torque [Nm]")
            axis.grid(True)
            axis.legend(loc="upper right")
            self.lines.append(line)

        # ===  ===
        self.timer = self.create_timer(0.05, self.update_plot)  # 20Hz

    def tau_callback(self, msg: Float64MultiArray):
        """"""
        t_now = self.get_clock().now().nanoseconds * 1e-9 - self.start_time
        tau = msg.data
        if len(tau) != 6:
            return
        self.time_data.append(t_now)
        for i in range(6):
            self.tau_data[i].append(tau[i])

    def update_plot(self):
        """"""
        if len(self.time_data) < 2:
            return
        t = np.array(self.time_data)

        
        for i, axis in enumerate(self.ax.flat):
            tau_i = np.array(self.tau_data[i])
            self.lines[i].set_data(t, tau_i)

            axis.set_xlim(0, t[-1] + 0.1)
            axis.set_ylim(np.min(tau_i) - 0.2, np.max(tau_i) + 0.2)

        plt.tight_layout()
        plt.pause(0.001)

def main(args=None):
    rclpy.init(args=args)
    node = TauPlotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 Plotting stopped by user.")
    finally:
        plt.ioff()
        plt.show()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
