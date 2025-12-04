#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import matplotlib.pyplot as plt


class TrajectoryPlotNode(Node):
    def __init__(self):
        super().__init__("trajectory_plot_node")

        # 
        self.t_list = []        # 
        self.q_des_list = []    # 
        self.q_list = []        # 

        self.last_q = None      # 
        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        # 
        self.sub_q = self.create_subscription(
            Float64MultiArray,
            "/vx300s/q",
            self.q_callback,
            10
        )
        self.sub_q_des = self.create_subscription(
            Float64MultiArray,
            "/vx300s/q_des",
            self.q_des_callback,
            10
        )

        self.get_logger().info("TrajectoryPlotNode started, waiting for /vx300s/q and /vx300s/q_des ...")

    def q_callback(self, msg: Float64MultiArray):
        # 
        # 
        self.last_q = np.array(msg.data, dtype=float)

    def q_des_callback(self, msg: Float64MultiArray):
        #
        if self.last_q is None:
            # 
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        t = now - self.start_time

        q_des = np.array(msg.data, dtype=float)

        self.t_list.append(t)
        self.q_des_list.append(q_des)
        self.q_list.append(self.last_q.copy())

    def plot_results(self):
        if len(self.t_list) == 0:
            self.get_logger().warn("No data collected, nothing to plot.")
            return

        t = np.array(self.t_list)
        q_des_arr = np.vstack(self.q_des_list)   # shape: [N, 6]
        q_arr = np.vstack(self.q_list)           # shape: [N, 6]

        joint_labels = [r"$q_1$", r"$q_2$", r"$q_3$", r"$q_4$", r"$q_5$", r"$q_6$"]

        fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
        axes = axes.flatten()

        for i in range(6):
            ax = axes[i]
            ax.plot(t, q_des_arr[:, i], linestyle="--", label=f"{joint_labels[i]} des")
            ax.plot(t, q_arr[:, i], label=f"{joint_labels[i]} act")
            ax.set_ylabel("rad")
            ax.legend()
            ax.grid(True)

        axes[-1].set_xlabel("time [s]")
        fig.suptitle("Joint Trajectory Tracking (reference vs actual)")
        fig.tight_layout()
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPlotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.plot_results()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
