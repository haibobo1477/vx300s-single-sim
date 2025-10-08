#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from collections import deque
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]

class EPlotterNode(Node):
    def __init__(self):
        super().__init__('e_plotter_node')

        # 订阅误差话题
        self.sub = self.create_subscription(Float64MultiArray, '/vx300s/e', self.cb_e, 50)

        # 缓冲区（保存最近 N 秒的数据）
        self.window_seconds = 30.0
        self.sample_hz = 50.0
        self.maxlen = int(self.window_seconds * self.sample_hz)

        self.t0 = time.time()
        self.ts = deque(maxlen=self.maxlen)
        self.es = [deque(maxlen=self.maxlen) for _ in range(6)]

        # ==== 建立一张包含 6 个子图的 Figure ====
        self.fig, self.axes = plt.subplots(3, 2, figsize=(10, 8))
        self.fig.suptitle("Joint Error e(t)", fontsize=14)

        self.lines = []
        for i, ax in enumerate(self.axes.flat):
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(f"e_{i+1} (rad)")
            ax.grid(True)
            line, = ax.plot([], [], lw=1.5)
            self.lines.append(line)

        # 定时刷新绘图
        self.timer = self.create_timer(0.1, self.timer_update_plot)

        # 文件名（保存图像时使用时间戳）
        self.output_file = os.path.join(
            os.path.expanduser("~"),
            f"joint_error_plot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        )

    # ====== 回调函数 ======
    def cb_e(self, msg: Float64MultiArray):
        now = time.time() - self.t0
        self.ts.append(now)

        data = np.array(msg.data, dtype=float)
        if data.shape[0] != 6:
            self.get_logger().warn(f"Expected 6-dim e, got shape {data.shape}")
            return
        for i in range(6):
            self.es[i].append(data[i])

    # ====== 定时刷新图像 ======
    def timer_update_plot(self):
        if len(self.ts) < 2:
            return

        t = np.array(self.ts)
        t_min = max(0.0, t[-1] - self.window_seconds)

        for i, ax in enumerate(self.axes.flat):
            y = np.array(self.es[i])
            self.lines[i].set_data(t, y)
            if y.size > 0:
                y_min, y_max = float(np.min(y)), float(np.max(y))
                if y_min == y_max:
                    y_min -= 1e-4
                    y_max += 1e-4
                ax.set_xlim(t_min, t[-1] + 0.01)
                ax.set_ylim(y_min - 0.05 * abs(y_min if y_min != 0 else 1.0),
                            y_max + 0.05 * abs(y_max if y_max != 0 else 1.0))
            ax.figure.canvas.draw()
        plt.pause(0.001)

    # ====== 程序结束时保存图像 ======
    def save_plot(self):
        for i, ax in enumerate(self.axes.flat):
            ax.relim()
            ax.autoscale_view()
        self.fig.tight_layout(rect=[0, 0, 1, 0.97])
        self.fig.savefig(self.output_file, dpi=200)
        self.get_logger().info(f"✅ Figure saved to {self.output_file}")

def main(args=None):
    rclpy.init(args=args)
    node = EPlotterNode()
    try:
        plt.ion()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        node.save_plot()   # 自动保存图像
        plt.show(block=False)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
