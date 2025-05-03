import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.models import load_model
import joblib
from visual_kinematics.RobotSerial import RobotSerial
import math

# Load model and scalers
model = load_model("ik_lstm_2d.h5")
x_scaler = joblib.load("x_scaler_2d.pkl")
y_scaler = joblib.load("y_scaler_2d.pkl")

# DH parameters for 2DOF
dh_params = np.array([[0, 0.165, 0, 0],
                      [0, 0.12, 0, 0]])
robot = RobotSerial(dh_params)

def predict_ik_from_pose(x, y, z, r11, r21, r31, r12, r22, r32, r13, r23, r33):
    input_vec = np.array([x, y, z, r11, r21, r31, r12, r22, r32, r13, r23, r33]).reshape(1, -1)
    input_scaled = x_scaler.transform(input_vec)
    input_reshaped = input_scaled.reshape((1, 1, 12))
    pred_scaled = model.predict(input_reshaped)
    return y_scaler.inverse_transform(pred_scaled)[0]

def fk_from_angles(q):
    theta = np.radians(q)
    f = robot.forward(theta)
    return f.t_3_1.reshape([3])

def ik_2dof(x, y, L1=0.165, L2=0.12):
    D = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if np.abs(D) > 1.0:
        raise ValueError("Target is out of reach")
    theta2 = np.arccos(D)
    k1 = L1 + L2 * np.cos(theta2)
    k2 = L2 * np.sin(theta2)
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    return theta1, theta2

class TrajectoryApp:
    def __init__(self, master):
        self.master = master
        self.master.title("2DOF Trajectory Planner")

        self.points = []
        self.L1 = 0.165
        self.L2 = 0.12
        self.N = 2

        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.ax.set_title("Click to add waypoints")
        self.ax.set_xlim(-0.3, 0.3)
        self.ax.set_ylim(0, 0.3)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack()
        self.cid = self.canvas.mpl_connect('button_press_event', self.onclick)

        tk.Button(master, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=10)
        tk.Button(master, text="Execute", command=self.execute).pack(side=tk.LEFT, padx=10)

    def onclick(self, event):
        if event.inaxes:
            self.points.append([event.xdata, event.ydata])
            self.ax.plot(event.xdata, event.ydata, 'ro')
            self.canvas.draw()

    def clear(self):
        self.points = []
        self.ax.cla()
        self.ax.set_title("Click to add waypoints")
        self.ax.set_xlim(-0.3, 0.3)
        self.ax.set_ylim(0, 0.3)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.canvas.draw()

    def load_points(self, points_list):
        self.clear()
        for x, y in points_list:
            self.points.append([x, y])
            self.ax.plot(x, y, 'ro')
        self.canvas.draw()

    def execute(self):
        if len(self.points) < 2:
            messagebox.showerror("Error", "Please click at least two points.")
            return

        desired_positions = []
        recon_positions = []

        for idx in range(len(self.points) - 1):
            x1, y1 = self.points[idx]
            x2, y2 = self.points[idx + 1]
            try:
                sol1 = ik_2dof(x1, y1, self.L1, self.L2)
                sol2 = ik_2dof(x2, y2, self.L1, self.L2)
            except:
                messagebox.showwarning("Warning", f"Point pair {idx}-{idx+1} is out of reach.")
                continue

            pose_A = self.build_pose(x1, y1, sol1)
            pose_B = self.build_pose(x2, y2, sol2)

            s_vals = np.linspace(0, 1, self.N)
            for s in s_vals:
                pose = pose_A + s * (pose_B - pose_A)
                x, y, z = pose[:3]
                rot = pose[3:]
                q = predict_ik_from_pose(x, y, z, *rot)
                x_fk, y_fk, z_fk = fk_from_angles(q)
                desired_positions.append([x, y, z])
                recon_positions.append([x_fk, y_fk, z_fk])

        self.plot_results(np.array(desired_positions), np.array(recon_positions))

    def build_pose(self, x, y, angles):
        a = angles[0] + angles[1]
        pose = np.array([
            x, y, 0,
            np.cos(a), np.sin(a), 0,
            -np.sin(a), np.cos(a), 0,
            0, 0, 1
        ])
        return pose

    def plot_results(self, desired, recon):
        self.ax.plot(desired[:, 0], desired[:, 1], 'k--', label="Desired Path")
        self.ax.plot(recon[:, 0], recon[:, 1], 'b-', label="Reconstructed Path")
        self.ax.legend()
        self.canvas.draw()

        error = np.linalg.norm(desired - recon, axis=1)
        plt.figure()
        plt.plot(error)
        plt.title("End-Effector Position Error")
        plt.xlabel("Step")
        plt.ylabel("||XYZ_error|| [m]")
        plt.grid()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = TrajectoryApp(root)

    predefined_points = [
        [0.05, 0.1],
        [0.1, 0.2],
        [0.15, 0.2],
        [0.2, 0.1],
        [0.175, 0.15],
        [0.075, 0.15]
    ]

    app.load_points(predefined_points)

    root.mainloop()
