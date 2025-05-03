import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

from google.colab import files
uploaded = files.upload()
data_path = list(uploaded.keys())[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ! pip install ikpy

# ! pip install visual_kinematics

import ikpy.chain
import numpy as np
import ikpy.utils.plot as plot_utils

import numpy as np
from math import pi
from visual_kinematics.RobotSerial import RobotSerial

from sympy import *
import pandas as pd

d1 = 0.1765
d4 = 0.191
d5 = 0.125
d6 = 0.1114
a2 = 0.607
a3 = 0.568

dh_params = np.array([
    [d1, 0, pi/2, 0],        # Joint 1
    [0, a2, 0, pi/2],        # Joint 2
    [0, a3, 0, 0],           # Joint 3
    [d4, 0, -pi/2, -pi/2],   # Joint 4
    [d5, 0, pi/2, 0],        # Joint 5
    [d6, 0, 0, 0 ]           # Joint 6
])

robot = RobotSerial(dh_params)

# Degrees list
degrees = [ 6.4710855, -58.90768,   -26.25425,    28.167387,   74.87567,    39.38521]

# Convert to radians
radians = np.radians(degrees)
theta = np.array(radians)
f = robot.forward(theta)
print(f.t_3_1.reshape([3, ]))  # XYZ coordinates of EE

def forward_kinematics(q):
    """
    q: NumPy array, shape (batch_size, 6)
    returns:
       pos: (batch_size, 3)
       rot: (batch_size, 3)
    """

    if q.ndim == 1:
        f = robot.forward(q)
        return f.t_3_1.copy(), f.euler_3.copy()

    # otherwise loop over the batch
    positions = []
    rotations = []
    for theta in q:               # theta is shape (6,)
        f = robot.forward(theta)
        positions.append(f.t_3_1.copy())   # each is (3,)
        rotations.append(f.euler_3.copy())
    # stack into arrays of shape (batch, 3) and (batch, ...)
    pos_arr = np.stack(positions, axis=0)
    rot_arr = np.stack(rotations, axis=0)
    return pos_arr, rot_arr

import torch
import torch.nn as nn

class InverseKinematicsNN(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 6)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        # Scale tanh output to [-π, π]
        theta = torch.pi * torch.tanh(self.fc3(x))
        return theta

model = InverseKinematicsNN()

import torch
import torch.nn.functional as F

# def physics_loss(model, x, y_true):
#     """
#     x_phys:  tensor containing the true EE x,y,z (orientation will be used later)
#     y_true:  true joint angles

#     """
#     # Data loss joints
#     theta_pred = model(x)
#     data_loss  = F.mse_loss(theta_pred, y_true)

#     theta_np = theta_pred.detach().cpu().numpy()

#     # forward‐kinematics
#     pos_pred, rot_pred = forward_kinematics(theta_np)
#     pos_true = x[:, :3]

#     # Physics loss position
#     pos_loss = F.mse_loss(pos_pred, pos_true)

#     #  Orientation loss:
#     #    rot_true = x_phys[:, 3:].view(-1,3,3)
#     #    # e.g. geodesic distance on SO(3):
#     #    R_err = rot_pred.bmm(rot_true.transpose(1,2))
#     #    ang_error = torch.acos(((R_err.diagonal(dim1=1,dim2=2).sum(-1) - 1)/2).clamp(-1,1))
#     #    ori_loss = ang_error.mean()
#     # else: ori_loss = 0

#     total_phys = pos_loss   # + ori_loss (if used)
#     total_loss = total_phys + 0.01*data_loss

#     return total_loss

def physics_loss(model, x, y_true):
    # 1) Data loss on joints
    theta_pred = model(x)
    data_loss  = F.mse_loss(theta_pred, y_true)

    # 2) Physics loss via  FK
    theta_np = theta_pred.detach().cpu().numpy()
    pos_np, _ = forward_kinematics(theta_np)

    pos_pred = torch.from_numpy(pos_np) \
                    .to(device=theta_pred.device) \
                    .type(theta_pred.dtype)
    pos_pred = pos_pred.squeeze(-1)


    pos_true = x[:, :3]
    pos_loss = F.mse_loss(pos_pred, pos_true)

    # 3) combine
    return pos_loss + 0.01 * data_loss



optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 50
for epoch in range(1, num_epochs+1):
    # — training —
    model.train()
    train_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        loss = physics_loss(model, x_batch, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x_batch.size(0)

    train_loss /= len(train_loader.dataset)

    # — validation —
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            loss = physics_loss(model, x_val, y_val)
            val_loss += loss.item() * x_val.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch:3d}  Train: {train_loss:.6f}  Val: {val_loss:.6f}")

# ─── TEST EVALUATION ────────────────────────────────────────────────
model.eval()
test_loss = 0.0
with torch.no_grad():
    for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        loss = physics_loss(model, x_test, y_test)
        test_loss += loss.item() * x_test.size(0)
test_loss /= len(test_loader.dataset)
print(f"\nFinal Test Loss: {test_loss:.6f}")


def predict_angles(x, y, z, model):
    # Normalize input
    x_norm = x
    y_norm = y
    z_norm = z
    input_tensor = torch.FloatTensor([[x_norm, y_norm, z_norm]])

    # Predict
    theta_pred_normalized = model(input_tensor)
   # theta_pred = theta_pred_normalized * torch.pi  # Rescale to radians
    theta1, theta2, theta3, theta4, theta5, theta6 = theta_pred_normalized.detach().numpy()[0]
    thetas = np.array([theta1, theta2, theta3, theta4, theta5, theta6])
    pos_arr, rot_arr = forward_kinematics(thetas)
    x_recon, y_recon, z_recon = pos_arr[0], pos_arr[1], pos_arr[2]

    # turn them into Python floats
    x_recon = float(x_recon)
    y_recon = float(y_recon)
    z_recon = float(z_recon)



    print(f"Input (x, y, z): ({x}, {y}, {z})")
    print(f"Predicted angles (rad): θ1={theta1:.4f}, θ2={theta2:.4f}, θ3={theta3:.4f}, θ4={theta4:.4f},θ5={theta5:.4f}, θ6={theta6:.4f}")
    print(f"Reconstructed (x, y, z): {x_recon:.4f}, {y_recon:.4f}, {z_recon:.4f}")
    return theta1, theta2, theta3, theta4, theta5, theta6


theta1, theta2, theta3, theta4, theta5, theta6 = predict_angles( -1.0913, -0.5069, 1.2875, model)