# This code is used for creating data set for 2-6DOF robotic arm with use of direct kinematic.

from sympy import symbols, pi, sin, cos, simplify
from sympy.matrices import Matrix
import numpy as np
import random
import pandas as pd
import os
import math
import time

# building the mdified DH parameters to build the DH table

def build_mod_dh_matrix(theta, alpha, d, a):
    return Matrix([
        [cos(theta), -cos(alpha)*sin(theta),  sin(alpha)*sin(theta), a*cos(theta)],
        [sin(theta),  cos(alpha)*cos(theta), -sin(alpha)*cos(theta), a*sin(theta)],
        [0,           sin(alpha),             cos(alpha),            d],
        [0,           0,                      0,                     1]
    ])

# using the derived DH table, build the final homogeneous transformation matrix (HTM) 
# this matrix contains all the needed features (position and orientation elements)

def forward_kinematics(joint_angles, dh_params):
    T = Matrix(np.eye(4))
    theta_syms = symbols(f'theta0:{len(joint_angles)}')

    for i in range(len(joint_angles)):
        theta_val = joint_angles[i]
        dh = dh_params[i]
        theta_offset = dh.get('theta_offset', 0)
        T_i = build_mod_dh_matrix(theta_syms[i], dh['alpha'], dh['d'], dh['a'])
        T = T * T_i.subs({theta_syms[i]: theta_val + theta_offset})
    
    return T.evalf()

def generate_random_angles(n_points, dof, limits=None):
    if limits is None:
        limits = [(-np.pi, np.pi)] * dof 
    
    return np.array([
        [random.uniform(*limits[j]) for j in range(dof)]
        for _ in range(n_points)
    ])

def generate_sequential_angles(n_points, dof, limits=None, step=None):
    if limits is None:
        limits = [(-np.pi, np.pi)] * dof
    
    if step is None:
        step = [(limits[j][1] - limits[j][0]) / (n_points - 1) for j in range(dof)]
    
    # Build each joint's angle list
    angle_lists = []
    for j in range(dof):
        start, end = limits[j]
        angle_list = np.arange(start, end + step[j], step[j])
        angle_lists.append(angle_list)

    # create a meshgrid for all joints 
    # (this function can be used only to collect few dataponts at a time in specific limited angle range)
    grids = np.meshgrid(*angle_lists, indexing='ij')
    all_combinations = np.stack(grids, axis=-1).reshape(-1, dof)

    return all_combinations

def generate_dataset(dof, n_points, dh_params, save_dir, limits=None):
    angles = generate_random_angles(n_points, dof, limits=limits)
    data = []

    for i, joint_angles in enumerate(angles):
        T = forward_kinematics(joint_angles, dh_params)
        pos = [T[0, 3], T[1, 3], T[2, 3]]
        n = [T[0, 0], T[1, 0], T[2, 0]]
        o = [T[0, 1], T[1, 1], T[2, 1]]
        a = [T[0, 2], T[1, 2], T[2, 2]]
        data.append(pos + n + o + a + list(map(math.degrees, joint_angles)))

    df = pd.DataFrame(data)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(script_dir)
    target_dir = os.path.join(script_dir, 'datasets_2DOF')
    os.makedirs(target_dir, exist_ok=True)
    csv_path = os.path.join(target_dir, '2dof_dataset1.csv')
    df.to_csv(csv_path)
    return df


#################TESTING WITH DIFFERENT DH PARAMETERS#####################

# Define DH parameters for 6DOF 
# dh_params_6dof = [
#     {'alpha': pi/2,  'd': 0.1765, 'a': 0,      'theta_offset': 0},
#     {'alpha': 0,     'd': 0,      'a': 0.607,  'theta_offset': pi/2},   # theta2 + pi/2
#     {'alpha': 0,     'd': 0,      'a': 0.568,  'theta_offset': 0},
#     {'alpha': -pi/2, 'd': 0.191,  'a': 0,      'theta_offset': -pi/2},  # theta4 - pi/2
#     {'alpha': pi/2,  'd': 0.125,  'a': 0,      'theta_offset': 0},
#     {'alpha': 0,     'd': 0.1114, 'a': 0,      'theta_offset': 0},
# ]

# dof = 6
# n_points = 8000
# save_dir = './datasets_6DOF'

# df = generate_dataset(dof, n_points, dh_params_6dof[:dof], save_dir)
# print(df.head())

# dh_params_6dof = [
#     {'alpha': 0,     'd': 0, 'a': 0.165, 'theta_offset': 0},
#     {'alpha': 0,     'd': 0, 'a': 0.12,  'theta_offset': 0},   # theta2 + pi/2
# ]

# dof = 2
# n_points = 2
# save_dir = './datasets_2DOF'

# df = generate_dataset(dof, n_points, dh_params_6dof[:dof], save_dir)
# print(df.head())