# This code is used to vizualize the robot workspace (the dataset).

from matplotlib import pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def plot_xyz_vs_index(df, start_idx, end_idx):
    df_range = df.iloc[start_idx:end_idx, :]
    xyz = df_range.iloc[:, 1:4]

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    labels = ['X', 'Y', 'Z']
    colors = ['red', 'green', 'blue']

    for i in range(3):
        axs[i].plot(range(start_idx, end_idx), xyz.iloc[:, i], label=f'{labels[i]}', color=colors[i])
        axs[i].set_ylabel(f'{labels[i]} (m)')
        axs[i].legend()
        axs[i].grid(True)

    axs[2].set_xlabel('Index')
    plt.tight_layout()
    plt.show()



df = pd.read_csv(r'C:\Users\Dell\Desktop\798k_proj\Universal_IK_Solver\dataset_generation\datasets_2DOF\2dof_dataset.csv',  encoding = 'utf8')
# Setting number of points for vizualization
number_points = 1000

df = df.iloc[0:number_points,:]
df = df.iloc[:,1:]

xyz = df.iloc[:,0:3]

# Plotting the points
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xyz.iloc[:,0]*10, xyz.iloc[:,1], xyz.iloc[:,2], c=xyz.iloc[:,2])
plot_xyz_vs_index(df, 0, 100)
plt.show()
