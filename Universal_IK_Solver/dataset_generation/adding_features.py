# This code is used for generating new features as radius vector, orientation and ect...

import pandas as pd
import numpy as np
from math import  atan2, sin, cos, sqrt
from scipy.spatial.transform import Rotation as R

df = pd.read_csv(r'Universal_IK_Solver/datasets_6DOF/6dof_dataset copy.csv',  encoding = 'utf8')
df = df.drop(['Unnamed: 0'], axis = 1)

fa_list = []
fo_list = []
fn_list = []
r_list = []


def get_rotation_representation(R_mat, mode='euler_zyx', degrees=True):
    
    rot = R.from_matrix(R_mat)

    if mode == 'euler_zyx':
        return rot.as_euler('zyx', degrees=degrees)
    elif mode == 'fixed_xyz':
        return rot.as_euler('xyz', degrees=degrees)
    elif mode == 'quaternion':
        return rot.as_quat()  # [x, y, z, w]
    else:
        raise ValueError(f"Unsupported mode '{mode}'")


for i in range(len(df)):
    
    x = df.iloc[i,0]; y = df.iloc[i,1]; z = df.iloc[i,2]
    r = sqrt(x*x + y*y +z*z)
    
    nx = df.iloc[i,3]; ny = df.iloc[i,4] ;nz = df.iloc[i,5]
    ox = df.iloc[i,6]; oy = df.iloc[i,7] ;oz = df.iloc[i,8]
    ax = df.iloc[i,9]; ay = df.iloc[i,10] ;az = df.iloc[i,11]
    
    fa = np.rad2deg( atan2(-ny, -nx) )
    fo = np.rad2deg ( atan2(-nz, ( nx * cos(fa) + ny * sin(fa) )) )
    fn = np.rad2deg( atan2( (-ay * cos(fa) + ax * sin(fa)), (oy * cos(fa) - ox * sin(fa)) ) )
    
    fa_list.append(round(fa,2)); fo_list.append(round(fo,2)), fn_list.append(round(fn,2))
    r_list.append(round(r,4))

#df = df.drop(['3','4','5','6','7','8','9','10','11'], axis = 1)
df_pom1 = df.iloc[:,:3]
df_pom2 = df.iloc[:,3:]
df_pom1.insert(3, '3', r_list); df_pom1.insert(4, '4', fn_list); df_pom1.insert(5, '5', fo_list); df_pom1.insert(6, '6',fa_list);   
df = pd.DataFrame(np.concatenate((df_pom1,df_pom2), axis =1 ))
