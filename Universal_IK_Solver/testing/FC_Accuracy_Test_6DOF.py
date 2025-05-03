####################################################################################################
####################################################################################################
# This code is used for 3D visualization of training and test points for robotic arm with 6 DOF.

from sympy import symbols, pi, sin, cos, simplify
from sympy.matrices import Matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import open3d as o3d
import math
from mpl_toolkits.mplot3d import Axes3D
import time

def build_mod_dh_matrix(s, theta, alpha, d, a):

    # transformation matrix  
    
    Ta_b = Matrix([ [cos(theta), -cos(alpha)*sin(theta),  sin(alpha)*sin(theta), a*cos(theta)],
                    [sin(theta),  cos(alpha)*cos(theta), -sin(alpha)*cos(theta), a*sin(theta)],
                    [0,           sin(alpha),             cos(alpha),            d           ],
                    [0,           0,                      0,                     1]          ])
    
    # Substitute in the DH parameters 
    
    Ta_b = Ta_b.subs(s)
    
    return Ta_b

def calculate_position(teta1, teta2, teta3, teta4, teta5, teta6):
    
    # DH param symbol
        
    theta1, theta2, theta3, theta4, theta5, theta6 = symbols('theta1:7')
    alpha0, alpha1, alpha2, alpha3, alpha4, alpha5 = symbols('alpha0:6')
    d1, d2, d3, d4, d5, d6 = symbols('d1:7')    # link offsets
    a0, a1, a2, a3, a4, a5 = symbols('a0:6')    # link lengths
    
    # DH params for CR10
    
    cr10 = {  alpha0:   pi/2,   d1:  0.1765,    a0:   0,        
              alpha1:   0,      d2:     0,      a1:   0.607,    theta2: (theta2 + pi/2),
              alpha2:   0,      d3:     0,      a2:   0.568,      
              alpha3:  -pi/2,   d4:  0.191,     a3:   0,        theta4: (theta4 - pi/2),
              alpha4:   pi/2,   d5:  0.125,     a4:   0,
              alpha5:   0,      d6:  0.1114,    a5:   0, } 
              
    
    # Define Modified DH Transformation matrix
              
    T0_1 = build_mod_dh_matrix(s=cr10, theta=theta1, alpha=alpha0, d=d1, a=a0)
    T1_2 = build_mod_dh_matrix(s=cr10, theta=theta2, alpha=alpha1, d=d2, a=a1)
    T2_3 = build_mod_dh_matrix(s=cr10, theta=theta3, alpha=alpha2, d=d3, a=a2)
    T3_4 = build_mod_dh_matrix(s=cr10, theta=theta4, alpha=alpha3, d=d4, a=a3)
    T4_5 = build_mod_dh_matrix(s=cr10, theta=theta5, alpha=alpha4, d=d5, a=a4)
    T5_6 = build_mod_dh_matrix(s=cr10, theta=theta6, alpha=alpha5, d=d6, a=a5)
    
    # Create individual transformation matrices
    
    T0_2 = simplify(T0_1 * T1_2)    # base link to link 2
    T0_3 = simplify(T0_2 * T2_3)    # base link to link 3
    T0_4 = simplify(T0_3 * T3_4)    # base link to link 4
    T0_5 = simplify(T0_4 * T4_5)    # base link to link 5
    T0_G = simplify(T0_5 * T5_6)    # base link to link 6

    T_total = simplify( T0_G )
    
    # Numerically evaluate transforms 
    
    print(T0_1)
    print(T1_2)
    print(T2_3)
    print(T3_4)
    print(T4_5)
    print(T5_6)
        
    result = T_total.evalf(subs={theta1: teta1, theta2: teta2, theta3: teta3, theta4: teta4, theta5: teta5, theta6: teta5})
    
    final = np.array(result).astype(np.float64)
    
    return final
st = time.time()

path = path = r"C:/Users/Dell/Desktop/798k_proj/Universal_IK_Solver/results/sequential neural network/6DOF/"

sample_num = 2 #200

y_pred = pd.read_csv(r'C:\Users\Dell\Desktop\798k_proj\saved_models\y_pred_c.csv',  encoding = 'utf8')
y_pred = y_pred.drop(['Unnamed: 0'], axis = 1)
y_pred = y_pred.iloc[0:sample_num,:]

y_test = pd.read_csv(r'C:\Users\Dell\Desktop\798k_proj\saved_models\X_test_c.csv',  encoding = 'utf8')
y_test = y_test.drop(['Unnamed: 0'], axis = 1)
y_test = y_test.iloc[0:sample_num,:]

X_test = pd.read_csv(r'C:\Users\Dell\Desktop\798k_proj\saved_models\X_test_c.csv',  encoding = 'utf8')
X_test = X_test.drop(['Unnamed: 0'], axis = 1)
X_test = X_test.iloc[0:sample_num,:]

y_pred = y_pred.values

n = np.zeros([1,3],dtype=int)
o = np.zeros([1,3],dtype=int)
a = np.zeros([1,3],dtype=int)
positions = np.zeros([1,3],dtype=int)

for i in range(len(y_pred)):
    
    print(i)
    final = calculate_position(math.radians(y_pred[i][0]), math.radians(y_pred[i][1]), math.radians(y_pred[i][2]), math.radians(y_pred[i][3]), math.radians(y_pred[i][4]), math.radians(y_pred[i][5]))
    
    position_xyz = []
    position_xyz.append( [final[0][3], final[1][3], final[2][3]] )

    n_xyz = []
    n_xyz.append( [final[0][0], final[1][0], final[2][0]] )
    
    o_xyz = []
    o_xyz.append( [final[0][1], final[1][1], final[2][1]] )
    
    a_xyz = []
    a_xyz.append( [final[0][2], final[1][2], final[2][2]] )
    
    positions = np.concatenate((positions,position_xyz) )
    n = np.concatenate((n, n_xyz))
    o = np.concatenate((o, o_xyz))
    a = np.concatenate((a, a_xyz))

X_pred = pd.DataFrame(np.concatenate((positions,n,o,a), axis = 1) )
X_pred = X_pred.iloc[1:,:]
X_pred.to_csv(path + '\X_pred_c.csv')

## only for plotting
X_pred = pd.read_csv(path + '\X_pred_c.csv',  encoding = 'utf8')
X_pred = X_pred.drop(['Unnamed: 0'], axis = 1)
X_pred = X_pred.iloc[0:sample_num,:]

X_pred = X_pred.iloc[:,:3]
X_test = X_test.iloc[:,:3]

X_pred_r = []
X_test_r = []

for i in range(len(X_pred)):
    
    X_pred_r.append( math.sqrt(X_pred.iloc[i][0]*X_pred.iloc[i][0] + X_pred.iloc[i][1]*X_pred.iloc[i][1] + X_pred.iloc[i][2]*X_pred.iloc[i][2]) )
    X_test_r.append( math.sqrt(X_test.iloc[i][0]*X_test.iloc[i][0] + X_test.iloc[i][1]*X_test.iloc[i][1] + X_test.iloc[i][2]*X_test.iloc[i][2]) )

X_pred['r'] = X_pred_r
X_test['r'] = X_test_r

result_mse = []
result_mae = []

for i in range(4):

   mse = mean_squared_error(X_test.iloc[:,i], X_pred.iloc[:,i])
   rmse = sqrt(mse) 
   mae = mean_absolute_error(X_test.iloc[:,i], X_pred.iloc[:,i])
   result_mse.append(mse)
   result_mae.append(mae)

print("RMSE", result_mse)
print("MAE", result_mae)

en = time.time()
print("time needed", en - st)

y_pred = pd.DataFrame(y_pred)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_pred.iloc[1:100,0], X_pred.iloc[1:100,1], X_pred.iloc[1:100,2], color='r')
ax.scatter3D(X_test.iloc[1:100,0], X_test.iloc[1:100,1], X_test.iloc[1:100,2], color='g')
plt.show()