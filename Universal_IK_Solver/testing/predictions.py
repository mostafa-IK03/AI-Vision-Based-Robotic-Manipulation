import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas as pd


model = load_model(r"C:\Users\Dell\Desktop\798k_proj\ik_lstm.h5")
x_scaler = joblib.load(r"C:\Users\Dell\Desktop\798k_proj\x_scaler.pkl")
y_scaler = joblib.load(r"C:\Users\Dell\Desktop\798k_proj\y_scaler.pkl")




df_raw = pd.read_csv(
    r'C:\Users\Dell\Desktop\798k_proj\Universal_IK_Solver\datasets_6DOF\6dof_dataset.csv',
    index_col=0               
)

# print(df_raw.head())


# construct input vector with fixed orientation
def build_input(x, y, z):
    R = np.eye(3)  # fixed identity rotation
    pose_matrix = np.array([
        R[0,0], R[0,1], R[0,2], x,
        R[1,0], R[1,1], R[1,2], y,
        R[2,0], R[2,1], R[2,2], z
    ])
    return pose_matrix.reshape(1, -1)

def predict_ik(x, y, z):
    input_vec = build_input(x, y, z)
    input_scaled = x_scaler.transform(input_vec)
    input_reshaped = input_scaled.reshape((1, 1, 12))
    pred_scaled = model.predict(input_reshaped)
    return y_scaler.inverse_transform(pred_scaled)[0]

def predict_ik_from_pose(x, y, z, r11, r21, r31, r12, r22, r32, r13, r23, r33):
    input_vec = np.array([x, y, z, r11, r21, r31, r12, r22, r32, r13, r23, r33]).reshape(1, -1)

    input_scaled = x_scaler.transform(input_vec)
    input_reshaped = input_scaled.reshape((1, 1, 12))
    pred_scaled = model.predict(input_reshaped)
    return y_scaler.inverse_transform(pred_scaled)[0]


def debug_one(row):
    raw = row.values.reshape(1,-1)
    scaled = x_scaler.transform(raw)
    batch_pred = model.predict(scaled.reshape(1,1,12))
    # print(scaled.reshape(1,1,12))
    batch_out = y_scaler.inverse_transform(batch_pred)[0]

    # now call your function with explicit unpacking
    x, y, z, r11, r21, r31, r12, r22, r32, r13, r23, r33 = raw[0]
    custom_out = predict_ik_from_pose(x, y, z, r11, r21, r31, r12, r22, r32, r13, r23, r33)

    print("batch_out: ", batch_out)
    print("custom_out:", custom_out)

# pick one example from your original DataFrame (before scaling)
X_raw = df_raw.iloc[:, :12]
debug_one( X_raw.iloc[0] )



# angles = predict_ik(1.22782801647777,-0.373865912461409,-0.0483448937113055) 
# print("Joint angles:", angles)

# angles = predict_ik_from_pose(
#     1.22782801647777,   -0.373865912461409, -0.0483448937113055,   0.330700312063051,
#    -0.824626287522075, -0.829198003152342,  0.449302351381115,   -4.315269700559516,
#    -0.921455658653309, -0.181111553616180,  0.343683101537638,  -0.208354783454295
# )

# print("Predicted joint angles:", angles)