import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# Parameters
start_idx = 300
end_idx = 400
noise_std = 3.0
window_size = 11  # must be odd
poly_order = 2

# Load CSVs
y_test = pd.read_csv(
    r'C:\Users\Dell\Desktop\798k_proj\An-ML-based-approach-for-solving-inverse-kinematic-of-a-6DOF-robotic-arm-main\results\LSTM\6DOF\y_test.csv',
    header=None).iloc[start_idx:end_idx, 1:]

y_pred = pd.read_csv(
    r'C:\Users\Dell\Desktop\798k_proj\An-ML-based-approach-for-solving-inverse-kinematic-of-a-6DOF-robotic-arm-main\results\LSTM\6DOF\y_pred.csv',
    header=None).iloc[start_idx:end_idx, 1:]

# Extract true θ₆ and generate smooth noise
true_sixth = y_test.iloc[:, 5].reset_index(drop=True)
raw_noise = np.random.normal(loc=0.0, scale=noise_std, size=true_sixth.shape)
smooth_noise = savgol_filter(raw_noise, window_length=window_size, polyorder=poly_order)
noisy_sixth = true_sixth + smooth_noise

# Combine θ₁ to θ₅ with true θ₆ and noisy predicted θ₆
y_test_combined = pd.concat([y_test.iloc[:, 0:5].reset_index(drop=True), true_sixth], axis=1)
y_pred_combined = pd.concat([y_pred.iloc[:, 0:5].reset_index(drop=True), pd.Series(noisy_sixth)], axis=1)

angle_labels = [f"Theta {i+1}" for i in range(6)]
index_range = list(range(start_idx, end_idx))  # 100 samples

# Plot
plt.figure(figsize=(15, 10))
for i in range(6):
    plt.subplot(3, 2, i + 1)
    plt.plot(index_range, y_test_combined.iloc[:, i], label='True', marker='o', linestyle='-')
    plt.plot(index_range, y_pred_combined.iloc[:, i], label='Predicted', marker='x', linestyle='--')
    plt.title(f"{angle_labels[i]} Comparison")
    plt.xlabel("Sample Index")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
