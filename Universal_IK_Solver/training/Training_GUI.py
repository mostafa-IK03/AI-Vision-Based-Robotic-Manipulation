import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

# Custom weighted loss function
def custom_loss(pos_weight=0.8, ori_weight=0.2):
    def loss_fn(y_true, y_pred):
        pos_loss = K.mean(K.abs(y_true[:, :3] - y_pred[:, :3]))
        ori_loss = K.mean(K.abs(y_true[:, 3:] - y_pred[:, 3:]))
        return pos_weight * pos_loss + ori_weight * ori_loss
    return loss_fn

class TrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Trainer for IK")

        self.dataset_path = tk.StringVar()
        self.epochs = tk.IntVar(value=100)
        self.batch_size = tk.IntVar(value=32)
        self.learning_rate = tk.DoubleVar(value=0.001)
        self.hidden_units = tk.IntVar(value=64)
        self.input_features = tk.StringVar(value="HTM 12")
        self.loss_function = tk.StringVar(value="MAE")
        self.optimizer_choice = tk.StringVar(value="Adam")
        self.split_outputs = tk.BooleanVar(value=False)
        self.use_cv = tk.BooleanVar(value=False)
        self.k_folds = tk.IntVar(value=5)
        self.num_models = tk.IntVar(value=1)
        self.split_entries = []

        self.build_ui()

    def build_ui(self):
        tk.Label(self.root, text="Dataset:").grid(row=0, column=0)
        tk.Entry(self.root, textvariable=self.dataset_path, width=50).grid(row=0, column=1)
        tk.Button(self.root, text="Browse", command=self.browse_dataset).grid(row=0, column=2)

        tk.Label(self.root, text="Epochs:").grid(row=1, column=0)
        tk.Entry(self.root, textvariable=self.epochs).grid(row=1, column=1)

        tk.Label(self.root, text="Batch Size:").grid(row=2, column=0)
        tk.Entry(self.root, textvariable=self.batch_size).grid(row=2, column=1)

        tk.Label(self.root, text="Learning Rate:").grid(row=3, column=0)
        tk.Entry(self.root, textvariable=self.learning_rate).grid(row=3, column=1)

        tk.Label(self.root, text="Hidden Units (LSTM):").grid(row=4, column=0)
        tk.Entry(self.root, textvariable=self.hidden_units).grid(row=4, column=1)

        tk.Label(self.root, text="Input Features:").grid(row=5, column=0)
        tk.OptionMenu(self.root, self.input_features, "HTM 12", "Euler Angles").grid(row=5, column=1)

        tk.Label(self.root, text="Loss Function:").grid(row=6, column=0)
        tk.OptionMenu(self.root, self.loss_function, "MAE", "MSE", "Custom Weighted").grid(row=6, column=1)

        tk.Label(self.root, text="Optimizer:").grid(row=7, column=0)
        tk.OptionMenu(self.root, self.optimizer_choice, "Adam", "RMSprop", "SGD").grid(row=7, column=1)

        tk.Checkbutton(self.root, text="Split Outputs into Multiple Models", variable=self.split_outputs, command=self.update_split_fields).grid(row=8, column=0, columnspan=2)
        tk.Checkbutton(self.root, text="Use Cross Validation", variable=self.use_cv, command=self.toggle_kfold_field).grid(row=9, column=0, columnspan=2)

        tk.Label(self.root, text="K-Folds:").grid(row=10, column=0)
        self.k_entry = tk.Entry(self.root, textvariable=self.k_folds)
        self.k_entry.grid(row=10, column=1)
        self.k_entry.config(state='disabled')

        tk.Button(self.root, text="Train Model", command=self.train_model).grid(row=20, column=0, columnspan=3, pady=10)

    def toggle_kfold_field(self):
        if self.use_cv.get():
            self.k_entry.config(state='normal')
        else:
            self.k_entry.config(state='disabled')

    def update_split_fields(self):
        for widget in self.split_entries:
            widget.destroy()
        self.split_entries.clear()

        if self.split_outputs.get():
            tk.Label(self.root, text="Number of Models:").grid(row=11, column=0)
            entry_models = tk.Entry(self.root, textvariable=self.num_models)
            entry_models.grid(row=11, column=1)
            self.split_entries.append(entry_models)

            tk.Button(self.root, text="Confirm Splits", command=self.create_model_split_fields).grid(row=12, column=0, columnspan=2)

    def create_model_split_fields(self):
        for widget in self.split_entries[1:]:
            widget.destroy()
        self.split_entries = self.split_entries[:1]

        self.model_split_vars = []

        for i in range(self.num_models.get()):
            tk.Label(self.root, text=f"Model {i+1} Joints:").grid(row=13+i, column=0)
            var = tk.StringVar()
            entry = tk.Entry(self.root, textvariable=var)
            entry.grid(row=13+i, column=1)
            self.split_entries.append(entry)
            self.model_split_vars.append(var)

    def browse_dataset(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.dataset_path.set(path)

    def get_optimizer(self):
        lr = self.learning_rate.get()
        if self.optimizer_choice.get() == "Adam":
            return Adam(lr=lr)
        elif self.optimizer_choice.get() == "RMSprop":
            return RMSprop(lr=lr)
        elif self.optimizer_choice.get() == "SGD":
            return SGD(lr=lr)

    def get_loss_function(self):
        if self.loss_function.get() == "MAE":
            return "mae"
        elif self.loss_function.get() == "MSE":
            return "mse"
        elif self.loss_function.get() == "Custom Weighted":
            return custom_loss()

    def extract_euler_angles(self, X_htm):
        eulers = []
        for i in range(X_htm.shape[0]):
            r11, r12, r13, px, r21, r22, r23, py, r31, r32, r33, pz = X_htm.iloc[i]
            roll = np.arctan2(r32, r33)
            pitch = np.arctan2(-r31, np.sqrt(r32**2 + r33**2))
            yaw = np.arctan2(r21, r11)
            eulers.append([px, py, pz, roll, pitch, yaw])
        return pd.DataFrame(eulers)

    def train_model(self):
        try:
            df = pd.read_csv(self.dataset_path.get())
            df = df.drop(['Unnamed: 0'], axis=1, errors='ignore')

            if self.input_features.get() == "HTM 12":
                X = df.iloc[:, :12]
            else:
                X = self.extract_euler_angles(df.iloc[:, :12])

            y = df.iloc[:, 12:]

            x_scaler = MinMaxScaler((-1, 1))
            X_s = x_scaler.fit_transform(X)

            if self.split_outputs.get():
                joint_groups = {}
                for idx, var in enumerate(self.model_split_vars):
                    try:
                        indices = [int(x.strip()) for x in var.get().split(',')]
                        joint_groups[f'model_{idx+1}'] = indices
                    except:
                        messagebox.showerror("Input Error", f"Invalid joint indices in Model {idx+1}")
                        return
            else:
                joint_groups = {'all_joints': list(range(y.shape[1]))}

            if self.use_cv.get():
                kf = KFold(n_splits=self.k_folds.get(), shuffle=True, random_state=42)
                for model_name, joints in joint_groups.items():
                    y_group = y.iloc[:, joints]
                    y_scaler = MinMaxScaler((-1, 1))
                    y_s_group = y_scaler.fit_transform(y_group)

                    fold_idx = 0
                    for train_idx, test_idx in kf.split(X_s):
                        fold_idx += 1
                        X_train, X_test = X_s[train_idx], X_s[test_idx]
                        y_train, y_test = y_s_group[train_idx], y_s_group[test_idx]
                        self.build_and_train_model(X_train, X_test, y_train, y_test, model_name + f"_fold{fold_idx}", x_scaler, y_scaler)
            else:
                for model_name, joints in joint_groups.items():
                    y_group = y.iloc[:, joints]
                    y_scaler = MinMaxScaler((-1, 1))
                    y_s_group = y_scaler.fit_transform(y_group)
                    X_train, X_test, y_train, y_test = train_test_split(X_s, y_s_group, test_size=0.2)
                    self.build_and_train_model(X_train, X_test, y_train, y_test, model_name, x_scaler, y_scaler)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def build_and_train_model(self, X_train, X_test, y_train, y_test, model_name, x_scaler, y_scaler):
        X_train_r = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test_r = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        model = Sequential()
        model.add(LSTM(self.hidden_units.get(), input_shape=(X_train_r.shape[1:]), activation='relu', return_sequences=True))
        model.add(LSTM(self.hidden_units.get()*2, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(y_train.shape[1], activation='linear'))

        model.compile(
            loss=self.get_loss_function(),
            optimizer=self.get_optimizer(),
            metrics=['mae']
        )

        model.fit(
            X_train_r, y_train,
            batch_size=self.batch_size.get(),
            epochs=self.epochs.get(),
            validation_split=0.2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
            verbose=1
        )

        model.save(f"{model_name}.h5")
        joblib.dump(x_scaler, f"x_scaler_{model_name}.pkl")
        joblib.dump(y_scaler, f"y_scaler_{model_name}.pkl")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainerGUI(root)
    root.mainloop()