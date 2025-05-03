import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
from sympy import pi  
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Universal_IK_Solver.dataset_generation.dataset_generation_6DOF import generate_dataset
import os

class DatasetGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Robotic Arm Dataset Generator")

        self.entries = []
        self.num_dof_var = tk.IntVar(value=6)
        self.num_points_var = tk.IntVar(value=10)
        self.filename_var = tk.StringVar(value="6dof_dataset.csv")

        self.build_ui()

    def build_ui(self):
        # DOF selection
        tk.Label(self.root, text="Degrees of Freedom (2-6):").grid(row=0, column=0, sticky="w")
        tk.Spinbox(self.root, from_=2, to=6, textvariable=self.num_dof_var, width=5, command=self.build_dh_table).grid(row=0, column=1)

        # Number of points
        tk.Label(self.root, text="Number of Points:").grid(row=1, column=0, sticky="w")
        tk.Entry(self.root, textvariable=self.num_points_var, width=10).grid(row=1, column=1)

        # Filename
        tk.Label(self.root, text="CSV File Name:").grid(row=2, column=0, sticky="w")
        tk.Entry(self.root, textvariable=self.filename_var, width=20).grid(row=2, column=1)

        # File path
        self.path_label = tk.Label(self.root, text="Output Directory:")
        self.path_label.grid(row=3, column=0, sticky="w")
        self.path_entry = tk.Entry(self.root, width=30)
        self.path_entry.grid(row=3, column=1)
        tk.Button(self.root, text="Browse", command=self.browse_folder).grid(row=3, column=2)

        # DH Parameters table
        self.dh_frame = tk.Frame(self.root)
        self.dh_frame.grid(row=4, column=0, columnspan=3, pady=10)
        self.build_dh_table()

        # Generate button
        tk.Button(self.root, text="Generate Dataset", command=self.generate).grid(row=5, column=0, columnspan=3, pady=10)

    def build_dh_table(self):
        for widget in self.dh_frame.winfo_children():
            widget.destroy()
        self.entries.clear()

        tk.Label(self.dh_frame, text="Link").grid(row=0, column=0)
        labels = ['alpha (rad)', 'a (m)', 'd (m)', 'theta offset (rad)', 'Min Angle (rad)', 'Max Angle (rad)']
        for i, label in enumerate(labels):
            tk.Label(self.dh_frame, text=label).grid(row=0, column=i+1)

        for i in range(self.num_dof_var.get()):
            tk.Label(self.dh_frame, text=f"{i+1}").grid(row=i+1, column=0)
            row_entries = []
            for j in range(6):  
                e = tk.Entry(self.dh_frame, width=10)
                e.grid(row=i+1, column=j+1)
                row_entries.append(e)
            self.entries.append(row_entries)


    def browse_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, path)

    def generate(self):
        try:
            dof = self.num_dof_var.get()
            n_points = self.num_points_var.get()
            filename = self.filename_var.get()
            folder = self.path_entry.get()

            # Parse DH parameters
            dh_params = []
            limits = []  

            for i in range(dof):
                alpha = float(eval(self.entries[i][0].get()))
                a = float(eval(self.entries[i][1].get()))
                d = float(eval(self.entries[i][2].get()))
                offset = float(eval(self.entries[i][3].get()))
                min_angle = float(eval(self.entries[i][4].get()))
                max_angle = float(eval(self.entries[i][5].get()))
                dh_params.append({'alpha': alpha, 'a': a, 'd': d, 'theta_offset': offset})
                limits.append((min_angle, max_angle))  # add limits

            #pass limits when generating the dataset
            df = generate_dataset(dof, n_points, dh_params, folder, limits=limits)
            df.to_csv(os.path.join(folder, filename), index=False)
            messagebox.showinfo("Success", f"Dataset saved to:\n{os.path.join(folder, filename)}")

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetGeneratorGUI(root)
    root.mainloop()
