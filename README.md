# Universal IK Solver: ML-Based Inverse Kinematics for Robotic Arms

## Description
This project presents a machine learning-based approach for solving inverse kinematics of robotic arms with multiple degrees of freedom. It demonstrates solutions for both 2DOF and 6DOF robotic arms (but the approach can be extended to other variations as will be presented later), enabling accurate positioning without the computational complexity of traditional analytical methods
Solving inverse kinematics is essential for robotic arm operation, whether during initial setup or after configuration changes. The ML-based approach enables online calibration, making robotic arms more resilient to environmental changes like vibrations or mechanical wear without requiring operational downtime.

## Hardware Demo
https://www.youtube.com/watch?v=0vlXIcAu7CQ

## Features
- Dataset generation for robotic arms with configurable DOF (2-6)
- LSTM Neural Network model to handle different robotic manipulator structure
- Interactive GUI tools for dataset generation, model training, and trajectory planning
- Visualization tools for dataset exploration and model accuracy testing
- Support for different DH parameter configurations
- Real-time prediction of joint angles from desired end-effector positions
- Vision model to detect the objects (food items) specified by the user
- Final integrated hardware setup for demo and testing purposes

## Installation Instructions

### Prerequisites
- Python 3.6+
- conda virtual environmnet
- installing the dependencies and requirements fron the 'venv-dependencies' folder

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Universal_IK_Solver.git
   cd Universal_IK_Solver
   ```

2. Create and activate a virtual environment:
   ```
   conda env create --name envname --file=env_inverse_kinematics.yml
   # On Windows: conda activate envname
   ```

3. Install dependencies:
   ```
   pip install -r venv_dependencies/requirements.txt
   ```

## Usage

### Dataset Generation
Use the dataset generation GUI to create training data:

```
python Universal_IK_Solver/dataset_generation/dataset_gui.py
```

Or use the command-line script for specific configurations:

```
python Universal_IK_Solver/dataset_generation/dataset_generation_6DOF.py
```
![WhatsApp Image 2025-04-29 at 09 30 48](https://github.com/user-attachments/assets/a6e7ed70-c3d0-43f7-8e28-f3ff9463af2b)


### Training Models
Train a model using the training GUI:

```
python Universal_IK_Solver/training/Training_GUI.py
```

Or use specific model training scripts:

```
python Universal_IK_Solver/training/LSTM_Neural_Network.py
```

![WhatsApp Image 2025-04-29 at 11 09 27](https://github.com/user-attachments/assets/a4618266-e472-4f5a-a3da-7e14246d1a24)


### Testing and Visualization
Test a trained model with trajectory planning:

```
python Universal_IK_Solver/testing/Tragectory_GUI.py
```

Visualize dataset points:

```
python Universal_IK_Solver/dataset_generation/dataset_visualization.py
```

![WhatsApp Image 2025-04-29 at 16 41 02](https://github.com/user-attachments/assets/ae63e950-f236-4435-834a-fb89f9d50a66)


## Configuration
- DH parameters can be configured in the dataset generation scripts
- Model hyperparameters can be adjusted in the training scripts or through the GUI
- Joint angle limits can be specified during dataset generation

## Project Structure
- `dataset_generation/`: Scripts for generating and processing datasets and saving the generated datasets (Generation GUI)
- `training/`: Model training implementations (LSTM, and training GUI)
- `testing/`: Tools for evaluating models and visualizing results (Trajectory GUI)
- `saved_models/`: Pre-trained models and scalers
- `venv_dependencies/`: Virtual environment configuration
- `GroundingDINO_with_Segment_Anything`: Python notebook to process the image captured from the calibrated camera and return the object position

## Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request


