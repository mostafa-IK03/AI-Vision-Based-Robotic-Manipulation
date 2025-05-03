# Universal IK Solver: ML-Based Inverse Kinematics for Robotic Arms

## Description
This project presents a machine learning-based approach for solving inverse kinematics of robotic arms with multiple degrees of freedom. It demonstrates solutions for both 3DOF and 6DOF robotic arms, enabling accurate positioning without the computational complexity of traditional analytical methods. This work was originally developed as part of a bachelor thesis.

Solving inverse kinematics is essential for robotic arm operation, whether during initial setup or after configuration changes. The ML-based approach enables online calibration, making robotic arms more resilient to environmental changes like vibrations or mechanical wear without requiring operational downtime.

## Features
- Dataset generation for robotic arms with configurable DOF (2-6)
- Multiple ML model implementations (LSTM, Sequential Neural Networks, Gradient Boosting)
- Interactive GUI tools for dataset generation, model training, and trajectory planning
- Visualization tools for dataset exploration and model accuracy testing
- Support for different DH parameter configurations
- Real-time prediction of joint angles from desired end-effector positions

## Installation Instructions

### Prerequisites
- Python 3.6+
- TensorFlow 2.x
- NumPy, Pandas, Matplotlib
- Sympy
- Scikit-learn
- Visual Kinematics

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Universal_IK_Solver.git
   cd Universal_IK_Solver
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
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

### Training Models
Train a model using the training GUI:

```
python Universal_IK_Solver/training/Training_GUI.py
```

Or use specific model training scripts:

```
python Universal_IK_Solver/training/LSTM_Neural_Network.py
```

### Testing and Visualization
Test a trained model with trajectory planning:

```
python Universal_IK_Solver/testing/Tragectory_GUI.py
```

Visualize dataset points:

```
python Universal_IK_Solver/dataset_generation/dataset_visualization.py
```

## Configuration
- DH parameters can be configured in the dataset generation scripts
- Model hyperparameters can be adjusted in the training scripts or through the GUI
- Joint angle limits can be specified during dataset generation

## Project Structure
- `dataset_generation/`: Scripts for generating and processing datasets
- `training/`: Model training implementations (LSTM, Sequential NN, etc.)
- `testing/`: Tools for evaluating models and visualizing results
- `saved_models/`: Pre-trained models and scalers
- `venv_dependencies/`: Virtual environment configuration

## Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

### Coding Standards
- Follow PEP 8 style guidelines
- Include docstrings for all functions and classes
- Write unit tests for new functionality

## Testing
Run tests using:
```
python -m unittest discover tests
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Credits
- Visual Kinematics library for forward kinematics calculations
- TensorFlow and Keras for neural network implementations
- Scikit-learn for preprocessing and regression models
- Matplotlib for visualization tools

## Contact
For questions or support, please contact:
- Email: your.email@example.com
- GitHub: [Your GitHub Profile](https://github.com/yourusername)
