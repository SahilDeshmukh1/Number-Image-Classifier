# Number-Image-Classifier

## Project Overview
This project focuses on developing machine learning models to classify handwritten digits using the MNIST dataset. The goal is to assist schools in identifying students who may need motor skills support based on their handwriting. Two models were developed and compared:
- **K-Nearest Neighbors (KNN)**: A simple, instance-based learning method.
- **Multi-Layer Perceptron (MLP)**: A neural network capable of capturing non-linear patterns.

Key objectives:
- Build and tune machine learning models for digit recognition.
- Compare models based on performance and computational efficiency.
- Recommend the best model for real-time applications in schools.

---

## Dataset
- **Source**: MNIST dataset, consisting of 60,000 handwritten digit images.
- **Features**:
  - 784 columns representing pixel intensities of 28x28 grayscale images.
  - 1 label column indicating the digit (0–9).
- **Data Preprocessing**:
  - Selected 50 predictive columns based on the percentage of non-zero values.
  - Normalized pixel intensities for KNN and standardized features for MLP.
  - Split into:
    - Training Set: 70% (42,000 observations).
    - Validation Set: 15% (9,000 observations).
    - Test Set: 15% (9,000 observations).

---

## Methodology
1. **Data Preparation**:
   - Extracted relevant features and normalized/standardized the dataset.
   - Split data into training, validation, and test sets.

2. **Model 1: K-Nearest Neighbors (KNN)**:
   - Hyperparameter tuning:
     - Number of neighbors (`k`): 3, 5, 7, 9.
     - Weighting: Uniform vs. Distance.
     - Distance metrics: Euclidean vs. Manhattan.
   - Best configuration: `k=5`, distance weighting, and Manhattan distance.
   - Validation Accuracy: 93.5%.
   - Test Accuracy: 93%.

3. **Model 2: Multi-Layer Perceptron (MLP)**:
   - Hyperparameter tuning:
     - Hidden layers: 50, 100, two layers of 50 neurons.
     - Activation: ReLU and Tanh.
     - Solvers: Adam and SGD.
     - Learning rates: 0.001 and 0.01.
   - Best configuration: Single hidden layer of 100 neurons, ReLU activation, Adam solver, and learning rate of 0.001.
   - Validation Accuracy: 92.5%.
   - Test Accuracy: 92%.

4. **Model Comparison**:
   - KNN achieves slightly higher accuracy but is computationally intensive.
   - MLP provides faster predictions and is better suited for real-time applications.

---

## Technologies Used
- **Python**: For data preprocessing, modeling, and evaluation.
- **Libraries**:
  - `pandas`, `numpy`: Data manipulation.
  - `scikit-learn`: Model building and evaluation.
  - `matplotlib`: Visualization.

---

## Results
### K-Nearest Neighbors (KNN):
- **Test Accuracy**: 93%.
- **Strengths**: High accuracy.
- **Weaknesses**: Computationally expensive for predictions due to instance-based design.

### Multi-Layer Perceptron (MLP):
- **Test Accuracy**: 92%.
- **Strengths**: Faster predictions after training, suitable for real-time applications.
- **Weaknesses**: Requires more computational resources during training.

### Recommendation:
- The **MLP model** is recommended for the school's needs due to its real-time prediction capability, making it ideal for touchpad handwriting recognition systems.
