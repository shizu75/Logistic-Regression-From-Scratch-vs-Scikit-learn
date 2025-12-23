# Logistic Regression: From Scratch vs Scikit-learn

## Overview
This repository implements binary classification using Logistic Regression in two ways: a complete from-scratch implementation using NumPy and gradient descent, and a standard implementation using Scikit-learn’s LogisticRegression model. The project focuses on understanding the mathematical foundations of logistic regression, optimization, cost minimization, and model evaluation.

## Datasets
The project uses four CSV files:
- train_X.csv – Training features  
- train_Y.csv – Training labels  
- test_X.csv – Testing features  
- test_Y.csv – Testing labels  

Each dataset contains an `Id` column that is removed during preprocessing. Features are numerical, and the target variable represents binary class labels.

## Data Preprocessing
All datasets are loaded using Pandas and converted to NumPy arrays. The `Id` column is dropped. Feature matrices are transposed to align with vectorized mathematical operations. Target vectors are reshaped to match model dimensions. In the Scikit-learn implementation, missing values in features such as Age and Fare are handled using mean imputation.

## Logistic Regression From Scratch
The first implementation builds Logistic Regression manually using NumPy. Model parameters (weights and bias) are initialized to zero. The sigmoid activation function is used to map predictions to probabilities. Binary cross-entropy loss is implemented as the cost function. Gradient descent is applied to iteratively update weights and bias over a fixed number of iterations with a defined learning rate. Training cost values are stored to analyze convergence.

## Training and Cost Visualization
During training, the cost is printed periodically to monitor optimization progress. After training, a cost-versus-iterations plot is generated to visualize convergence behavior of gradient descent.

## Model Evaluation (From Scratch)
Model performance is evaluated using classification accuracy. Predictions are thresholded at 0.5 to obtain binary outputs. Accuracy is computed manually and reported as a percentage on the test dataset.

## Scikit-learn Logistic Regression
The second implementation uses Scikit-learn’s LogisticRegression model. Training data is fitted directly using the library. Predictions are generated for the test dataset. Model evaluation includes Mean Squared Error, R-squared score, accuracy score, and built-in model scores for both training and testing data.

## Visualization
A prediction plot is generated using the Scikit-learn model to visualize predicted outputs against one of the input features. This provides an intuitive comparison between model predictions and input distribution.

## Technologies Used
Python, Pandas, NumPy, Matplotlib, Scikit-learn

## How to Run
Clone the repository, install required dependencies, update dataset paths in the scripts, and run the Python files for both implementations.

## Applications
Binary classification problems, survival prediction, medical diagnosis, risk analysis, and machine learning education.

## Future Work
Possible extensions include feature scaling, regularization techniques, hyperparameter tuning, ROC-AUC analysis, confusion matrix visualization, and extension to multiclass classification.

## Author
Soban Saeed
GitHub: https://github.com/shizu75

## License
MIT
