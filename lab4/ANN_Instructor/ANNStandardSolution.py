import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Load, preprocess the data, and split into features and target
# input: file_path: str (path to the dataset)
# output: tuple of X (features) and y (target)
def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Drop the 'Id' and 'garbage_column' columns
    df = df.drop(['Id', 'garbage_column'], axis=1)
    
    # Separate features and target
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    return X, y


# Split the data into training and testing sets and standardize the features
# input: 1) X: list/ndarray (features)
#        2) y: list/ndarray (target)
# output: split: tuple of X_train, X_test, y_train, y_test
def split_and_standardize(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    split = X_train, X_test, y_train, y_test
    return split


# Create and train 2 MLP classifiers with different parameters
# input:  1) X_train: list/ndarray
#         2) y_train: list/ndarray
# output: 1) models: model1, model2 - tuple
def create_model(X_train, y_train):
    parameters1 = {
        "hidden_layer_sizes": (64, 32, 16),
        "learning_rate_init": 0.001, 
        "max_iter": 1000,
        "random_state": 42,
        "activation": "tanh"
    }
    model1 = MLPClassifier(**parameters1)
    model1.fit(X_train, y_train)

    parameters2 = {
        "hidden_layer_sizes": (16, 8, 4), 
        "learning_rate_init": 0.005,
        "random_state": 42,
        "max_iter": 1000,
        "solver": "adam",
        "activation": "tanh"
    }
    model2 = MLPClassifier(**parameters2)
    model2.fit(X_train, y_train)

    models = model1, model2
    return models


# Predict and evaluate the model's performance
# input  : 1) model: MLPClassifier after training
#          2) X_test: list/ndarray
#          3) y_test: list/ndarray
# output : 1) metrics: tuple - accuracy, precision, recall, fscore, confusion matrix
def predict_and_evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    fscore = f1_score(y_test, y_pred, average="macro")
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    metrics = accuracy, precision, recall, fscore, conf_matrix
    return metrics


# Example usage
if __name__ == "__main__":
    # Load and preprocess the data
    X, y = load_and_preprocess_data('modified_wineQT.csv')
    
    # Split and standardize the data
    X_train, X_test, y_train, y_test = split_and_standardize(X, y)
    
    # Create and train the models
    model1, model2 = create_model(X_train, y_train)
    
    # Evaluate the models
    metrics1 = predict_and_evaluate(model1, X_test, y_test)
    metrics2 = predict_and_evaluate(model2, X_test, y_test)
    
    # Print the metrics for both models
    print("Model 1 Metrics:")
    print(f"Accuracy: {metrics1[0]:.4f}, Precision: {metrics1[1]:.4f}, Recall: {metrics1[2]:.4f}, F1 Score: {metrics1[3]:.4f}")
    print("Confusion Matrix:\n", metrics1[4])

    print("\nModel 2 Metrics:")
    print(f"Accuracy: {metrics2[0]:.4f}, Precision: {metrics2[1]:.4f}, Recall: {metrics2[2]:.4f}, F1 Score: {metrics2[3]:.4f}")
    print("Confusion Matrix:\n", metrics2[4])
