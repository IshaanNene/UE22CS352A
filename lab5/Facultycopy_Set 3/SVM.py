import pandas as pd  # For data manipulation
from sklearn.preprocessing import StandardScaler  # To standardize features by removing the mean and scaling to unit variance
from sklearn.svm import SVC, SVR  # SVM classifiers and regressors from scikit-learn
from sklearn.model_selection import train_test_split  # For splitting dataset into training and test sets
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_percentage_error  # Metrics for evaluating models
import matplotlib.pyplot as plt  # For data visualization

# Define a class for SVM-based classification
class SVM_Classification:
    def __init__(self) -> None:
        """Initialize the SVM Classification model class."""
        self.model = None  # Placeholder for the SVM model
    
    def dataset_read(self, dataset_path):
        """
        Reads a dataset from a JSON file and separates it into features and target.
        :param dataset_path: The file path to the dataset in JSON format.
        :return: X (features), y (target)
        """
        data = pd.read_json(dataset_path)  # Read dataset from JSON file
        # X contains all columns except the last one (features)
        X = data.iloc[:, 0:-1]
        # y contains the last column (target)
        y = data.iloc[:, -1]
        return X, y
    
    def preprocess(self, X, y):
        """
        Preprocess the dataset by handling missing values and standardizing features.
        :param X: Features (input variables)
        :param y: Target (output variable)
        :return: Preprocessed X and y
        """
        # Check for any missing values and handle them by replacing with mean
        if X.isnull().sum().sum() > 0:
            X = X.fillna(X.mean())  # Replace missing values with the column mean
        
        # Standardize the feature set (X) using StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)  # Fit to the data and transform
        
        return X, y
    
    def train_classification_model(self, X_train, y_train):
        """
        Train the SVM classifier using the training data.
        :param X_train: Training set features
        :param y_train: Training set labels
        """
        # Initialize the SVM classifier with a sigmoid kernel and random state for reproducibility
        self.model = SVC(kernel='sigmoid', random_state=42)
        # Fit the model to the training data
        self.model.fit(X_train, y_train)
    
    def predict_accuracy(self, X_test, y_test):
        """
        Predict labels on test data and calculate accuracy.
        :param X_test: Test set features
        :param y_test: True labels for the test set
        :return: Accuracy of the model on test set
        """
        # Predict the labels for the test set
        y_pred = self.model.predict(X_test)
        # Calculate accuracy score by comparing predicted and actual labels
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy


# Define a class for SVM-based regression
class SVM_Regression:
    def __init__(self) -> None:
        """Initialize the SVM Regression model class."""
        self.model = None  # Placeholder for the SVM model
    
    def dataset_read(self, dataset_path):
        """
        Reads a dataset from a JSON file and separates it into features and target.
        :param dataset_path: The file path to the dataset in JSON format.
        :return: X (features), y (target)
        """
        data = pd.read_json(dataset_path)  # Read dataset from JSON file
        # X contains all columns except the last one (features)
        X = data.iloc[:, 0:-1]
        # y contains the last column (target)
        y = data.iloc[:, -1]
        return X, y
    
    def preprocess(self, X, y):
        """
        Preprocess the dataset by handling missing values and standardizing features.
        :param X: Features (input variables)
        :param y: Target (output variable)
        :return: Preprocessed X and y
        """
        # Check for any missing values and handle them by replacing with mean
        if X.isnull().sum().sum() > 0:
            X = X.fillna(X.mean())  # Replace missing values with the column mean
        
        # Standardize the feature set (X) using StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)  # Fit to the data and transform
        
        return X, y
    
    def train_regression_model(self, X_train, y_train):
        """
        Train the SVM regression model using the training data.
        :param X_train: Training set features
        :param y_train: Training set target values
        """
        # Initialize the SVM regressor with RBF kernel, regularization parameter C, and epsilon for defining tolerance
        self.model = SVR(kernel='rbf', C=1, epsilon=0.5)
        # Fit the model to the training data
        self.model.fit(X_train, y_train)
    
    def predict_accuracy(self, X_test, y_test):
        """
        Predict target values on test data and calculate the model's accuracy using Mean Absolute Percentage Error (MAPE).
        :param X_test: Test set features
        :param y_test: True target values for the test set
        :return: Accuracy of the regression model (1 - MAPE)
        """
        # Predict the target values for the test set
        y_pred = self.model.predict(X_test)
        # Calculate Mean Absolute Percentage Error (MAPE)
        err = mean_absolute_percentage_error(y_test, y_pred)
        # Return accuracy as 1 - MAPE (since lower MAPE means higher accuracy)
        return 1 - err

    def visualize(self, X_test, y_test, y_pred):
        """
        Visualize the actual vs predicted values on a scatter plot.
        :param X_test: Test set features
        :param y_test: Actual target values
        :param y_pred: Predicted target values
        """
        plt.figure(figsize=(10, 6))  # Set the figure size
        
        # Scatter plot of actual values (y_test) vs features (X_test)
        plt.scatter(X_test, y_test, color='blue', alpha=0.6, edgecolor='k', label='Actual Target')
        
        # Scatter plot of predicted values (y_pred) vs features (X_test)
        plt.scatter(X_test, y_pred, color='red', alpha=0.6, edgecolor='k', label='Predicted Target')
        
        # Set plot title and labels
        plt.title('Feature vs Target')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.legend()  # Show the legend
        plt.grid(True)  # Show grid lines
        plt.show()  # Display the plot

class SVM_Spiral:
    def __init__(self) -> None:
        """Initialize the SVM classification model for the spiral dataset."""
        # Initialize the model attribute, which will hold the trained SVM model
        self.model = None

    # Function to read the dataset from a JSON file
    def dataset_read(self, dataset_path):
        """
        Reads the dataset from a JSON file and separates it into features and target.
        :param dataset_path: The file path to the dataset in JSON format.
        :return: Features (X) and target variable (y).
        """
        # Read the data from a JSON file into a pandas DataFrame
        data = pd.read_json(dataset_path)
        
        # X -> Features (all columns except the last one)
        X = data.iloc[:, 0:-1]
        
        # y -> Target variable (the last column)
        y = data.iloc[:, -1]
        
        return X, y
    
    # Function to preprocess the data
    def preprocess(self, X, y):
        """
        Handles missing values in the dataset and standardizes the features.
        :param X: Features (input variables).
        :param y: Target (output variable).
        :return: Preprocessed features (X) and target (y).
        """
        # Check for any missing values in the dataset
        if X.isnull().sum().sum() > 0:
            # If missing values exist, fill them with the column mean
            X = X.fillna(X.mean())

        # Standardize the feature set (mean=0 and std=1), as SVMs perform better with normalized data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, y
    
    # Function to train the SVM model for the spiral dataset
    def train_spiral_model(self, X_train, y_train):
        """
        Train the SVM classification model for the spiral dataset using an RBF kernel.
        :param X_train: Training set features.
        :param y_train: Training set labels.
        """
        # Initialize the SVC model with an RBF kernel, gamma=20, and C=20 for non-linear classification
        self.model = SVC(kernel='rbf', gamma=20, C=20)
        
        # Fit the model on the training data
        self.model.fit(X_train, y_train)
    
    # Function to evaluate the trained model's accuracy on the test data
    def predict_accuracy(self, X_test, y_test):
        """
        Predict labels on the test set and evaluate accuracy.
        :param X_test: Test set features.
        :param y_test: True test set labels.
        :return: Accuracy score of the model.
        """
        # Predict the target values using the test data
        y_pred = self.model.predict(X_test)
        
        # Calculate and return the accuracy score between true values and predicted values
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
