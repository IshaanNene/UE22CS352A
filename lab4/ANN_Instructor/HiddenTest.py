import importlib
import os
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

LAB_NO = 1
SRN_PATTERN = re.compile(r'PES2UG21CS\d{3}', re.IGNORECASE)


# Load the dataset
data = pd.read_csv('modified_wineQT.csv')
X = data.drop('quality', axis=1)
y = data['quality']

model1 = model2 = None
X_train = X_test = y_train = y_test = None

def test_case(subname, srn):
    print("-" * 100)
    print(srn)

    try:
        mymodule = importlib.import_module(subname[:-3])
        split_and_standardize = mymodule.split_and_standardize
        create_model = mymodule.create_model
        predict_and_evaluate = mymodule.predict_and_evaluate
    
    except AttributeError as e:
        print(f"Error loading functions: {e}")
        print("Could not load functions")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Could not load functions")
        return

    try:
        X_train, X_test, y_train, y_test = split_and_standardize(X, y)
        if (X_train.shape[0] + X_test.shape[0] == X.shape[0] and
            X.shape[0] == (y_train.shape[0] + y_test.shape[0]) and
            np.allclose(np.mean(X_train), 0, atol=1e-1) and
            np.allclose(np.mean(X_test), 0, atol=1e-1)):
            print("Test Case 1 for the function split_and_standardize PASSED")
        else:
            print("Test Case 1 for the function split_and_standardize FAILED")
    except Exception as e:
        print(f"Error in split_and_standardize: {e}")
        print("Test Case 1 for the function split_and_standardize FAILED [ERROR]")

    try:
        model1, model2 = create_model(X_train, y_train)
        if (len(model1.get_params()['hidden_layer_sizes']) == 3 and
            len(model2.get_params()['hidden_layer_sizes']) == 3 ):
            print("Test Case 2 for the function create_model PASSED")
        else:
            print("Test Case 2 for the function create_model FAILED")
    except Exception as e:
        print(f"Error in create_model: {e}")
        print("Test Case 2 for the function create_model FAILED [ERROR]")

    try:
        accuracy, precision, recall, fscore, conf_matrix = predict_and_evaluate(model1, X_test, y_test)
        if accuracy >= 0.55:
            print("Test Case 3 for the function predict_and_evaluate PASSED")
        else:
            print("Test Case 3 for the function predict_and_evaluate FAILED")
    except Exception as e:
        print(f"Error in predict_and_evaluate: {e}")
        print("Test Case 3 for the function predict_and_evaluate FAILED [ERROR]")

    print("-" * 100)

if __name__ == "__main__":
    STUDENT_SOLUTION_PATH = os.getcwd()

    for studentSolution in os.listdir(STUDENT_SOLUTION_PATH):
        if "PES" in studentSolution:
            try:
                srn = re.findall(SRN_PATTERN, studentSolution)[0]
                test_case(studentSolution, srn)
            except IndexError:
                print(f"No SRN found in file name: {studentSolution}")
            except Exception as e:
                print(f"Unexpected error processing file {studentSolution}: {e}")
