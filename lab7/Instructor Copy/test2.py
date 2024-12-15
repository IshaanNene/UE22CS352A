import sys
import importlib
import argparse
import torch 
import time

parser = argparse.ArgumentParser()
parser.add_argument('--ID', required=True)
args = parser.parse_args()
subname = args.ID

try:
    mymodule = importlib.import_module(subname)
except Exception as e:
    print(e)
    print("Rename your written program as CAMPUS_SECTION_SRN_Lab6.py and run python test.py --ID CAMPUS_SECTION_SRN_Lab6")
    sys.exit()

HMM = mymodule.HMM  

def test_case():
    # Financial states and their transitions in credit behavior
    states = ["Good Standing", "Mild Financial Strain", "High Financial Risk", "Default", "Recovering Credit"]
    observations = ["On-Time Payments", "Minimum Payment Only", "Late Payments", "Missed Payments", "Debt Consolidation"]

    # New transition probabilities
    transition = torch.tensor([
        [0.6, 0.2, 0.1, 0.05, 0.05],
        [0.3, 0.5, 0.1, 0.05, 0.05],
        [0.1, 0.1, 0.6, 0.1, 0.1],
        [0.05, 0.05, 0.15, 0.7, 0.05],
        [0.05, 0.05, 0.1, 0.05, 0.75]
    ])

    # New emission probabilities
    emission = torch.tensor([
        [0.7, 0.15, 0.1, 0.025, 0.025],
        [0.1, 0.7, 0.15, 0.025, 0.025],
        [0.025, 0.1, 0.7, 0.15, 0.025],
        [0.025, 0.025, 0.15, 0.7, 0.1],
        [0.05, 0.05, 0.1, 0.1, 0.7]
    ])

    # New prior probabilities
    priors = torch.tensor([0.5, 0.2, 0.15, 0.1, 0.05])

    # New observation sequence
    observation_sequence = ["Debt Consolidation", "Late Payments", "Missed Payments", "Minimum Payment Only"]

    model = HMM(transition, states, observations, priors, emission)

    # Test Viterbi Algorithm
    try:
        result = model.viterbi_algorithm(observation_sequence)
        expected = ['High Financial Risk', 'High Financial Risk', 'High Financial Risk', 'High Financial Risk']
        print("Test Case for Viterbi Test PASSED" if result == expected else "Test Case for Viterbi Test Fail")
    except Exception as e:
        print("Test Case for Viterbi Test Failed with Error:", e)

    # Test likelihood
    try:
        likelihood = round(model.calculate_likelihood(observation_sequence), 5)
        expected_range = (0.0003, 0.0004)
        print("Test Case for Likelihood Test PASSED" if expected_range[0] <= likelihood <= expected_range[1] else "Test Case for Likelihood Test Fail")
    except Exception as e:
        print("Test Case for Likelihood Test Failed with Error:", e)

    # Test likelihood with time limit
    try:
        lemon = ["Late Payments", "Missed Payments", "Debt Consolidation"]
        start_time = time.time()  
        jam = round(model.calculate_likelihood(lemon), 5)
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 5)
        if 0.003 <= jam <= 0.004 and elapsed_time < 0.002:
            print("New Test Case for the likelihood with time limit PASSED")
        else:
            print(f"New Test Case for the likelihood with time limit FAILED.")
    except Exception as e:
        print(f"New Test Case for the likelihood with time limit FAILED [ERROR]\n{e}")

if __name__ == "__main__":
    test_case()
