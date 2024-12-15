from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import argparse
import importlib
from os import path
import sys


def run_tests(test_cases):

    # Defining the training sentences and categories
    sentences = [
        "The new smartphone has a stunning display and battery life.",  # Technology
    "AI is transforming the future of work.",  # Technology
    "5G networks are rolling out across the globe.",  # Technology
    "Electric cars are the future of transportation.",  # Technology
    "Virtual reality is reshaping gaming.",  # Technology
    "This pizza recipe is easy and delicious.",  # Food
    "This pasta dish is perfect for family dinners.",  # Food
    "Chocolate cake is my favorite dessert.",  # Food
    "The ice cream truck arrived on a hot day.",  # Food
    "A healthy smoothie can boost your energy.",  # Food
    "The latest movie has been a box office hit.",  # Entertainment
    "The concert was an unforgettable experience.",  # Entertainment
    "The streaming platform has released new shows.",  # Entertainment
    "A new documentary series is captivating audiences.",  # Entertainment
    "The theater production received rave reviews.",  # Entertainment
    "Regular exercise is essential for a healthy lifestyle.",  # Health
    "Meditation can improve mental well-being.",  # Health
    "Eating a balanced diet is crucial for good health.",  # Health
    "Getting enough sleep can boost your productivity.",  # Health
    "The beach resort offers breathtaking views.",  # Travel
    "Exploring new cities can broaden your perspective.",  # Travel
    "Hiking in the mountains is a refreshing escape.",  # Travel
    "Cultural festivals showcase the diversity of traditions.",  # Travel
    "The latest fashion trends are bold and vibrant.",  # Fashion
    "Sustainable clothing is becoming increasingly popular.",  # Fashion
    "This season's colors are inspired by nature.",  # Fashion
    "Traveling can enrich your life experiences.",  # Travel
    "Missing label",  # Missing label (noise)
    "Incorrect label"  # Incorrect label (noise)
    ]

    categories = [
        "Technology",
    "Technology",
    "Technology",
    "Technology",
    "Technology",
    "Food",
    "Food",
    "Food",
    "Food",
    "Food",
    "Entertainment",
    "Entertainment",
    "Entertainment",
    "Entertainment",
    "Entertainment",
    "Health",
    "Health",
    "Health",
    "Health",
    "Travel",
    "Travel",
    "Travel",
    "Travel",
    "Fashion",
    "Fashion",
    "Fashion",
    "Travel",
    "Missing label (noise)",
    "Incorrect label (noise)"
    ]
    
    try:
        # Preprocessing step
        sentences, categories = NaiveBayesClassifier.preprocess(sentences, categories)
    except Exception as e:
        print(f'\nError in preprocess function:\n{e}')
        sys.exit(1)

    try:
        # Vectorizing the text data using CountVectorizer
        vectorizer = CountVectorizer()
        X_train_vec = vectorizer.fit_transform(sentences)
    except Exception as e:
        print(f'\nError in vectorizing output of preprocess fucntion:\n{e}')
        sys.exit(1)

    try:
        # Fitting the Naive Bayes model
        class_probs, word_probs = NaiveBayesClassifier.fit(X_train_vec.toarray(), categories)
    except Exception as e:
        print(f'\nError in fit function:\n{e}')
        sys.exit(1)

    num_passed = 0

    try:
        print()
        for test_sentence, correct_category in test_cases:
            test_vector = vectorizer.transform([test_sentence]).toarray()
            prediction = NaiveBayesClassifier.predict(test_vector, class_probs, word_probs, np.unique(categories))[0]

            if prediction == correct_category:
                print(f"Test Passed: '{test_sentence}' - Predicted: {prediction} | Correct: {correct_category}")
                num_passed += 1
            else:
                print(f"Test Failed: '{test_sentence}' - Predicted: {prediction} | Correct: {correct_category}")
    except Exception as e:
        print(f'\nError in counting checking test cases:\n{e}')
        sys.exit(1)

    return num_passed


if __name__ == "__main__":

    parser = argparse.ArgumentParser(usage=f"python {path.basename(__file__)} --id CAMPUS_SECTION_SRN_Lab6.py")
    parser.add_argument("--id", required=True, help="where ID is CAMPUS_SECTION_SRN_Lab6.py")
    args = parser.parse_args()
    module = importlib.import_module(args.id.rstrip(".py"))
    NaiveBayesClassifier = module.NaiveBayesClassifier

    test_cases = [
    ("The smartphone has amazing battery life.", "Technology"),
    ("Exploring the ancient temples in Angkor Wat is a breathtaking experience.", "Travel"),
    ("The movie was a box office hit.", "Entertainment"),
    ("AI will shape the future of transportation.", "Technology"),
    ("Traveling through Europe has been a life-changing experience.", "Travel"),
    ("The concert was mind-blowing.", "Entertainment"),
    ("The latest advancements in electric cars are impressive.", "Technology"),
    ("A road trip across the country offers countless adventures.", "Travel"),
    ("The latest blockbuster movie kept everyone on the edge of their seats.", "Entertainment"),
    ("The impact of technology on our daily lives is profound.", "Technology"),
    ("A healthy smoothie made with spinach and banana can kickstart your day.", "Food"),
    ("Virtual reality games offer immersive experiences that captivate players.", "Technology"),
    ("The documentary highlighted the struggles and triumphs of the local community.", "Entertainment"),
    ("A delicious pasta dish with homemade sauce can bring families together.", "Food"),
    ("Cutting-edge technology is driving advancements in healthcare and medicine.", "Technology")
    ]

    num_passed = run_tests(test_cases)
    print(f"\nNumber of Test Cases Passed: {num_passed} out of {len(test_cases)}")
    
    
