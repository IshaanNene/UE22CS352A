from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from naive_bayes_solution import NaiveBayesClassifier
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def run_tests(test_cases):
    """
    Runs test cases on the trained Naive Bayes classifier.

    Args:
        test_cases (list): List of tuples with test sentence and correct category.

    Returns:
        int: Number of test cases that passed.
    """
    # Defining the training sentences and categories (with possible noise and missing labels)
    sentences = [
        "The new smartphone features advanced AI capabilities.",  # Technology
        "Traveling to Japan offers a rich cultural experience.",  # Travel
        "The latest blockbuster movie has stunning visual effects.",  # Entertainment
        "Artificial intelligence is transforming various industries.",  # Technology
        "A journey through the Alps is unforgettable.",  # Travel
        "The documentary highlighted the importance of wildlife conservation.",  # Entertainment
        "5G technology enables faster internet speeds.",  # Technology
        "Exploring the beaches of Thailand is a dream for many travelers.",  # Travel
        "This animated film is a must-watch for families.",  # Entertainment
        "The rise of electric vehicles is changing the automotive industry.",  # Technology
        "Adventure tourism is becoming increasingly popular.",  # Travel
        "The series finale left fans wanting more.",  # Entertainment
        "Cloud computing is revolutionizing data storage.",  # Technology
        "I enjoy backpacking through South America.",  # Travel
        "The film festival showcases independent filmmakers."  # Entertainment
    ]

    categories = [
        "technology", "travel", "entertainment", "technology", "travel", "entertainment", 
        "technology", "travel", "entertainment", "technology", "travel", "entertainment",
        "technology", "travel", "entertainment"
    ]

    # Preprocessing step
    sentences, categories = NaiveBayesClassifier.preprocess(sentences, categories)

    # Vectorizing the text data using CountVectorizer
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(sentences)

    # Fitting the Naive Bayes model
    class_probs, word_probs = NaiveBayesClassifier.fit(X_train_vec.toarray(), categories)

    num_passed = 0

    for test_sentence, correct_category in test_cases:
        test_vector = vectorizer.transform([test_sentence]).toarray()
        prediction = NaiveBayesClassifier.predict(test_vector, class_probs, word_probs, np.unique(categories))[0]

        if prediction == correct_category:
            print(f"Test Passed: '{test_sentence}' - Predicted: {prediction} | Correct: {correct_category}")
            num_passed += 1
        else:
            print(f"Test Failed: '{test_sentence}' - Predicted: {prediction} | Correct: {correct_category}")

    return num_passed


if __name__ == "__main__":

    test_cases = [
        ("The new smartphone has impressive features.", "technology"),
        ("Traveling through Europe is an enriching experience.", "travel"),
        ("This film received rave reviews for its storytelling.", "entertainment"),
        ("Artificial intelligence is advancing rapidly.", "technology"),
        ("Exploring the ancient ruins of Petra in Jordan is on my travel bucket list.", "travel"),
        ("The concert was a spectacular show.", "entertainment"),
        ("The development of autonomous vehicles is reshaping urban transportation.", "technology"),
        ("Exploring the wonders of the Amazon rainforest is on my bucket list.", "travel"),
        ("The movie's soundtrack was incredible.", "entertainment"),
        ("The impact of technology on society is profound.", "technology")
    ]

    num_passed = run_tests(test_cases)
    print(f"\nNumber of Test Cases Passed: {num_passed} out of {len(test_cases)}")
