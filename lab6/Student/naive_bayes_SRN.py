import numpy as np
import warnings
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

warnings.filterwarnings("ignore", category=RuntimeWarning)


class NaiveBayesClassifier:
    """
    A simple implementation of the Naive Bayes Classifier for text classification.
    """
    @staticmethod
    def preprocess(sentences, categories):
        """
        Preprocess the dataset to remove stop words, and missing or incorrect labels.

        Args:
            sentences (list): List of sentences to be processed.
            categories (list): List of corresponding labels.

        Returns:
            tuple: A tuple of two lists - (cleaned_sentences, cleaned_categories).
        """
        # TO DO 
        
    @staticmethod
    def fit(X, y):
        """
        Trains the Naive Bayes Classifier using the provided training data.
        
        Args:
            X (numpy.ndarray): The training data matrix where each row represents a document
                              and each column represents the presence (1) or absence (0) of a word.
            y (numpy.ndarray): The corresponding labels for the training documents.

        Returns:
            tuple: A tuple containing two dictionaries:
                - class_probs (dict): Prior probabilities of each class in the training set.
                - word_probs (dict): Conditional probabilities of words given each class.
        """
        
        class_probs = {}
        word_probs = {}
       
       # TO DO

        return class_probs, word_probs

    @staticmethod
    def predict(X, class_probs, word_probs, classes):
        """
        Predicts the classes for the given test data using the trained classifier.

        Args:
            X (numpy.ndarray): The test data matrix where each row represents a document
                              and each column represents the presence (1) or absence (0) of a word.
            class_probs (dict): Prior probabilities of each class obtained from the training phase.
            word_probs (dict): Conditional probabilities of words given each class obtained from training.
            classes (numpy.ndarray): The unique classes in the dataset.

        Returns:
            list: A list of predicted class labels for the test documents.
        """
        predictions = []

        # TO DO

        return predictions

    
