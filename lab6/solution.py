import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class NaiveBayesClassifier:

    @staticmethod
    def preprocess(sentences: list, categories: list) -> tuple:

        cleaned_sentences = []
        cleaned_categories = []

        # Define valid categories
        valid_categories = {"entertainment", "technology", "travel", "food", "literature", "health"}

        # Filter out records with missing or incorrect labels
        filtered_data = [(s, c) for s, c in zip(sentences, categories) if c in valid_categories]

        # Remove stop words from sentences
        cleaned_sentences = []
        for sentence, category in filtered_data:
            cleaned_sentence = " ".join([word for word in sentence.split() if word.lower() not in ENGLISH_STOP_WORDS])
            cleaned_sentences.append(cleaned_sentence)

        cleaned_categories = [category for _, category in filtered_data]

        return cleaned_sentences, cleaned_categories
    
    @staticmethod
    def fit(X: np.ndarray, y: np.ndarray) -> tuple:

        class_probs = {}
        word_probs = {}

        classes = np.unique(y)
        total_documents = len(y)

        for c in classes:
            c_index = [index for index, value in enumerate(y) if value == c]
            X_c = X[c_index]
            class_probs[c] = float(len(X_c)) / float(total_documents)
            word_probs[c] = (np.sum(X_c, axis=0) + 1.0) / float(np.sum(X_c) + len(X[0]))

        return class_probs, word_probs

    @staticmethod
    def predict(X, class_probs, word_probs, classes):

        predictions = []

        for x in X:
            class_scores = {c: np.log(class_probs[c]) for c in classes}
            for c in classes:
                class_scores[c] += np.sum(np.log(word_probs[c]) * x)

            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)

        return predictions

    
