import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from collections import Counter

def evaluate_model(X_test, y_test, n_iterations=10):
    # Load the trained model
    model = joblib.load('data/model.joblib')

    # Load label encoder for CDR class
    le_cdr_class = joblib.load('data/le_cdr_class.joblib')

    # Store accuracies from multiple predictions
    accuracies = []

    print("Evaluating the model over multiple iterations:")
    for i in range(n_iterations):
        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        # Decode the predicted labels
        y_pred_labels = le_cdr_class.inverse_transform(y_pred)

        # Find the most common label
        most_common_label = Counter(y_pred_labels).most_common(1)[-1]

        print(f'\nIteration {i + 1}:')
        print(f'Accuracy: {accuracy:.2f}')
        print('Most common predicted label:', most_common_label)

    # Calculate and print median accuracy
    median_accuracy = np.median(accuracies)
    print(f'\nMedian Accuracy across all iterations: {median_accuracy:.2f}')

if __name__ == "__main__":
    # Load the test data
    X_test = joblib.load('data/X_test.joblib')
    y_test = joblib.load('data/y_test.joblib')

    evaluate_model(X_test, y_test)
