import numpy as np
import joblib
from sklearn.metrics import accuracy_score

def evaluate_model(X_test, y_test, n_iterations=10):
    # Load the trained model
    model = joblib.load('data/model.joblib')

    # Store accuracies from multiple predictions
    accuracies = []

    for i in range(n_iterations):
        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f'Iteration {i + 1}, Accuracy: {accuracy:.2f}')

    # Calculate median accuracy
    median_accuracy = np.median(accuracies)
    print(f'\nMedian Accuracy: {median_accuracy:.2f}')

if __name__ == "__main__":
    # Load the test data
    X_test = joblib.load('data/X_test.joblib')
    y_test = joblib.load('data/y_test.joblib')

    evaluate_model(X_test, y_test)
