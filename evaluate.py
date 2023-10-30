import joblib
from sklearn.metrics import accuracy_score

def evaluate_model(X_test, y_test):
    # Load the trained model
    model = joblib.load('model.joblib')
    # Make predictions
    y_pred = model.predict(X_test)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

if __name__ == "__main__":
    # This script is supposed to be run after train_model.py
    # Therefore, X_test and y_test should be provided by the user or saved and loaded properly
    pass