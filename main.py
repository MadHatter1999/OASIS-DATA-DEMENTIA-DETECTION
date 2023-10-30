from train import train_model
from evaluate import evaluate_model

if __name__ == "__main__":
    X_test, y_test = train_model()
    evaluate_model(X_test, y_test)
