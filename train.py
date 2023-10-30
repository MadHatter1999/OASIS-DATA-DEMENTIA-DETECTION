import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    # Load the dataset
    data = pd.read_csv('data\oasis_longitudinal_modified.csv')
    # Encode categorical variables
    le_gender = LabelEncoder()
    data['M/F'] = le_gender.fit_transform(data['M/F'])

    le_cdr_class = LabelEncoder()
    data['CDR_Class'] = le_cdr_class.fit_transform(data['CDR_Class'])

    # Define features and target variable
    X = data.drop(columns=['Subject ID', 'MRI ID', 'Group', 'CDR', 'CDR_Class'])
    y = data['CDR_Class']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Save the scaler
    joblib.dump(scaler, 'scaler.joblib')
    # Build and train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save the model
    joblib.dump(model, 'model.joblib')
    return X_test_scaled, y_test

if __name__ == "__main__":
    
    train_model()