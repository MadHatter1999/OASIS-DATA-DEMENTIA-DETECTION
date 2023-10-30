import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
from imblearn.over_sampling import SMOTE

def train_model():
    print("Loading the dataset")
    data = pd.read_csv('data/oasis_longitudinal_modified.csv')

    print("Encoding categorical variables")
    le_gender = LabelEncoder()
    data['M/F'] = le_gender.fit_transform(data['M/F'])

    le_cdr_class = LabelEncoder()
    data['CDR_Class'] = le_cdr_class.fit_transform(data['CDR_Class'])

    print("Defining features and target variable")
    X = data.drop(columns=['Subject ID', 'MRI ID', 'Group', 'CDR', 'CDR_Class'])
    y = data['CDR_Class']

    print("Splitting the data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Applying SMOTE")
    smote = SMOTE(random_state=42, k_neighbors=2)  # Set k_neighbors to 2 or less
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print("Scaling numerical features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_resampled)
    X_test_scaled = scaler.transform(X_test)

    print("Saving the scaler")
    joblib.dump(scaler, 'data/scaler.joblib')

    print("Building and training the model")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_resampled)

    print("Saving the model")
    joblib.dump(model, 'data/model.joblib')

    print("Model training complete")
    return X_test_scaled, y_test

if __name__ == "__main__":
    train_model()