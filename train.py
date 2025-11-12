import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import os

def train_and_save_model(dataset_path, model_save_path, scaler_save_path, target_column, id_column=None):
    """
    Loads data, trains a Support Vector Machine (SVM) model, and saves the model and scaler.
    """
    if not os.path.exists(dataset_path):
        print(f"Error: The file {dataset_path} was not found. Please download it and place it in the 'datasets' folder.")
        return

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    try:
        df = pd.read_csv(dataset_path)

        # Separate features (X) and target (Y)
        Y = df[target_column]
        # --- THIS IS THE FIX ---
        # Drop the target column and the non-numeric ID column if it exists
        X = df.drop(columns=[target_column], axis=1)
        if id_column:
            X = X.drop(columns=[id_column], axis=1)
        # ---------------------

        scaler = StandardScaler()
        scaler.fit(X)
        X_standardized = scaler.transform(X)

        X_train, X_test, Y_train, Y_test = train_test_split(X_standardized, Y, test_size=0.2, stratify=Y, random_state=2)

        classifier = SVC(kernel='linear', probability=True)
        classifier.fit(X_train, Y_train)

        joblib.dump(classifier, model_save_path)
        joblib.dump(scaler, scaler_save_path)
        
        print(f"Successfully trained and saved model from {dataset_path} to {model_save_path}")

    except Exception as e:
        print(f"An error occurred while processing {dataset_path}: {e}")

if __name__ == "__main__":
    # 1. Train Diabetes Model
    train_and_save_model(
        dataset_path='datasets/diabetes.csv',
        model_save_path='models/diabetes_model.pkl',
        scaler_save_path='models/diabetes_scaler.pkl',
        target_column='Outcome'
    )

    # 2. Train Heart Disease Model
    train_and_save_model(
        dataset_path='datasets/heart.csv',
        model_save_path='models/heart_model.pkl',
        scaler_save_path='models/heart_scaler.pkl',
        target_column='target'
    )

    # 3. Train Parkinson's Model (with the fix)
    train_and_save_model(
        dataset_path='datasets/parkinsons.csv',
        model_save_path='models/parkinsons_model.pkl',
        scaler_save_path='models/parkinsons_scaler.pkl',
        target_column='status',
        id_column='name'  # Tell the function to also drop the 'name' column
    )
