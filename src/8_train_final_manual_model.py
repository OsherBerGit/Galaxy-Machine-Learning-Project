"""
8_train_final_manual_model.py - Train and Save the Final Model

This script trains the final GalaxyAdaBoost model (our manual implementation)
and saves it as a pickle file for deployment in the Streamlit app.

Process:
    1. Load the feature dataset (galaxy_features.csv)
    2. Split into train/test (80/20)
    3. Initialize GalaxyAdaBoost with 200 estimators
    4. Train on training data
    5. Evaluate on test data and print accuracy
    6. Save the trained model to models/final_galaxy_model.pkl

Output: models/final_galaxy_model.pkl (ready for production use)
"""
import pandas as pd
import pickle
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Important Import ---
# We import the class from the file we created above
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from galaxy_adaboost import GalaxyAdaBoost

def train_and_save_final():
    print("--- Training Final Manual Model (Separated Class) ---")
    
    # 1. Load data
    try:
        df = pd.read_csv('data/galaxy_features.csv')
    except FileNotFoundError:
        print("Error: galaxy_features.csv not found.")
        return

    X = df.drop(columns=['filename', 'label']).values
    y = df['label'].values

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Use the imported class
    print("Initializing GalaxyAdaBoost...")
    my_model = GalaxyAdaBoost(n_estimators=200)
    
    # 3. Training
    print("Fitting model...")
    my_model.fit(X_train, y_train)

    # 4. Testing
    preds = my_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n✅ Final Accuracy: {acc*100:.2f}%")

    # 5. Save
    model_path = 'models/final_galaxy_model.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(my_model, f)
    
    print(f"✅ Model saved to: {model_path}")
    print("   (Ready for UI integration)")

if __name__ == "__main__":
    train_and_save_final()