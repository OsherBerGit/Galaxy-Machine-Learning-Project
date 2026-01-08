import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- 1. The Weak Learner (Decision Stump) ---
class DecisionStump:
    """
    A 'Stump' is a Decision Tree with depth=1.
    It splits the data based on a single feature and a single threshold.
    """
    def __init__(self):
        self.polarity = 1       # Determines if sample should be classified as -1 or 1 for the given threshold
        self.feature_idx = None # The index of the feature used for splitting
        self.threshold = None   # The value used for splitting
        self.alpha = None       # The 'say' (weight) of this stump in the final decision

    def predict(self, X):
        """
        Predicts class labels (-1 or 1) based on the learned threshold.
        """
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)

        # Logic: If polarity is 1, classify samples below threshold as -1
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column < self.threshold] = 1 # Inverse logic

        return predictions

# --- 2. The Main Algorithm (AdaBoost) ---
class CustomAdaBoost:
    """
    Implementation of Adaptive Boosting algorithm from scratch.
    """
    def __init__(self, n_clf=50):
        self.n_clf = n_clf  # Number of classifiers (stumps) to train
        self.clfs = []      # List to store all the trained stumps

    def fit(self, X, y):
        """
        Main training loop.
        X: Feature matrix
        y: Labels (must be -1 and 1)
        """
        n_samples, n_features = X.shape
        
        # Initialize weights: w_i = 1/N
        w = np.full(n_samples, (1 / n_samples))
        
        self.clfs = [] # Reset classifiers

        print(f"--- Starting training of {self.n_clf} classifiers ---")

        # Iterate to build n_clf stumps
        for clf_idx in range(self.n_clf):
            
            clf = DecisionStump()
            min_error = float('inf') # Track the best split found so far
            
            # Greedy search: Try every feature and every threshold
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column) # Possible split points

                for threshold in thresholds:
                    # Polarity 1: Predict 1 if > threshold
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    # Calculate Weighted Error
                    # Error is the sum of weights of misclassified samples
                    misclassified = w[y != predictions]
                    error = sum(misclassified)

                    # Store best configuration
                    if error > 0.5:
                        # If error is worse than random guessing, flip polarity
                        error = 1 - error
                        p = -1

                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error

            # Step 2: Calculate 'Alpha' (Amount of Say)
            # Small epsilon to avoid division by zero
            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

            # Step 3: Update Weights
            # Increase weights for misclassified, decrease for correct
            predictions = clf.predict(X)
            
            # Formula: w_new = w_old * exp(-alpha * y * prediction)
            # If y == pred, y*pred is 1 -> exp(-alpha) -> weight decreases
            # If y != pred, y*pred is -1 -> exp(alpha) -> weight increases
            w *= np.exp(-clf.alpha * y * predictions)
            
            # Normalize weights so they sum to 1
            w /= np.sum(w)

            # Save the classifier
            self.clfs.append(clf)
            
            # Print progress every 10 classifiers
            if (clf_idx + 1) % 10 == 0:
                print(f"Learner {clf_idx+1}/{self.n_clf} trained. Alpha: {clf.alpha:.4f}")

    def predict(self, X):
        """
        Weighted vote of all weak learners.
        Returns 0 or 1 (converted from -1/1).
        """
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        
        # Sum all weighted predictions
        y_pred = np.sum(clf_preds, axis=0)
        
        # Sign function: if sum > 0 return 1, else return -1
        y_pred = np.sign(y_pred)
        
        # Convert back to 0/1 for compatibility with our dataset
        # -1 becomes 0, 1 stays 1
        return np.where(y_pred == -1, 0, 1)

# --- 3. Pipeline Execution ---
def run_training_pipeline(features_csv, model_output_path):
    print(f"Loading features from {features_csv}...")
    try:
        df = pd.read_csv(features_csv)
    except FileNotFoundError:
        print("Error: Features file not found. Run step 2 first.")
        return

    # Prepare Data
    # Drop non-numeric columns to get X matrix
    X = df.drop(columns=['filename', 'label']).values
    
    # Get y vector and convert 0 to -1 (AdaBoost math requires -1/1)
    y = df['label'].values
    y_ada = np.where(y == 0, -1, 1) 

    # Split into Train and Test sets (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y_ada, test_size=0.2, random_state=42)

    print(f"Training Data shape: {X_train.shape}")
    print(f"Testing Data shape: {X_test.shape}")

    # Initialize and Train Model
    # You can increase n_clf for better accuracy (but slower training)
    model = CustomAdaBoost(n_clf=30) 
    model.fit(X_train, y_train)

    # Evaluate
    print("\n--- Evaluation on Test Set ---")
    predictions = model.predict(X_test)
    
    # Convert y_test back to 0/1 for metric calculation
    y_test_original = np.where(y_test == -1, 0, 1)
    
    acc = accuracy_score(y_test_original, predictions)
    print(f"Model Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test_original, predictions))

    # Save the trained model to a file
    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {model_output_path}")

if __name__ == "__main__":
    # Paths
    INPUT_CSV = 'data/galaxy_features.csv'
    MODEL_PATH = 'data/adaboost_model.pkl'
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    run_training_pipeline(INPUT_CSV, MODEL_PATH)