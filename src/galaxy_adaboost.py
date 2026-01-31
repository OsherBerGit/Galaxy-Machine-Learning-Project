import numpy as np
from sklearn.tree import DecisionTreeClassifier

class GalaxyAdaBoost:
    """
    Manual Implementation of AdaBoost for Galaxy Classification.
    Logic based on: Error -> Alpha -> Weight Update.
    """
    def __init__(self, n_estimators=200):
        self.n_estimators = n_estimators
        self.models = []  # Stores the weak learners (Stumps)
        self.alphas = []  # Stores the voting power of each stump

    def fit(self, X, y):
        """
        Train the model using the AdaBoost algorithm manually.
        """
        n_samples = len(y)
        
        # Initialize weights: 1/N
        weights = np.ones(n_samples) / n_samples
        
        # Convert labels: 0/1 -> -1/1 (Critical for the math formula)
        y_signed = np.where(y == 0, -1, 1)

        for i in range(self.n_estimators):
            # 1. Weak Learner (Stump)
            stump = DecisionTreeClassifier(max_depth=1, random_state=42)
            stump.fit(X, y, sample_weight=weights)
            
            # 2. Predict on train set to find errors
            pred = stump.predict(X)
            pred_signed = np.where(pred == 0, -1, 1)

            # 3. Calculate Weighted Error
            is_error = (pred_signed != y_signed).astype(int)
            err = np.dot(weights, is_error)
            
            # Avoid division by zero
            err = max(1e-10, min(err, 1 - 1e-10))

            # 4. Calculate Alpha
            alpha = 0.5 * np.log((1 - err) / err)

            # 5. Update Weights
            # Formula: w_new = w_old * exp(-alpha * y_true * y_pred)
            weights *= np.exp(-alpha * y_signed * pred_signed)
            
            # Normalize
            weights /= np.sum(weights)

            # Save
            self.models.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        """
        Weighted Vote Prediction.
        """
        final_preds = np.zeros(X.shape[0])
        
        for alpha, model in zip(self.alphas, self.models):
            pred = model.predict(X)
            pred_signed = np.where(pred == 0, -1, 1)
            final_preds += alpha * pred_signed
        
        # If sum >= 0 -> Class 1 (Spiral), else Class 0 (Elliptical)
        return np.where(final_preds >= 0, 1, 0)