import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_single_feature():
    print("--- Single Feature Test: std_blue ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv('data/galaxy_features.csv')
    except FileNotFoundError:
        print("Error: galaxy_features.csv not found.")
        return

    # 2. Select ONLY 'std_blue'
    # We focus solely on this feature to test its individual power
    feature_name = 'std_blue'
    X = df[[feature_name]].values
    y = df['label'].values

    # 3. Split Data (Same random_state as before for fair comparison)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train a simple Decision Stump (Depth=1)
    # A "Stump" makes exactly one cut (Threshold check)
    clf = DecisionTreeClassifier(max_depth=1, random_state=42)
    clf.fit(X_train, y_train)

    # 5. Evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    # Extract the threshold (The specific number used for the cut)
    threshold = clf.tree_.threshold[0]
    
    print(f"\nResults for feature '{feature_name}':")
    print(f"------------------------------------------------")
    print(f"Optimal Threshold Found: {threshold:.4f}")
    print(f"Rule: If {feature_name} <= {threshold:.4f} -> Classify as Elliptical (0)")
    print(f"------------------------------------------------")
    print(f"Accuracy using ONLY this feature: {acc*100:.2f}%")

if __name__ == "__main__":
    test_single_feature()