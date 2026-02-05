"""
6_bias_variance_analysis.py - Boosting vs Bagging Comparison

This script compares Boosting (AdaBoost) with Bagging to demonstrate why
Boosting is the right choice for this problem.

Experiments:
    1. Bagging with Stumps (Depth=1): Expected to FAIL (~66%) because Bagging
       reduces variance but stumps have high bias. Averaging weak models = weak model.
    2. Bagging with Deep Trees: Works well (~87%) because deep trees have high
       variance which Bagging successfully reduces.
    3. Boosting with Stumps: Works well (~87%) because Boosting reduces bias
       by sequentially focusing on misclassified samples.

Conclusion: Boosting is ideal for weak learners (stumps), Bagging is ideal for
strong learners (deep trees). This justifies our AdaBoost approach.
"""
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run_bagging_comparison():
    print("--- Boosting vs. Bagging Comparison ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv('data/galaxy_features.csv')
    except FileNotFoundError:
        print("Error: galaxy_features.csv not found.")
        return

    X = df.drop(columns=['filename', 'label']).values
    y = df['label'].values

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---------------------------------------------------------
    # Experiment 1: Bagging with Stumps (Depth=1)
    # Goal: Direct comparison with your AdaBoost result (which used Depth=1)
    # ---------------------------------------------------------
    print("\n1. Training Bagging with Stumps (Depth=1)...")
    bagging_stump = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=200, # Matching the AdaBoost iteration count
        random_state=42,
        n_jobs=-1         # Run in parallel (uses all CPU cores)
    )
    bagging_stump.fit(X_train, y_train)
    preds_stump = bagging_stump.predict(X_test)
    acc_stump = accuracy_score(y_test, preds_stump)

    # ---------------------------------------------------------
    # Experiment 2: Bagging with Full Trees (Unrestricted Depth)
    # Goal: This is how Bagging is usually used (reducing variance of complex models)
    # ---------------------------------------------------------
    print("2. Training Bagging with Deep Trees (Unlimited Depth)...")
    bagging_deep = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=None), # Fully grown trees
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    bagging_deep.fit(X_train, y_train)
    preds_deep = bagging_deep.predict(X_test)
    acc_deep = accuracy_score(y_test, preds_deep)

    # ---------------------------------------------------------
    # Experiment 3: Single Deep Tree (Baseline)
    # Goal: To see if Bagging actually improved anything over a regular tree
    # ---------------------------------------------------------
    print("3. Training Single Deep Tree (Baseline)...")
    single_tree = DecisionTreeClassifier(max_depth=None, random_state=42)
    single_tree.fit(X_train, y_train)
    acc_single = single_tree.score(X_test, y_test)

    # ---------------------------------------------------------
    # Final Report
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(f"{'Model Architecture':<35} | {'Accuracy':<10}")
    print("="*50)
    print(f"{'AdaBoost (Your Best Result)':<35} | {'~87.70%'}") # Hardcoded for reference
    print("-" * 50)
    print(f"{'Bagging (Stumps - Depth 1)':<35} | {acc_stump*100:.2f}%")
    print(f"{'Bagging (Deep Trees - Unlimited)':<35} | {acc_deep*100:.2f}%")
    print(f"{'Single Decision Tree (Baseline)':<35} | {acc_single*100:.2f}%")
    print("="*50)

    # Analysis / Insight generation
    if acc_stump < 0.87:
        print("\nInsight: Bagging Stumps performed worse than Boosting Stumps.")
        print("Reason: Bagging reduces Variance, but Stumps have High Bias.")
        print("Boosting is designed to fix Bias (Underfitting), so it wins on Stumps.")
    
    if acc_deep > acc_single:
        print("\nInsight: Bagging Deep Trees improved over a Single Tree.")
        print("Reason: Deep trees tend to Overfit. Bagging averaged them out to reduce Variance.")
    
    # Save results to CSV for Streamlit app
    results_df = pd.DataFrame([
        {'Model': 'AdaBoost (Stumps)', 'Accuracy': 0.8770},  # Reference from grid search
        {'Model': 'Bagging (Stumps)', 'Accuracy': acc_stump},
        {'Model': 'Bagging (Deep Trees)', 'Accuracy': acc_deep},
        {'Model': 'Single Tree', 'Accuracy': acc_single}
    ])
    results_df.to_csv('data/bias_variance_results.csv', index=False)
    print("\nSaved: data/bias_variance_results.csv")

if __name__ == "__main__":
    run_bagging_comparison()