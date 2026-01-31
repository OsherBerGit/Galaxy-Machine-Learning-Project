import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def run_grid_search_with_logging():
    print("--- Starting Grid Search with Detailed Logging ---")

    # 1. Load Data
    try:
        df = pd.read_csv('data/galaxy_features.csv')
    except FileNotFoundError:
        print("Error: galaxy_features.csv not found.")
        return

    # Save feature names to map indices back to names
    feature_names = df.drop(columns=['filename', 'label']).columns.tolist()

    # Prepare X and y
    X = df.drop(columns=['filename', 'label']).values
    y = df['label'].values

    # Normalize Data (Critical for Perceptron/Logistic Regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 2. Define Models
    base_models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=1),
        'Perceptron': SGDClassifier(loss='perceptron', max_iter=1000, tol=1e-3),
        'Logistic Regression': LogisticRegression(solver='lbfgs')
    }

    n_estimators_list = [10, 50, 100, 200]
    
    # Storage for results
    detailed_logs = [] 
    summary_results = []
    
    # Variables for tracking the winner
    best_acc = 0
    best_config = ""

    # Print Table Header
    print(f"\n{'Base Model':<30} | {'Iterations':<10} | {'Accuracy':<10} | {'Status'}")
    print("-" * 75)

    # 3. Grid Search Loop
    for model_name, model_obj in base_models.items():
        for n_est in n_estimators_list:
            
            try:
                # Initialize AdaBoost
                ada = AdaBoostClassifier(
                    estimator=model_obj,
                    n_estimators=n_est,
                    random_state=42
                )
                
                # Train
                ada.fit(X_train, y_train)
                
                # Evaluate
                preds = ada.predict(X_test)
                acc = accuracy_score(y_test, preds)
                summary_results.append({'Model': model_name, 'Total_Iters': n_est, 'Final_Accuracy': acc})

                # Check if this is the new best model
                is_best = ""
                if acc > best_acc:
                    best_acc = acc
                    best_config = f"{model_name} with {n_est} iters"
                    is_best = "(*) NEW BEST"
                
                # Print row in table format
                print(f"{model_name:<30} | {n_est:<10} | {acc:.4f}     | {is_best}")

                # --- History Extraction Logic ---
                for i, estimator in enumerate(ada.estimators_):
                    error = ada.estimator_errors_[i]
                    alpha = ada.estimator_weights_[i]
                    
                    rule_desc = "Linear Combination (Equation)"
                    
                    if hasattr(estimator, 'tree_'):
                        feat_idx = estimator.tree_.feature[0]
                        threshold = estimator.tree_.threshold[0]
                        
                        if feat_idx >= 0:
                            feat_name = feature_names[feat_idx]
                            rule_desc = f"Is {feat_name} <= {threshold:.4f} ?"
                        else:
                            rule_desc = "No Split (Leaf Node)"

                    detailed_logs.append({
                        'Model_Type': model_name,
                        'Configuration_Iters': n_est,
                        'Round_Number': i + 1,
                        'Error': error,
                        'Alpha': alpha,
                        'Rule': rule_desc
                    })

            except Exception as e:
                print(f"{model_name:<30} | {n_est:<10} | FAILED     | {str(e)}")

    # 4. Summary & Save
    print("-" * 75)
    print(f"🏆 WINNER: {best_config}")
    print(f"🏆 ACCURACY: {best_acc:.4f}")
    
    # Save files
    df_details = pd.DataFrame(detailed_logs)
    df_details.to_csv('data/grid_search_detailed_logs.csv', index=False)
    
    df_summary = pd.DataFrame(summary_results)
    df_summary.to_csv('data/grid_search_summary.csv', index=False)
    
    print("\n✅ Files saved:")
    print("   1. data/grid_search_summary.csv")
    print("   2. data/grid_search_detailed_logs.csv")

if __name__ == "__main__":
    run_grid_search_with_logging()