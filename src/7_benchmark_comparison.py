import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def run_advanced_comparison():
    print("--- Advanced Model Comparison: Clash of the Titans ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv('data/galaxy_features.csv')
    except FileNotFoundError:
        print("Error: galaxy_features.csv not found.")
        return

    X = df.drop(columns=['filename', 'label']).values
    y = df['label'].values

    # Scaling is crucial for SVM (and helpful for others)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 2. Define the Contenders
    models = {
        # המודל שלך (משתמשים בספרייה רק לצורך ההשוואה ההוגנת כאן)
        "AdaBoost (Baseline)": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=200, 
            random_state=42
        ),
        
        # השיפור של Bagging: כל עץ רואה רק חלק מהפיצ'רים
        "Random Forest": RandomForestClassifier(
            n_estimators=200, 
            max_depth=None,     # Deep trees
            max_features='sqrt',# Random feature selection per split
            random_state=42
        ),
        
        # השיפור המודרני של Boosting: לומד את השאריות (Residuals)
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,        # Gradient Boosting uses shallow trees (depth 3-5)
            random_state=42
        ),
        
        # מודל גיאומטרי שונה לחלוטין (לא מבוסס עצים)
        "SVM (RBF Kernel)": SVC(
            kernel='rbf',       # Radial Basis Function
            C=1.0, 
            gamma='scale',
            random_state=42
        )
    }

    results = []
    
    print(f"\n{'Model Name':<25} | {'Accuracy':<10} | {'Remarks'}")
    print("-" * 65)

    # 3. Train and Evaluate
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        results.append({'Model': name, 'Accuracy': acc})
        
        remark = ""
        if name == "AdaBoost (Baseline)": remark = "Your Project Algo"
        if acc > 0.90: remark = "Excellent!"
        
        print(f"{name:<25} | {acc:.4f}     | {remark}")

    # 4. Visualization
    print("-" * 65)
    print("Generating comparison chart...")
    
    results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
    
    plt.figure(figsize=(12, 6))
    
    # Create Bar Plot
    ax = sns.barplot(data=results_df, x='Accuracy', y='Model', palette='magma')
    
    # Customize layout
    plt.xlim(0.8, 1.0) # Focus on the 80%-100% range
    plt.title('Final Showdown: Which Algorithm Rules the Galaxy?', fontsize=16)
    plt.xlabel('Accuracy Score', fontsize=12)
    plt.ylabel('Algorithm', fontsize=12)
    
    # Add numbers to bars
    for i, v in enumerate(results_df['Accuracy']):
        ax.text(v + 0.005, i, f"{v*100:.2f}%", color='black', fontweight='bold', va='center')

    plt.tight_layout()
    output_path = 'plots/advanced_comparison.png'
    plt.savefig(output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    run_advanced_comparison()