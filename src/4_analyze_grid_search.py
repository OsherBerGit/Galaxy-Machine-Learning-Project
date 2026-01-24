import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_training():
    print("--- Generating Visualization Plots ---")
    
    # 1. Load Data
    csv_path = 'data/grid_search_detailed_logs.csv'
    if not os.path.exists(csv_path):
        print("Error: CSV file not found. Please run grid_search first.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Filter: Focus on the longest run (200 iterations) to observe the full training evolution.
    # (Plotting all runs together would make the charts too cluttered).
    df_long = df[df['Configuration_Iters'] == 200]
    
    if df_long.empty:
        print("Warning: No run with 200 iterations found. Using all available data.")
        df_long = df

    # Create output directory
    os.makedirs('plots', exist_ok=True)
    sns.set_theme(style="whitegrid")

    # --- Plot 1: Alpha Decay (Voting Power) ---
    # Shows how the "authority" of the learners decreases over time.
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_long, x='Round_Number', y='Alpha', hue='Model_Type', linewidth=2.5)
    plt.title('Alpha Decay: How "Loud" is each learner?', fontsize=16)
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Alpha (Voting Power)', fontsize=12)
    plt.savefig('plots/alpha_decay.png')
    print("Saved: plots/alpha_decay.png")
    plt.close()

    # --- Plot 2: Weighted Error Evolution ---
    # Shows how difficult the remaining samples become as the algorithm progresses.
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_long, x='Round_Number', y='Error', hue='Model_Type', linewidth=2.5)
    plt.title('Weighted Error Evolution (Harder cases get higher weights)', fontsize=16)
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Weighted Error', fontsize=12)
    plt.savefig('plots/error_evolution.png')
    print("Saved: plots/error_evolution.png")
    plt.close()

    # --- Plot 3: Feature Importance (Decision Trees Only) ---
    # Which features did the trees select most often to split the data?
    
    # Filter for Decision Trees only
    tree_data = df_long[df_long['Model_Type'] == 'Decision Tree'].copy()
    
    # Helper function to extract feature name from rule string
    # Converts "Is std_blue <= -0.84" -> "std_blue"
    def extract_feature_name(rule_str):
        if isinstance(rule_str, str) and "Is " in rule_str:
            return rule_str.split(" ")[1] # Take the second word
        return None

    tree_data['Feature_Name'] = tree_data['Rule'].apply(extract_feature_name)
    
    # Count occurrences of each feature
    feature_counts = tree_data['Feature_Name'].value_counts().reset_index()
    feature_counts.columns = ['Feature', 'Count']

    # Plot Bar Chart
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_counts, y='Feature', x='Count', palette='viridis')
    plt.title('Feature Importance (Frequency of selection by Trees)', fontsize=16)
    plt.xlabel('Number of times selected as a splitting rule', fontsize=12)
    plt.ylabel('Feature Name', fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    print("Saved: plots/feature_importance.png")
    plt.close()

    print("\nDone! Check the 'plots' folder for images.")

if __name__ == "__main__":
    visualize_training()