"""
1_prepare_data.py - Data Preparation Pipeline

This script processes the raw Galaxy Zoo dataset to create a clean, balanced dataset
for binary classification (Spiral vs Elliptical galaxies).

Steps:
    1. Load the raw training_solutions_rev1.csv from Kaggle
    2. Filter galaxies using confidence thresholds (Class1.1 > 0.8 for Elliptical, etc.)
    3. Balance the dataset using undersampling to avoid class imbalance
    4. Save the final dataset with filenames and labels to galaxy_dataset_final.csv

Output: data/galaxy_dataset_final.csv
"""
import pandas as pd

def load_data(input_csv):
    """
    Step 1: Load raw data from CSV.
    """
    print(f"--- Loading data from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
        return df
    except FileNotFoundError:
        print("Error: File not found. Please check the filename and path.")
        return None

def filter_galaxies(df):
    """
    Step 2: Filter data based on Galaxy Zoo Decision Tree logic.
    Returns two separate DataFrames: elliptical and spiral.
    """
    print("--- Filtering data based on decision tree logic...")

    # --- Group A: Elliptical Galaxies ---
    # Logic: Question 1 = Smooth (Class1.1) with high confidence
    elliptical_mask = df['Class1.1'] > 0.8
    
    # --- Group B: Spiral Galaxies ---
    # Complex Logic based on the decision tree path:
    # 1. Q1: Has features/disk (Class1.2)
    # 2. AND Q2: Is NOT edge-on (Face-on) (Class2.2) -> Critical for seeing arms
    # 3. AND Q4: Has clear spiral arm pattern (Class4.1)
    
    spiral_mask = (
        (df['Class1.2'] > 0.5) & 
        (df['Class2.2'] > 0.5) & 
        (df['Class4.1'] > 0.8)
    )

    # Create filtered DataFrames
    df_elliptical = df[elliptical_mask].copy()
    df_elliptical['label'] = 0  # Label 0 for Elliptical
    df_elliptical['type'] = 'Elliptical'

    df_spiral = df[spiral_mask].copy()
    df_spiral['label'] = 1      # Label 1 for Spiral
    df_spiral['type'] = 'Spiral'
    
    print(f"Found {len(df_elliptical)} Ellipticals")
    print(f"Found {len(df_spiral)} Spirals")

    return df_elliptical, df_spiral

def balance_datasets(df_elliptical, df_spiral):
    """
    Step 3: Balance the dataset using Undersampling.
    Ensures both classes have the exact same number of images.
    """
    count_ellip = len(df_elliptical)
    count_spiral = len(df_spiral)

    if count_spiral == 0 or count_ellip == 0:
        print("Critical Error: One class is empty! Check thresholds.")
        return None

    # Find the minimum count to balance towards
    min_samples = min(count_ellip, count_spiral)
    print(f"--- Balancing dataset to {min_samples} images per class...")

    # Random sampling to match the minority class size
    df_ellip_balanced = df_elliptical.sample(n=min_samples, random_state=42)
    df_spiral_balanced = df_spiral.sample(n=min_samples, random_state=42)

    # Concatenate both classes
    balanced_df = pd.concat([df_ellip_balanced, df_spiral_balanced])
    return balanced_df

def save_final_dataset(df, output_csv):
    """
    Step 4: Final processing and saving to CSV.
    Clean up columns to avoid data leakage.
    """
    print("--- Finalizing and saving...")
    
    # Add .jpg extension to match actual filenames
    df['filename'] = df['GalaxyID'].astype(str) + ".jpg"
    
    # Shuffle the dataset so classes are mixed
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Keep only relevant columns for training (Avoiding data leakage!)
    columns = ['filename', 'label', 'type']
    df[columns].to_csv(output_csv, index=False)
    
    print(f"Done! Saved to {output_csv}")
    print(f"Total images: {len(df)}")
    print("Final Distribution:")
    print(df['type'].value_counts())

# --- Pipeline Manager ---
def process_galaxy_pipeline(input_file, output_file):
    # 1. Load
    df = load_data(input_file)
    if df is None: return

    # 2. Filter
    df_ellip, df_spiral = filter_galaxies(df)

    # 3. Balance
    balanced_df = balance_datasets(df_ellip, df_spiral)
    if balanced_df is None: return

    # 4. Save
    save_final_dataset(balanced_df, output_file)

# --- Execution ---
if __name__ == "__main__":
    # Ensure raw CSV file is in the same folder
    input_csv = 'data/training_solutions_rev1.csv'
    output_csv = 'data/galaxy_dataset_final.csv'
    
    process_galaxy_pipeline(input_csv, output_csv)