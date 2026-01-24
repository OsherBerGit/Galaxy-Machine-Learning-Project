import pandas as pd
import numpy as np
import cv2
import os
from skimage.measure import shannon_entropy

# --- Configuration Constants ---
CROP_SIZE = 100  # Radius of the center crop (result size will be 200x200)
THRESHOLD_VAL = 50  # Binary threshold for shape detection

def load_image(image_path):
    """
    Loads an image from disk using OpenCV.
    Returns None if loading fails.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    return img

def crop_center(img):
    """
    Crops the center of the image to remove black borders.
    Returns the cropped image.
    """
    h, w, _ = img.shape
    center_x, center_y = w // 2, h // 2
    
    # Slicing the numpy array to get the center
    img_crop = img[center_y-CROP_SIZE:center_y+CROP_SIZE, center_x-CROP_SIZE:center_x+CROP_SIZE]
    return img_crop

def extract_color_features(img):
    """
    Calculates statistical color features (Mean, Std Dev) for R, G, B channels.
    Spiral galaxies tend to be bluer, Elliptical galaxies tend to be redder.
    """
    # OpenCV loads images as BGR
    b_channel, g_channel, r_channel = cv2.split(img)
    
    features = {
        'mean_blue': np.mean(b_channel),
        'mean_green': np.mean(g_channel),
        'mean_red': np.mean(r_channel),
        'std_blue': np.std(b_channel),
        'std_red': np.std(r_channel)
    }
    return features

def extract_texture_features(img_gray):
    """
    Calculates texture complexity using Shannon Entropy.
    Elliptical galaxies are smooth (low entropy).
    Spiral galaxies have arms/structure (high entropy).
    """
    features = {}
    features['entropy'] = shannon_entropy(img_gray)
    return features

def extract_shape_features(img_gray):
    """
    Calculates geometric features based on contours:
    - Circularity: How close the shape is to a perfect circle.
    - Eccentricity: How elongated the shape is (fitting an ellipse).
    """
    features = {
        'circularity': 0.0,
        'eccentricity': 0.0,
        'area': 0.0
    }
    
    # Thresholding to create a binary mask (Black & White)
    _, binary = cv2.threshold(img_gray, THRESHOLD_VAL, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return features

    # Assume the largest contour is the galaxy
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    features['area'] = area

    # 1. Calculate Circularity
    if perimeter > 0:
        # Formula: 4 * pi * Area / Perimeter^2 (1.0 is perfect circle)
        features['circularity'] = (4 * np.pi * area) / (perimeter ** 2)
        
    # 2. Calculate Eccentricity (Fit Ellipse)
    if len(largest_contour) >= 5: # FitEllipse needs at least 5 points
        try:
            (x, y), (MA, ma), angle = cv2.fitEllipse(largest_contour)
            # MA = Minor Axis, ma = Major Axis
            if ma > 0:
                # Eccentricity formula: sqrt(1 - (minor/major)^2)
                features['eccentricity'] = np.sqrt(1 - (MA / ma)**2)
        except Exception:
            pass # Keep default 0.0 if fitting fails

    return features

def process_single_image(image_path):
    """
    Orchestrates the feature extraction for a single image file.
    Combines Color, Texture, and Shape features into one dictionary.
    """
    img = load_image(image_path)
    if img is None:
        return None
    
    # Preprocessing
    img_crop = crop_center(img)
    img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    
    # Feature Extraction
    color_feats = extract_color_features(img_crop)
    texture_feats = extract_texture_features(img_gray)
    shape_feats = extract_shape_features(img_gray)
    
    # Merge all dictionaries (Python 3.5+)
    all_features = {**color_feats, **texture_feats, **shape_feats}
    
    return all_features

def run_feature_extraction_pipeline(input_csv, images_dir, output_csv):
    """
    Main pipeline:
    1. Reads the file list from CSV.
    2. Loops through images and extracts features.
    3. Saves the resulting feature table to a new CSV.
    """
    print(f"--- Loading dataset list from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print("Error: Input CSV not found.")
        return

    print(f"--- Starting feature extraction on {len(df)} images...")
    
    extracted_data = []
    
    for index, row in df.iterrows():
        filename = row['filename']
        label = row['label'] # Preserving the label (0 or 1)
        
        full_path = os.path.join(images_dir, filename)
        
        # Extract features
        features = process_single_image(full_path)
        
        if features:
            features['filename'] = filename
            features['label'] = label
            extracted_data.append(features)
        
        # Progress log every 500 images
        if (index + 1) % 500 == 0:
            print(f"Processed {index + 1}/{len(df)}...")

    # Save to CSV
    if extracted_data:
        features_df = pd.DataFrame(extracted_data)
        
        # Reorder columns to put filename/label first (for readability)
        cols = ['filename', 'label'] + [c for c in features_df.columns if c not in ['filename', 'label']]
        features_df = features_df[cols]
        
        features_df.to_csv(output_csv, index=False)
        print(f"\n--- Success! Features saved to {output_csv}")
        print(f"Final shape: {features_df.shape}")
        print("First 5 rows:")
        print(features_df.head())
    else:
        print("Error: No features were extracted. Check image paths.")

# --- Execution ---
if __name__ == "__main__":
    # Define paths relative to the project root
    INPUT_LIST_CSV = 'data/galaxy_dataset_final.csv'
    IMAGES_FOLDER = 'data/images_training_rev1' # Or 'data/images_final_dataset' if you copied them
    OUTPUT_FEATURES_CSV = 'data/galaxy_features.csv'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FEATURES_CSV), exist_ok=True)

    run_feature_extraction_pipeline(INPUT_LIST_CSV, IMAGES_FOLDER, OUTPUT_FEATURES_CSV)