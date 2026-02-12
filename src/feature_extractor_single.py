"""
feature_extractor_single.py - Single Image Feature Extractor (Production)

This module extracts features from a SINGLE uploaded image for live prediction
in the Streamlit app. It mirrors the exact feature extraction logic used during
training (2_feature_extraction.py) to ensure consistency.

Features (11 total, in exact order):
    1. mean_red      - Average red channel intensity
    2. mean_green    - Average green channel intensity  
    3. mean_blue     - Average blue channel intensity
    4. std_red       - Standard deviation of red channel
    5. std_green     - Standard deviation of green channel
    6. std_blue      - Standard deviation of blue channel (KEY FEATURE)
    7. entropy       - Shannon entropy (texture complexity)
    8. area          - Contour area in pixels
    9. perimeter     - Contour perimeter in pixels
    10. circularity  - How circular the galaxy shape is
    11. eccentricity - How elongated the galaxy shape is

CRITICAL: The feature order MUST match the training data exactly!
"""
import cv2
import numpy as np
from scipy.stats import entropy

def extract_features_from_single_image(image_path):
    """
    Function that takes an image path and returns a numpy array of 9 features.
    The order is critical and must be identical to the order used during model training!
    """
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    # 2. Crop center (same as during training)
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2
    crop_size = 100
    img = img[center_y-crop_size:center_y+crop_size, center_x-crop_size:center_x+crop_size]

    # 3. Separate color channels (OpenCV uses BGR)
    blue_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    red_channel = img[:, :, 2]

    # 4. Convert to grayscale (for shape calculations)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Calculate features ---
    
    # Color features (6 features: Mean RGB + Std RGB)
    feat_mean_red = np.mean(red_channel)
    feat_mean_green = np.mean(green_channel)
    feat_mean_blue = np.mean(blue_channel)
    feat_std_red = np.std(red_channel)
    feat_std_green = np.std(green_channel)
    feat_std_blue = np.std(blue_channel)

    # Texture feature (1 feature)
    feat_entropy = entropy(gray_img.flatten(), base=2)

    # Shape features (4 features: area, perimeter, circularity, eccentricity)
    threshold_val = 50
    _, thresh = cv2.threshold(gray_img, threshold_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        feat_area = cv2.contourArea(largest_contour)
        feat_perimeter = cv2.arcLength(largest_contour, True)
        
        if feat_perimeter == 0:
            feat_circularity = 0
        else:
            feat_circularity = (4 * np.pi * feat_area) / (feat_perimeter ** 2)
            
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            feat_eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else 0
        else:
            feat_eccentricity = 0
    else:
        feat_area = 0
        feat_perimeter = 0
        feat_circularity = 0
        feat_eccentricity = 0

    # Build the final list (11 features in exact order as documented above)
    features_list = [
        feat_mean_red, feat_mean_green, feat_mean_blue,
        feat_std_red, feat_std_green, feat_std_blue,
        feat_entropy,
        feat_area, feat_perimeter, feat_circularity, feat_eccentricity
    ]
    
    # Convert to 2D array (rows, columns) as expected by the model
    return np.array([features_list])