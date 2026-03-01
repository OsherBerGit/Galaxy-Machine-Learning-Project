"""
feature_extractor_single.py - Single Image Feature Extractor (Production)

This module extracts features from a SINGLE uploaded image for live prediction
in the Streamlit app. It delegates directly to process_single_image() from
2_feature_extraction.py to guarantee 100% consistency with training.

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
import os
import sys
import numpy as np

# Ensure the src directory is on the path so the import works regardless of
# where the caller (e.g. the Streamlit app) is launched from.
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module

# 2_feature_extraction starts with a digit, so use importlib
_fe = import_module("2_feature_extraction")
process_single_image = _fe.process_single_image

# The exact feature order must match the training CSV columns
_FEATURE_ORDER = [
    'mean_red', 'mean_green', 'mean_blue',
    'std_red', 'std_green', 'std_blue',
    'entropy',
    'area', 'perimeter', 'circularity', 'eccentricity',
]

def extract_features_from_single_image(image_path):
    """
    Takes an image path and returns a (1, 11) numpy array of features.
    Delegates to process_single_image() from 2_feature_extraction.py so that
    the exact same logic (crop, entropy method, eccentricity formula, etc.)
    is used both during training and at prediction time.
    """
    features = process_single_image(image_path)
    if features is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert the dict to a 2-D array in the required column order
    return np.array([[features[k] for k in _FEATURE_ORDER]])