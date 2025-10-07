"""
Enhanced Feature Extraction with Image Quality Metrics
Extracts comprehensive features including lighting, blur, and color information
Run from project root: python scripts/01_extract_enhanced_features.py
"""

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from pathlib import Path
from pillow_heif import register_heif_opener
from typing import Dict, List, Tuple

register_heif_opener()


def calculate_brightness(img_array: np.ndarray) -> float:
    """Calculate average brightness (0-255 scale)"""
    return np.mean(img_array)


def calculate_contrast(img_array: np.ndarray) -> float:
    """Calculate RMS contrast"""
    return np.std(img_array)


def detect_blur(img_array: np.ndarray) -> float:
    """Detect blur using Laplacian variance method
    Higher values = sharper image, Lower values = more blur
    """
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


def calculate_saturation(img_array: np.ndarray) -> float:
    """Calculate average saturation in HSV space"""
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].mean()
    return saturation


def calculate_lighting_uniformity(img_array: np.ndarray) -> float:
    """Calculate lighting uniformity (lower std = more uniform)"""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    return np.std(gray)


def extract_color_moments(channel: np.ndarray) -> Tuple[float, float, float]:
    """Extract first 3 statistical moments from a color channel"""
    mean = np.mean(channel)
    std = np.std(channel)
    skewness = np.mean(((channel - mean) / (std + 1e-6)) ** 3)
    return mean, std, skewness


def extract_enhanced_features(img_path: Path) -> Dict[str, float]:
    """
    Extract comprehensive features including:
    - Color features (RGB statistics and ratios)
    - Image quality metrics (blur, brightness, contrast)
    - Color distribution (moments)
    
    Returns dict with all features
    """
    try:
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        # Separate channels
        r_channel = img_array[:, :, 0]
        g_channel = img_array[:, :, 1]
        b_channel = img_array[:, :, 2]
        
        # ===== COLOR FEATURES =====
        # Mean values per channel
        mean_r = r_channel.mean()
        mean_g = g_channel.mean()
        mean_b = b_channel.mean()
        
        # RGB Percentages (absolute)
        total = img_array.sum()
        red_pct = (r_channel.sum() / total) * 100
        green_pct = (g_channel.sum() / total) * 100
        blue_pct = (b_channel.sum() / total) * 100
        
        # Relative RGB Ratios (lighting-independent)
        total_mean = mean_r + mean_g + mean_b + 1e-6
        r_ratio = mean_r / total_mean
        g_ratio = mean_g / total_mean
        b_ratio = mean_b / total_mean
        
        # Color ratios (important for anemia detection)
        rg_ratio = mean_r / (mean_g + 1e-6)
        rb_ratio = mean_r / (mean_b + 1e-6)
        gb_ratio = mean_g / (mean_b + 1e-6)
        
        # Color dominance
        red_dominance = mean_r - mean_b
        redness_index = (mean_r - mean_g) / (mean_r + mean_g + 1e-6)
        
        # Normalized values (0-1 scale)
        r_norm = mean_r / 255
        g_norm = mean_g / 255
        b_norm = mean_b / 255
        
        # ===== IMAGE QUALITY FEATURES =====
        brightness = calculate_brightness(img_array)
        contrast = calculate_contrast(img_array)
        blur_score = detect_blur(img_array)
        saturation = calculate_saturation(img_array)
        lighting_uniformity = calculate_lighting_uniformity(img_array)
        
        # ===== COLOR DISTRIBUTION FEATURES =====
        # Statistical moments for each channel
        r_mean, r_std, r_skew = extract_color_moments(r_channel)
        g_mean, g_std, g_skew = extract_color_moments(g_channel)
        b_mean, b_std, b_skew = extract_color_moments(b_channel)
        
        # Compile all features
        features = {
            # Basic color features (13)
            'red_pct': red_pct,
            'green_pct': green_pct,
            'blue_pct': blue_pct,
            'r_ratio': r_ratio,
            'g_ratio': g_ratio,
            'b_ratio': b_ratio,
            'rg_ratio': rg_ratio,
            'rb_ratio': rb_ratio,
            'gb_ratio': gb_ratio,
            'red_dominance': red_dominance,
            'r_norm': r_norm,
            'g_norm': g_norm,
            'b_norm': b_norm,
            
            # Image quality features (5)
            'brightness': brightness,
            'contrast': contrast,
            'blur_score': blur_score,
            'saturation': saturation,
            'lighting_uniformity': lighting_uniformity,
            
            # Extended color features (2)
            'redness_index': redness_index,
            
            # Color distribution moments (9)
            'r_mean': r_mean,
            'r_std': r_std,
            'r_skew': r_skew,
            'g_mean': g_mean,
            'g_std': g_std,
            'g_skew': g_skew,
            'b_mean': b_mean,
            'b_std': b_std,
            'b_skew': b_skew,
        }
        
        return features
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def main():
    """Extract enhanced features from all lip images"""
    
    print("=" * 70)
    print("ENHANCED FEATURE EXTRACTION")
    print("=" * 70)
    
    # Load labels
    labels_path = Path('../data/starter/labels.csv')
    if not labels_path.exists():
        print(f"\n❌ Labels file not found: {labels_path}")
        print("Please run create_labels_csv.py first")
        return
    
    df = pd.read_csv(labels_path)
    print(f"\n📊 Found {len(df)} labeled images")
    
    # Extract features
    features_list = []
    valid_indices = []
    failed_images = []
    
    print("\n🔍 Extracting features...")
    for idx, row in df.iterrows():
        img_path = Path(row['filepath'])
        if not img_path.exists():
            # Try with ../ prefix
            img_path = Path('..') / img_path
        
        features = extract_enhanced_features(img_path)
        if features is not None:
            features_list.append(features)
            valid_indices.append(idx)
            print(f"  ✓ [{idx+1}/{len(df)}] {img_path.name}")
        else:
            failed_images.append((idx, img_path.name))
            print(f"  ✗ [{idx+1}/{len(df)}] {img_path.name} - FAILED")
    
    # Convert to arrays
    features_df = pd.DataFrame(features_list)
    feature_array = features_df.values
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    
    # Save outputs
    output_dir = Path('../data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    feature_path = output_dir / 'enhanced_features.npy'
    labels_path = output_dir / 'labels_valid.csv'
    feature_names_path = output_dir / 'feature_names.txt'
    
    np.save(feature_path, feature_array)
    df_valid.to_csv(labels_path, index=False)
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(features_df.columns))
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ FEATURE EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"  Successfully processed: {len(features_list)}/{len(df)} images")
    print(f"  Total features extracted: {feature_array.shape[1]}")
    
    # Warn about mismatches
    if len(features_list) != len(df):
        print("\n" + "⚠️  " * 20)
        print(f"⚠️  IMPORTANT NOTE:")
        print("=" * 70)
        print(f"You have {len(df)} images but lip_rgb_features.npy has {len(features_list)} samples!")
        print(f"This mismatch suggests:")
        print(f"1. Some images failed during feature extraction")
        print(f"2. Or features were extracted from different images")
        print(f"\nFailed images ({len(failed_images)}):")
        for idx, name in failed_images:
            print(f"  ❌ [{idx+1}] {name}")
        print(f"\nRecommendation: Fix or remove failed images, then re-run this script")
        print("=" * 70)
    print(f"\n📁 Output files:")
    print(f"  - Features: {feature_path}")
    print(f"  - Labels: {labels_path}")
    print(f"  - Feature names: {feature_names_path}")
    
    print(f"\n📋 Feature breakdown:")
    print(f"  - Basic color features: 13")
    print(f"  - Image quality features: 5")
    print(f"  - Extended color features: 1")
    print(f"  - Color distribution moments: 9")
    print(f"  - TOTAL: {feature_array.shape[1]} features")
    
    # Display feature statistics
    print(f"\n📊 Feature statistics:")
    print(features_df.describe().T[['mean', 'std', 'min', 'max']].round(2))


if __name__ == '__main__':
    main()
