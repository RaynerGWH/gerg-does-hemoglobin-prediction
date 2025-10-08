"""
Process Conjunctiva Data from Kaggle Dataset
Extracts and normalizes features to match lip image features
Run from project root: python scripts/03_process_conjunctiva_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict


def extract_conjunctiva_features(row: pd.Series) -> Dict[str, float]:
    """
    Convert Kaggle conjunctiva RGB percentages to enhanced feature set
    matching the format from enhanced feature extraction
    """
    
    # Get basic RGB percentages
    r_pct = row['%Red Pixel']
    g_pct = row['%Green pixel']
    b_pct = row['%Blue pixel']
    
    # Estimate mean RGB values (approximate scaling)
    # Original percentages are relative to total pixel sum
    scale = 255 / 3  # Approximate scaling factor
    mean_r = r_pct * scale / 100
    mean_g = g_pct * scale / 100
    mean_b = b_pct * scale / 100
    
    # Calculate relative features
    total_mean = mean_r + mean_g + mean_b + 1e-6
    r_ratio = mean_r / total_mean
    g_ratio = mean_g / total_mean
    b_ratio = mean_b / total_mean
    
    # Color ratios
    rg_ratio = mean_r / (mean_g + 1e-6)
    rb_ratio = mean_r / (mean_b + 1e-6)
    gb_ratio = mean_g / (mean_b + 1e-6)
    
    # Color dominance
    red_dominance = mean_r - mean_b
    redness_index = (mean_r - mean_g) / (mean_r + mean_g + 1e-6)
    
    # Normalized values
    r_norm = mean_r / 255
    g_norm = mean_g / 255
    b_norm = mean_b / 255
    
    # Image quality features (use defaults for conjunctiva data)
    # Since we don't have actual images, use neutral values
    brightness = (mean_r + mean_g + mean_b) / 3
    contrast = np.std([mean_r, mean_g, mean_b])  # Simple estimate
    blur_score = 100.0  # Assume sharp (neutral value)
    saturation = np.std([mean_r, mean_g, mean_b])  # Estimate from RGB variation
    lighting_uniformity = 20.0  # Assume moderate uniformity (neutral value)
    
    # Color distribution moments (simplified for conjunctiva)
    # Use mean values as approximations since we don't have full distributions
    r_mean = mean_r
    r_std = contrast  # Use overall contrast as proxy
    r_skew = 0.0  # Assume symmetric distribution
    
    g_mean = mean_g
    g_std = contrast
    g_skew = 0.0
    
    b_mean = mean_b
    b_std = contrast
    b_skew = 0.0
    
    features = {
        # Basic color features (13)
        'red_pct': r_pct,
        'green_pct': g_pct,
        'blue_pct': b_pct,
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
        
        # Extended color features (1)
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


def main():
    """Process conjunctiva data to match enhanced feature format"""
    
    print("=" * 70)
    print("PROCESSING CONJUNCTIVA DATA")
    print("=" * 70)
    
    # Load Kaggle anemia dataset
    kaggle_path = Path('../data/external/kaggle_anemia/anemia_dataset.csv')
    if not kaggle_path.exists():
        print(f"\n❌ Kaggle dataset not found: {kaggle_path}")
        print("Please ensure the anemia dataset is in data/external/kaggle_anemia/")
        return
    
    df = pd.read_csv(kaggle_path)
    print(f"\n📊 Loaded {len(df)} conjunctiva samples")
    
    # Extract features
    print("\n🔍 Extracting features from conjunctiva data...")
    features_list = []
    labels_list = []
    
    for idx, row in df.iterrows():
        features = extract_conjunctiva_features(row)
        features_list.append(features)
        labels_list.append(row['Hb'])
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} samples")
    
    # Convert to arrays
    features_df = pd.DataFrame(features_list)
    feature_array = features_df.values
    labels_array = np.array(labels_list)
    
    # Save outputs
    output_dir = Path('../data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    feature_path = output_dir / 'conjunctiva_features.npy'
    labels_path = output_dir / 'conjunctiva_labels.npy'
    
    np.save(feature_path, feature_array)
    np.save(labels_path, labels_array)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ CONJUNCTIVA DATA PROCESSING COMPLETE")
    print("=" * 70)
    print(f"  Samples processed: {len(features_list)}")
    print(f"  Features per sample: {feature_array.shape[1]}")
    print(f"\n📁 Output files:")
    print(f"  - Features: {feature_path}")
    print(f"  - Labels: {labels_path}")
    
    # Display statistics
    print(f"\n📊 HgB distribution:")
    print(f"  Min: {labels_array.min():.1f} g/dL")
    print(f"  Max: {labels_array.max():.1f} g/dL")
    print(f"  Mean: {labels_array.mean():.1f} g/dL")
    print(f"  Std: {labels_array.std():.1f} g/dL")
    
    print(f"\n📋 Feature statistics:")
    print(features_df.describe().T[['mean', 'std', 'min', 'max']].round(2).head(15))


if __name__ == '__main__':
    main()
