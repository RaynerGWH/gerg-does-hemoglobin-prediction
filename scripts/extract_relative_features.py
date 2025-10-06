import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from pillow_heif import register_heif_opener

register_heif_opener()

def extract_relative_color_features(img_path):
    """Extract both absolute and relative color features"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        pixels = np.array(img)
        
        # Mean values per channel
        mean_r = pixels[:,:,0].mean()
        mean_g = pixels[:,:,1].mean()
        mean_b = pixels[:,:,2].mean()
        
        # 1. ABSOLUTE: Original percentages (lighting-dependent)
        total = pixels.sum()
        red_pct = (pixels[:,:,0].sum() / total) * 100
        green_pct = (pixels[:,:,1].sum() / total) * 100
        blue_pct = (pixels[:,:,2].sum() / total) * 100
        
        # 2. RELATIVE: RGB Ratios (lighting-independent)
        total_mean = mean_r + mean_g + mean_b + 1e-6
        r_ratio = mean_r / total_mean
        g_ratio = mean_g / total_mean
        b_ratio = mean_b / total_mean
        
        # 3. RELATIVE: Color ratios (important for anemia detection)
        rg_ratio = mean_r / (mean_g + 1e-6)  # Red-Green ratio
        rb_ratio = mean_r / (mean_b + 1e-6)  # Red-Blue ratio
        gb_ratio = mean_g / (mean_b + 1e-6)  # Green-Blue ratio
        
        # 4. RELATIVE: Color dominance (difference metrics)
        red_dominance = mean_r - mean_b  # Higher in healthy, lower in anemic
        
        # 5. Normalized values (0-1 scale)
        r_norm = mean_r / 255
        g_norm = mean_g / 255
        b_norm = mean_b / 255
        
        # Return comprehensive feature set
        return [
            # Absolute features (3)
            red_pct, green_pct, blue_pct,
            # Relative ratios (3)
            r_ratio, g_ratio, b_ratio,
            # Color ratios (3)
            rg_ratio, rb_ratio, gb_ratio,
            # Dominance (1)
            red_dominance,
            # Normalized (3)
            r_norm, g_norm, b_norm
        ]  # Total: 13 features
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# Load Genesis labels
df = pd.read_csv('data/starter/labels.csv')

features = []
valid_indices = []

print("Extracting relative color features...")
for idx, filepath in enumerate(df['filepath']):
    feat = extract_relative_color_features(filepath)
    if feat is not None:
        features.append(feat)
        valid_indices.append(idx)

# Save features and valid labels
X = np.array(features)
df_valid = df.iloc[valid_indices].reset_index(drop=True)

np.save('data/starter/relative_features.npy', X)
df_valid.to_csv('data/starter/labels_valid.csv', index=False)

print(f"\nExtracted {X.shape[1]} features for {len(X)}/{len(df)} lip images")
print(f"Feature breakdown:")
print(f"  - Absolute RGB%: 3 features")
print(f"  - Relative ratios: 3 features")
print(f"  - Color ratios: 3 features")
print(f"  - Dominance: 1 feature")
print(f"  - Normalized RGB: 3 features")
print(f"\nSaved to data/starter/relative_features.npy")