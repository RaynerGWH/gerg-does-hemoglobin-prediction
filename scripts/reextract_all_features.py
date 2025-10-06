import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from pillow_heif import register_heif_opener

register_heif_opener()

def extract_relative_color_features(img_path):
    """Extract comprehensive color features"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        pixels = np.array(img)
        
        # Mean values per channel
        mean_r = pixels[:,:,0].mean()
        mean_g = pixels[:,:,1].mean()
        mean_b = pixels[:,:,2].mean()
        
        # 1. ABSOLUTE: Original percentages
        total = pixels.sum()
        red_pct = (pixels[:,:,0].sum() / total) * 100
        green_pct = (pixels[:,:,1].sum() / total) * 100
        blue_pct = (pixels[:,:,2].sum() / total) * 100
        
        # 2. RELATIVE: RGB Ratios (lighting-independent)
        total_mean = mean_r + mean_g + mean_b + 1e-6
        r_ratio = mean_r / total_mean
        g_ratio = mean_g / total_mean
        b_ratio = mean_b / total_mean
        
        # 3. RELATIVE: Color ratios
        rg_ratio = mean_r / (mean_g + 1e-6)
        rb_ratio = mean_r / (mean_b + 1e-6)
        gb_ratio = mean_g / (mean_b + 1e-6)
        
        # 4. RELATIVE: Color dominance
        red_dominance = mean_r - mean_b
        
        # 5. Normalized values
        r_norm = mean_r / 255
        g_norm = mean_g / 255
        b_norm = mean_b / 255
        
        return [
            red_pct, green_pct, blue_pct,
            r_ratio, g_ratio, b_ratio,
            rg_ratio, rb_ratio, gb_ratio,
            red_dominance,
            r_norm, g_norm, b_norm
        ]
    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")
        return None

print("=" * 60)
print("RE-EXTRACTING FEATURES FROM ALL IMAGES")
print("=" * 60)

# Load labels CSV - use cropped version if available
labels_options = [
    Path('data/starter/labels_cropped.csv'),  # Prefer cropped
    Path('data/starter/labels_valid.csv'),
    Path('data/starter/labels.csv')
]

labels_path = None
for path in labels_options:
    if path.exists():
        labels_path = path
        break

if labels_path is None:
    print("❌ No labels CSV found! Run create_labels_csv.py first")
    exit(1)

df = pd.read_csv(labels_path)
print(f"\n✓ Loaded {len(df)} labels from labels.csv")

# Extract features for each image
features = []
valid_indices = []

print(f"\nExtracting features:")
for idx, row in df.iterrows():
    filepath = row['filepath']
    hgb = row['hgb']
    
    feat = extract_relative_color_features(filepath)
    if feat is not None:
        features.append(feat)
        valid_indices.append(idx)
        print(f"  ✓ {idx+1:2d}. {row['filename']:45s} HgB={hgb:.1f}")
    else:
        print(f"  ❌ {idx+1:2d}. {row['filename']:45s} FAILED")

# Save features
X = np.array(features)
df_valid = df.iloc[valid_indices].reset_index(drop=True)

# Save both old format (3 features) and new format (13 features)
np.save('data/starter/lip_rgb_features.npy', X[:, :3])  # Old format (backward compat)
np.save('data/starter/relative_features.npy', X)         # New format (all 13 features)
df_valid.to_csv('data/starter/labels_valid.csv', index=False)

print(f"\n" + "=" * 60)
print("SAVED FILES:")
print("=" * 60)
print(f"✅ lip_rgb_features.npy: {X.shape[0]} samples × 3 features (RGB only)")
print(f"✅ relative_features.npy: {X.shape[0]} samples × {X.shape[1]} features (all)")
print(f"✅ labels_valid.csv: {len(df_valid)} samples with HgB values")

print(f"\n" + "=" * 60)
print("FEATURE BREAKDOWN:")
print("=" * 60)
print("  Absolute RGB%: 3 features")
print("  Relative ratios: 3 features")
print("  Color ratios: 3 features")
print("  Dominance: 1 feature")
print("  Normalized RGB: 3 features")
print(f"  TOTAL: {X.shape[1]} features")

print(f"\n" + "=" * 60)
print("NEXT STEP:")
print("=" * 60)
print("Run: python train_combined_model_relative.py")