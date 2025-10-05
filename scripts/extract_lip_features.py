import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from pillow_heif import register_heif_opener

# Register HEIF opener with Pillow
register_heif_opener()

def extract_rgb_percentages(img_path):
    """Extract RGB percentages like Kaggle dataset"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        pixels = np.array(img)
        
        # Calculate percentages
        total = pixels.sum()
        red_pct = (pixels[:,:,0].sum() / total) * 100
        green_pct = (pixels[:,:,1].sum() / total) * 100
        blue_pct = (pixels[:,:,2].sum() / total) * 100
        
        return [red_pct, green_pct, blue_pct]
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# Load Genesis labels
df = pd.read_csv('data/starter/labels.csv')

features = []
valid_indices = []

for idx, filepath in enumerate(df['filepath']):
    rgb = extract_rgb_percentages(filepath)
    if rgb is not None:
        features.append(rgb)
        valid_indices.append(idx)

# Save features and valid labels only
X = np.array(features)
df_valid = df.iloc[valid_indices].reset_index(drop=True)

np.save('data/starter/lip_rgb_features.npy', X)
df_valid.to_csv('data/starter/labels_valid.csv', index=False)

print(f"Extracted RGB features for {len(X)}/{len(df)} lip images")
print(f"Saved to data/starter/lip_rgb_features.npy")