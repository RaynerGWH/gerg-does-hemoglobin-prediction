import pandas as pd
import re
from pathlib import Path

def extract_hgb_from_filename(filename):
    """Extract HgB value from filename like 'HgB_10.7gdl_Individual01.heic' or 'Random_11.6gdl_...'"""
    # Pattern: HgB_<number>gdl OR Random_<number>gdl OR just <number>gdl
    match = re.search(r'(?:HgB_|Random_)?([\d.]+)gdl', filename)
    if match:
        return float(match.group(1))
    return None

# Get all images
images_dir = Path('data/starter/images')
image_files = sorted([f for f in images_dir.iterdir() 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.heic']])

print("=" * 60)
print("CREATING LABELS CSV FROM FILENAMES")
print("=" * 60)

# Extract data
data = []
for img_path in image_files:
    hgb = extract_hgb_from_filename(img_path.name)
    
    if hgb is not None:
        data.append({
            'image_id': img_path.stem,  # filename without extension
            'filename': img_path.name,
            'filepath': str(img_path),
            'hgb': hgb
        })
        print(f"✓ {img_path.name:50s} → HgB: {hgb:.1f} g/dL")
    else:
        print(f"⚠️  Could not extract HgB from: {img_path.name}")

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
output_path = Path('data/starter/labels.csv')
df.to_csv(output_path, index=False)

print(f"\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"✅ Created labels.csv with {len(df)} samples")
print(f"📁 Saved to: {output_path}")
print(f"\nHgB Statistics:")
print(f"  Min: {df['hgb'].min():.1f} g/dL")
print(f"  Max: {df['hgb'].max():.1f} g/dL")
print(f"  Mean: {df['hgb'].mean():.1f} g/dL")
print(f"  Std: {df['hgb'].std():.1f} g/dL")

print(f"\nFirst 5 rows:")
print(df.head())

# Check for unique HgB values
unique_hgb = df['hgb'].unique()
print(f"\nUnique HgB values: {sorted(unique_hgb)}")
print(f"Number of unique values: {len(unique_hgb)}")

print(f"\n" + "=" * 60)
print("⚠️  IMPORTANT NOTE:")
print("=" * 60)
print(f"You have {len(df)} images but lip_rgb_features.npy has 28 samples!")
print("This mismatch suggests:")
print("1. Some images failed during feature extraction")
print("2. Or features were extracted from different images")
print("\nRecommendation: Re-extract features to match these {len(df)} images")