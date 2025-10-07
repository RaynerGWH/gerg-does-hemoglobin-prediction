"""
Prepare test data for inference in competition format
- Converts images to simple IDs (001.jpg, 002.jpg, etc.)
- Creates meta.csv with required columns
- Creates labels.csv for comparison (optional)
"""

import pandas as pd
import re
import shutil
from pathlib import Path
from PIL import Image

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


def extract_hgb_from_filename(filename):
    """Extract HgB value from filename"""
    match = re.search(r'(?:HgB_|Random_)?([\d.]+)gdl', filename)
    if match:
        return float(match.group(1))
    return None


def prepare_test_data():
    """Prepare test data in competition format"""
    
    print("=" * 80)
    print("PREPARING TEST DATA FOR INFERENCE")
    print("=" * 80)
    
    # Paths
    source_images_dir = Path('../data/starter/images')
    test_dir = Path('../test_inference_setup')
    test_images_dir = test_dir / 'images'
    
    # Get source images
    image_files = sorted([f for f in source_images_dir.iterdir() 
                          if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.heic']])
    
    print(f"\n📂 Found {len(image_files)} source images")
    
    # Prepare data
    meta_data = []
    labels_data = []
    
    print(f"\n🔄 Processing images...")
    
    for idx, src_img in enumerate(image_files, start=1):
        # Create simple image_id (001, 002, etc.)
        image_id = f"{idx:03d}"
        
        # Convert to JPG and copy
        dest_filename = f"{image_id}.jpg"
        dest_path = test_images_dir / dest_filename
        
        try:
            # Load and convert to RGB
            img = Image.open(src_img).convert('RGB')
            img.save(dest_path, 'JPEG', quality=95)
            
            # Extract HgB value
            hgb = extract_hgb_from_filename(src_img.name)
            
            # Create meta.csv row (with dummy data for testing)
            meta_data.append({
                'image_id': image_id,  
                'device_id': 'test_device',
                'device_brand': 'Unknown',
                'device_model': 'Unknown',
                'iso_bucket': 'medium',
                'exposure_bucket': 'normal',
                'wb_bucket': 'auto',
                'ambient_light': 'indoor',
                'distance_band': 'normal',
                'skin_tone_proxy': 'medium',
                'age_band': 'adult',
                'gender': 'unknown'
            })
            
            # Create labels.csv row (if HgB extracted)
            if hgb is not None:
                labels_data.append({
                    'image_id': image_id,
                    'hgb': hgb
                })
            
            print(f"  ✓ [{idx:3d}/{len(image_files)}] {src_img.name:<50s} → {dest_filename:<12s} (HgB: {hgb if hgb else 'N/A'})")
            
        except Exception as e:
            print(f"  ✗ [{idx:3d}/{len(image_files)}] {src_img.name:<50s} FAILED: {e}")
    
    # Save meta.csv
    meta_df = pd.DataFrame(meta_data)
    meta_path = test_dir / 'meta.csv'
    meta_df.to_csv(meta_path, index=False)
    print(f"\n✅ Created meta.csv with {len(meta_df)} rows")
    print(f"   Saved to: {meta_path.absolute()}")
    
    # Save labels.csv (ground truth for comparison)
    labels_df = pd.DataFrame(labels_data)
    labels_path = test_dir / 'labels.csv'
    labels_df.to_csv(labels_path, index=False)
    print(f"\n✅ Created labels.csv with {len(labels_df)} rows")
    print(f"   Saved to: {labels_path.absolute()}")
    
    # Show statistics
    if len(labels_df) > 0:
        print(f"\n📊 HgB Statistics (Ground Truth):")
        print(f"   Count: {len(labels_df)}")
        print(f"   Min: {labels_df['hgb'].min():.1f} g/dL")
        print(f"   Max: {labels_df['hgb'].max():.1f} g/dL")
        print(f"   Mean: {labels_df['hgb'].mean():.1f} g/dL")
        print(f"   Median: {labels_df['hgb'].median():.1f} g/dL")
    
    print("\n" + "=" * 80)
    print("✅ TEST DATA READY!")
    print("=" * 80)
    print(f"\n📁 Directory structure:")
    print(f"   {test_dir}/")
    print(f"   ├── images/")
    print(f"   │   ├── 001.jpg")
    print(f"   │   ├── 002.jpg")
    print(f"   │   └── ... ({len(image_files)} images)")
    print(f"   ├── meta.csv ({len(meta_df)} rows)")
    print(f"   └── labels.csv ({len(labels_df)} rows)")
    
    print("\n" + "=" * 80)
    print("🚀 RUN INFERENCE")
    print("=" * 80)
    print(f"\n  python inference.py --images {test_images_dir} --meta {meta_path} --out predictions.csv")
    print("\n" + "=" * 80)
    print("📊 COMPARE RESULTS")
    print("=" * 80)
    print(f"  Ground truth: {labels_path}")
    print(f"  Predictions: predictions.csv")
    print("\n  Compare with:")
    print("  python -c \"import pandas as pd; gt=pd.read_csv('test_inference_setup/labels.csv'); pred=pd.read_csv('predictions.csv'); merged=gt.merge(pred.rename(columns={'predicted_hgb':'pred'}), left_on='image_id', right_on='filename'); print(merged[['image_id','hgb','pred']]); mae=(merged['hgb']-merged['pred']).abs().mean(); print(f'\\nMAE: {mae:.3f} g/dL')\"")
    
    print("\n" + "=" * 80)
    
    # Show sample rows
    print("\n📋 Sample meta.csv (first 3 rows):")
    print(meta_df.head(3).to_string(index=False))
    
    print("\n📋 Sample labels.csv (first 5 rows):")
    print(labels_df.head(5).to_string(index=False))


if __name__ == '__main__':
    prepare_test_data()
