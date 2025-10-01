# src/data/load_data.py
import re
import pandas as pd
from pathlib import Path

def parse_hgb_from_filename(filename):
    """
    Extract hemoglobin value from filename.
    Handles formats like:
    - HgB_8.9gdl_Individual02_01
    - Random_HgB_4.1gdl
    """
    match = re.search(r'HgB_(\d+\.?\d*)gdl', filename)
    if match:
        return float(match.group(1))
    return None

def load_starter_data(data_dir='data/starter/images'):
    """
    Load all image files (jpg, jpeg, png, heic) and extract HgB labels.
    """
    data_path = Path(data_dir)
    
    # Common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.heic', '.HEIC', '.JPG', '.JPEG', '.PNG']
    
    records = []
    for img_file in data_path.iterdir():
        # Check if it's a file (not directory) and has image extension
        if img_file.is_file() and img_file.suffix in image_extensions:
            hgb = parse_hgb_from_filename(img_file.name)
            if hgb is not None:
                records.append({
                    'image_id': img_file.stem,
                    'filepath': str(img_file),
                    'hgb': hgb,
                    'extension': img_file.suffix
                })
            else:
                print(f"Warning: Could not parse HgB from {img_file.name}")
    
    if not records:
        print(f"ERROR: No images found in {data_dir}")
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    return df.sort_values('hgb').reset_index(drop=True)

if __name__ == '__main__':
    df = load_starter_data()
    
    if len(df) == 0:
        print("No data loaded. Check your data/starter/images directory.")
    else:
        print(f"\nLoaded {len(df)} images")
        print(f"HgB range: {df['hgb'].min():.1f} - {df['hgb'].max():.1f} g/dL")
        print(f"\nFile types found:")
        print(df['extension'].value_counts())
        print(f"\nFirst 10 samples:")
        print(df[['image_id', 'hgb', 'extension']].head(10))
        
        # Save
        df.to_csv('data/starter/labels.csv', index=False)
        print(f"\nSaved labels to data/starter/labels.csv")