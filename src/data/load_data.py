# src/data/load_data.py
import re
import pandas as pd
from pathlib import Path

def parse_hgb_from_filename(filename):
    """
    Extract hemoglobin value from filename.
    
    Examples:
        HgB_8.9gdl_Individual02_01.jpg -> 8.9
        HgB_10.7gdl_Individual01.jpg -> 10.7
    """
    match = re.search(r'HgB_(\d+\.?\d*)gdl', filename)
    if match:
        return float(match.group(1))
    return None

def load_starter_data(data_dir='data/starter/images'):
    """
    Load starter dataset and create labels dataframe.
    
    Returns:
        pd.DataFrame with columns: image_id, filepath, hgb
    """
    data_path = Path(data_dir)
    
    records = []
    for img_file in data_path.glob('*.jpg'):  # adjust extension if needed
        hgb = parse_hgb_from_filename(img_file.name)
        if hgb is not None:
            records.append({
                'image_id': img_file.stem,
                'filepath': str(img_file),
                'hgb': hgb
            })
    
    df = pd.DataFrame(records)
    return df.sort_values('hgb').reset_index(drop=True)

if __name__ == '__main__':
    # Test the loader
    df = load_starter_data()
    print(f"Loaded {len(df)} images")
    print(f"\nHgB range: {df['hgb'].min():.1f} - {df['hgb'].max():.1f} g/dL")
    print(f"\nFirst few samples:")
    print(df.head())
    
    # Save to CSV for easy access
    df.to_csv('data/starter/labels.csv', index=False)
    print("\nSaved labels to data/starter/labels.csv")