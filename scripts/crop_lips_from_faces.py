import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from pillow_heif import register_heif_opener
import pandas as pd

register_heif_opener()

def detect_and_crop_lips(img_path, output_dir):
    """Detect face and crop lip region using OpenCV"""
    # Read image
    img = Image.open(img_path).convert('RGB')
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        print(f"  ⚠️  No face detected in {Path(img_path).name}")
        # If no face detected, assume it's already a lip closeup - just copy it
        img.save(output_dir / Path(img_path).name)
        return output_dir / Path(img_path).name
    
    # Get the largest face
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    
    # Define lip region (lower third of face)
    # Lips are typically in bottom 40% of face, centered horizontally
    lip_y_start = y + int(h * 0.6)  # Start at 60% down the face
    lip_y_end = y + h               # End at bottom of face
    lip_x_start = x + int(w * 0.25) # Start at 25% from left
    lip_x_end = x + int(w * 0.75)   # End at 75% from left
    
    # Crop lip region
    lip_region = img_cv[lip_y_start:lip_y_end, lip_x_start:lip_x_end]
    
    # Convert back to RGB for PIL
    lip_region_rgb = cv2.cvtColor(lip_region, cv2.COLOR_BGR2RGB)
    lip_img = Image.fromarray(lip_region_rgb)
    
    # Save cropped lip
    output_path = output_dir / Path(img_path).name
    lip_img.save(output_path)
    
    return output_path

def main():
    print("=" * 60)
    print("LIP CROPPING FROM FACE IMAGES")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path('data/starter/lips_cropped')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load labels (try multiple possible filenames)
    # Handle both running from root or scripts folder
    labels_files = [
        'data/starter/labels_valid.csv',
        'data/starter/labels.csv',
        'data/starter/labels_clean.csv',
        '../data/starter/labels_valid.csv',
        '../data/starter/labels.csv',
        '../data/starter/labels_clean.csv'
    ]
    
    df = None
    for labels_file in labels_files:
        if Path(labels_file).exists():
            df = pd.read_csv(labels_file)
            print(f"✅ Loaded labels from: {labels_file}")
            break
    
    if df is None:
        print("❌ Error: No labels CSV found!")
        print("Expected one of:")
        for f in labels_files:
            print(f"  - {f}")
        return
    
    print(f"\n🔍 Processing {len(df)} images...\n")
    
    successful_crops = []
    
    for idx, row in df.iterrows():
        filepath = row['filepath']
        filename = Path(filepath).name
        
        print(f"{idx+1:2d}. {filename:50s}", end=" ")
        
        try:
            cropped_path = detect_and_crop_lips(filepath, output_dir)
            successful_crops.append({
                'image_id': row['image_id'],
                'filename': cropped_path.name,
                'filepath': str(cropped_path),
                'hgb': row['hgb']
            })
            print("✅ Cropped")
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    # Save updated labels for cropped images
    df_cropped = pd.DataFrame(successful_crops)
    df_cropped.to_csv('data/starter/labels_cropped.csv', index=False)
    
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully processed: {len(successful_crops)}/{len(df)} images")
    print(f"📁 Cropped lips saved to: {output_dir}")
    print(f"📄 New labels saved to: data/starter/labels_cropped.csv")
    
    print(f"\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Check cropped images in data/starter/lips_cropped/")
    print("2. Update reextract_all_features.py to use 'labels_cropped.csv'")
    print("3. Run: python scripts/reextract_all_features.py")
    print("4. Run: python scripts/train_combined_model_relative.py")

if __name__ == "__main__":
    main()