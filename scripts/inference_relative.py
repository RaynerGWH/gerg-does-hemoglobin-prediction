import pickle
import numpy as np
from PIL import Image
import sys
from pillow_heif import register_heif_opener

register_heif_opener()

def extract_relative_color_features(img_path):
    """Extract relative color features from image (same as training)"""
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
    
    # 2. RELATIVE: RGB Ratios
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
    
    # Return 13 features (matching training)
    return [
        red_pct, green_pct, blue_pct,
        r_ratio, g_ratio, b_ratio,
        rg_ratio, rb_ratio, gb_ratio,
        red_dominance,
        r_norm, g_norm, b_norm
    ]

def predict(image_path):
    """Predict hemoglobin from image using relative features"""
    # Load model
    with open('weights/final_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Extract features
    features = np.array([extract_relative_color_features(image_path)])
    
    # Predict
    hgb = model.predict(features)[0]
    
    # Display feature values for debugging
    print(f"\nExtracted features:")
    feature_names = [
        'Red%', 'Green%', 'Blue%',
        'R_ratio', 'G_ratio', 'B_ratio',
        'RG_ratio', 'RB_ratio', 'GB_ratio',
        'Red_dom', 'R_norm', 'G_norm', 'B_norm'
    ]
    for name, val in zip(feature_names, features[0]):
        print(f"  {name}: {val:.3f}")
    
    return hgb

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inference_relative.py <image_path>")
        print("\nExample:")
        print("  python inference_relative.py data/starter/images/HgB_10.7gdl_Individual01.heic")
        sys.exit(1)
    
    img_path = sys.argv[1]
    print(f"Analyzing image: {img_path}")
    
    prediction = predict(img_path)
    
    print(f"\n{'='*60}")
    print(f"🩸 Predicted HgB: {prediction:.1f} g/dL")
    print(f"{'='*60}")