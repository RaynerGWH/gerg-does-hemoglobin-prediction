import pickle
import numpy as np
from PIL import Image
import sys
from pillow_heif import register_heif_opener

register_heif_opener()

def extract_rgb_percentages(img_path):
    """Extract RGB percentages from image"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    pixels = np.array(img)
    
    total = pixels.sum()
    red_pct = (pixels[:,:,0].sum() / total) * 100
    green_pct = (pixels[:,:,1].sum() / total) * 100
    blue_pct = (pixels[:,:,2].sum() / total) * 100
    
    return [red_pct, green_pct, blue_pct]

def predict(image_path):
    """Predict hemoglobin from image"""
    # Load model
    with open('weights/final_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Extract features
    features = np.array([extract_rgb_percentages(image_path)])
    
    # Predict
    hgb = model.predict(features)[0]
    
    return hgb

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    prediction = predict(img_path)
    print(f"Predicted HgB: {prediction:.1f} g/dL")