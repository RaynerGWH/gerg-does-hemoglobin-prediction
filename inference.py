"""
Competition Inference Script
One-command run: python inference.py --images <path> --meta meta.csv --out preds.csv

All feature extraction code is embedded - no external dependencies.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
import sys

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


# ===== FEATURE EXTRACTION (EMBEDDED) =====

def calculate_brightness(img_array):
    """Calculate average brightness (0-255 scale)"""
    return float(np.mean(img_array))


def calculate_contrast(img_array):
    """Calculate RMS contrast"""
    return float(np.std(img_array))


def detect_blur(img_array):
    """Detect blur using Laplacian variance (higher = sharper)"""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_saturation(img_array):
    """Calculate average saturation in HSV space"""
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 1].mean())


def calculate_lighting_uniformity(img_array):
    """Calculate lighting uniformity (lower std = more uniform)"""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    return float(np.std(gray))


def extract_color_moments(channel):
    """Extract first 3 statistical moments from color channel"""
    mean = np.mean(channel)
    std = np.std(channel)
    skewness = np.mean(((channel - mean) / (std + 1e-6)) ** 3)
    return float(mean), float(std), float(skewness)


def extract_enhanced_features(img_path):
    """
    Extract 28 comprehensive features from lip image
    
    Returns:
        Dictionary with 28 features, or None if extraction fails
    """
    try:
        # Load and resize image
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        # Separate RGB channels
        r_channel = img_array[:, :, 0]
        g_channel = img_array[:, :, 1]
        b_channel = img_array[:, :, 2]
        
        # Mean values
        mean_r = r_channel.mean()
        mean_g = g_channel.mean()
        mean_b = b_channel.mean()
        
        # RGB Percentages (absolute)
        total = img_array.sum()
        red_pct = (r_channel.sum() / total) * 100
        green_pct = (g_channel.sum() / total) * 100
        blue_pct = (b_channel.sum() / total) * 100
        
        # Relative RGB Ratios (lighting-independent)
        total_mean = mean_r + mean_g + mean_b + 1e-6
        r_ratio = mean_r / total_mean
        g_ratio = mean_g / total_mean
        b_ratio = mean_b / total_mean
        
        # Color ratios
        rg_ratio = mean_r / (mean_g + 1e-6)
        rb_ratio = mean_r / (mean_b + 1e-6)
        gb_ratio = mean_g / (mean_b + 1e-6)
        
        # Color dominance
        red_dominance = mean_r - mean_b
        redness_index = (mean_r - mean_g) / (mean_r + mean_g + 1e-6)
        
        # Normalized values (0-1)
        r_norm = mean_r / 255
        g_norm = mean_g / 255
        b_norm = mean_b / 255
        
        # Image quality
        brightness = calculate_brightness(img_array)
        contrast = calculate_contrast(img_array)
        blur_score = detect_blur(img_array)
        saturation = calculate_saturation(img_array)
        lighting_uniformity = calculate_lighting_uniformity(img_array)
        
        # Color distribution moments
        r_mean, r_std, r_skew = extract_color_moments(r_channel)
        g_mean, g_std, g_skew = extract_color_moments(g_channel)
        b_mean, b_std, b_skew = extract_color_moments(b_channel)
        
        # Compile features (order matters!)
        features = {
            'red_pct': red_pct,
            'green_pct': green_pct,
            'blue_pct': blue_pct,
            'r_ratio': r_ratio,
            'g_ratio': g_ratio,
            'b_ratio': b_ratio,
            'rg_ratio': rg_ratio,
            'rb_ratio': rb_ratio,
            'gb_ratio': gb_ratio,
            'red_dominance': red_dominance,
            'r_norm': r_norm,
            'g_norm': g_norm,
            'b_norm': b_norm,
            'brightness': brightness,
            'contrast': contrast,
            'blur_score': blur_score,
            'saturation': saturation,
            'lighting_uniformity': lighting_uniformity,
            'redness_index': redness_index,
            'r_mean': r_mean,
            'r_std': r_std,
            'r_skew': r_skew,
            'g_mean': g_mean,
            'g_std': g_std,
            'g_skew': g_skew,
            'b_mean': b_mean,
            'b_std': b_std,
            'b_skew': b_skew,
        }
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from {img_path}: {e}", file=sys.stderr)
        return None


# ===== MODEL LOADING AND INFERENCE =====

def load_onnx_models():
    """Load ONNX model and scaler from weights/ directory"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime is required. Install with: pip install onnxruntime", file=sys.stderr)
        sys.exit(1)
    
    # Find weights directory
    script_dir = Path(__file__).parent
    weights_dir = script_dir / 'weights'
    
    if not weights_dir.exists():
        print(f"ERROR: Weights directory not found: {weights_dir}", file=sys.stderr)
        sys.exit(1)
    
    scaler_path = weights_dir / 'scaler.onnx'
    model_path = weights_dir / 'model.onnx'
    
    if not scaler_path.exists():
        print(f"ERROR: Scaler not found: {scaler_path}", file=sys.stderr)
        sys.exit(1)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load ONNX sessions
    scaler_session = ort.InferenceSession(str(scaler_path))
    model_session = ort.InferenceSession(str(model_path))
    
    return scaler_session, model_session


def predict(images_dir, meta_csv, output_csv='predictions.csv'):
    """
    Make predictions on images
    
    Args:
        images_dir: Directory containing images
        meta_csv: CSV file with image metadata (must have 'image_id' column)
        output_csv: Output CSV file path
    
    Returns:
        predictions_df: DataFrame with predictions
    """
    print("\n" + "=" * 80)
    print("HEMOGLOBIN PREDICTION - INFERENCE")
    print("=" * 80)
    
    # Load ONNX models
    print("\n📦 Loading ONNX models...")
    scaler_session, model_session = load_onnx_models()
    print("   ✓ Models loaded successfully")
    
    # Load metadata
    print(f"\n📊 Loading metadata from: {meta_csv}")
    meta_df = pd.read_csv(meta_csv, dtype={'image_id': str})  # Force image_id to be read as string
    
    if 'image_id' not in meta_df.columns:
        print("ERROR: Metadata CSV must contain 'image_id' column", file=sys.stderr)
        sys.exit(1)
    
    print(f"   ✓ Found {len(meta_df)} images")
    
    # Extract features and predict
    print("\n🔍 Processing images...")
    predictions = []
    image_ids = []
    failed_count = 0
    
    for idx, row in meta_df.iterrows():
        image_id = str(row['image_id'])  # Convert to string to preserve formatting
        # Construct filename from image_id (e.g., '001' -> '001.jpg')
        filename = f"{image_id}.jpg"
        img_path = Path(images_dir) / filename
        
        # Debug: Print first path to verify
        if idx == 0:
            print(f"  DEBUG: Looking for: {img_path.absolute()}")
        
        # Validate image exists
        if not img_path.exists():
            print(f"  ✗ [{idx+1}/{len(meta_df)}] {filename} - NOT FOUND (path: {img_path})")
            predictions.append(np.nan)
            image_ids.append(image_id)
            failed_count += 1
            continue
        
        # Extract features
        features = extract_enhanced_features(img_path)
        if features is None:
            print(f"  ✗ [{idx+1}/{len(meta_df)}] {filename} - FEATURE EXTRACTION FAILED")
            predictions.append(np.nan)
            image_ids.append(image_id)
            failed_count += 1
            continue
        
        # Convert to numpy array (shape: 1x28)
        feature_array = np.array(list(features.values()), dtype=np.float32).reshape(1, -1)
        
        # Scale features using ONNX scaler
        scaler_input_name = scaler_session.get_inputs()[0].name
        scaled_features = scaler_session.run(None, {scaler_input_name: feature_array})[0]
        
        # Predict using ONNX model
        model_input_name = model_session.get_inputs()[0].name
        prediction = model_session.run(None, {model_input_name: scaled_features})[0]
        
        pred_value = float(prediction[0])
        predictions.append(pred_value)
        image_ids.append(image_id)
        print(f"  ✓ [{idx+1}/{len(meta_df)}] {filename:<40s} -> {pred_value:.2f} g/dL")
    
    # Create output DataFrame
    predictions_df = pd.DataFrame({
        'image_id': image_ids,
        'predicted_hgb': predictions
    })
    
    # Save predictions
    predictions_df.to_csv(output_csv, index=False)
    
    print("\n" + "=" * 80)
    print("✅ INFERENCE COMPLETE")
    print("=" * 80)
    print(f"   Successful: {len(meta_df) - failed_count}/{len(meta_df)}")
    print(f"   Failed: {failed_count}/{len(meta_df)}")
    print(f"\n💾 Predictions saved to: {Path(output_csv).absolute()}")
    
    # Statistics (only for successful predictions)
    valid_predictions = np.array([p for p in predictions if not np.isnan(p)])
    if len(valid_predictions) > 0:
        print(f"\n📊 Prediction Statistics:")
        print(f"   Count: {len(valid_predictions)}")
        print(f"   Mean: {np.mean(valid_predictions):.2f} g/dL")
        print(f"   Std: {np.std(valid_predictions):.2f} g/dL")
        print(f"   Min: {np.min(valid_predictions):.2f} g/dL")
        print(f"   Max: {np.max(valid_predictions):.2f} g/dL")
        print(f"   Median: {np.median(valid_predictions):.2f} g/dL")
    
    return predictions_df


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(
        description='Hemoglobin Prediction Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --images ./test_images --meta meta.csv --out predictions.csv
  python inference.py --images /path/to/images --meta metadata.csv
        """
    )
    
    parser.add_argument('--images', required=True, 
                       help='Directory containing test images')
    parser.add_argument('--meta', required=True,
                       help='CSV file with image metadata (must have "image_id" column)')
    parser.add_argument('--out', default='predictions.csv',
                       help='Output CSV file (default: predictions.csv)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.images).exists():
        print(f"❌ Error: Images directory not found: {args.images}")
        sys.exit(1)
    
    if not Path(args.meta).exists():
        print(f"❌ Error: Metadata file not found: {args.meta}")
        sys.exit(1)
    
    # Run inference
    predict(args.images, args.meta, args.out)


if __name__ == '__main__':
    main()
