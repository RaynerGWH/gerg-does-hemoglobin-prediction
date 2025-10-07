"""
Competition Inference Script
One-command run: python inference.py --images <path> --meta meta.csv --out preds.csv

This script loads the trained model and makes predictions on new images.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import feature extraction (will use importlib if needed)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "extract_enhanced_features",
    Path(__file__).parent / "01_extract_enhanced_features.py"
)
efe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(efe_module)
extract_enhanced_features = efe_module.extract_enhanced_features


def load_model():
    """Load trained model and scaler"""
    try:
        import onnxruntime as ort
        use_onnx = True
        weights_dir = Path(__file__).parent.parent / 'weights'
        
        # Load ONNX models
        scaler_session = ort.InferenceSession(str(weights_dir / 'scaler.onnx'))
        model_session = ort.InferenceSession(str(weights_dir / 'model.onnx'))
        
        print("✓ Loaded ONNX models (competition format)")
        return {'scaler': scaler_session, 'model': model_session}, 'onnx'
    
    except (ImportError, FileNotFoundError):
        # Fallback to pickle
        import pickle
        weights_dir = Path(__file__).parent.parent / 'weights'
        
        with open(weights_dir / 'final_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(weights_dir / 'feature_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        print("✓ Loaded pickle models (development format)")
        return {'scaler': scaler, 'model': model}, 'pickle'


def predict(images_dir, meta_csv, output_csv='predictions.csv', model_type='onnx'):
    """
    Make predictions on images
    
    Args:
        images_dir: Directory containing images
        meta_csv: CSV file with image metadata (must have 'filename' column)
        output_csv: Output CSV file path
        model_type: 'onnx' or 'pickle'
    
    Returns:
        predictions_df: DataFrame with predictions
    """
    print("\n" + "=" * 70)
    print("HEMOGLOBIN PREDICTION - INFERENCE")
    print("=" * 70)
    
    # Load model
    print("\n📦 Loading model...")
    models, fmt = load_model()
    
    # Load metadata
    print(f"\n📊 Loading metadata from: {meta_csv}")
    meta_df = pd.read_csv(meta_csv)
    print(f"   Found {len(meta_df)} images")
    
    # Extract features and predict
    print("\n🔍 Extracting features and making predictions...")
    predictions = []
    filenames = []
    failed_images = []
    
    for idx, row in meta_df.iterrows():
        filename = row['filename']
        img_path = Path(images_dir) / filename
        
        if not img_path.exists():
            print(f"  ✗ [{idx+1}/{len(meta_df)}] {filename} - NOT FOUND")
            failed_images.append(filename)
            predictions.append(np.nan)
            filenames.append(filename)
            continue
        
        # Extract features
        features = extract_enhanced_features(img_path)
        
        if features is None:
            print(f"  ✗ [{idx+1}/{len(meta_df)}] {filename} - FEATURE EXTRACTION FAILED")
            failed_images.append(filename)
            predictions.append(np.nan)
            filenames.append(filename)
            continue
        
        # Convert to array
        feature_array = np.array(list(features.values())).reshape(1, -1)
        
        # Predict
        if fmt == 'onnx':
            # ONNX inference
            scaler_input = {models['scaler'].get_inputs()[0].name: feature_array.astype(np.float32)}
            scaled_features = models['scaler'].run(None, scaler_input)[0]
            
            model_input = {models['model'].get_inputs()[0].name: scaled_features.astype(np.float32)}
            prediction = models['model'].run(None, model_input)[0][0][0]
        else:
            # Pickle inference
            scaled_features = models['scaler'].transform(feature_array)
            prediction = models['model'].predict(scaled_features)[0]
        
        predictions.append(prediction)
        filenames.append(filename)
        
        print(f"  ✓ [{idx+1}/{len(meta_df)}] {filename}: Predicted HgB = {prediction:.1f} g/dL")
    
    # Create output DataFrame
    predictions_df = pd.DataFrame({
        'filename': filenames,
        'predicted_hgb': predictions
    })
    
    # Save predictions
    predictions_df.to_csv(output_csv, index=False)
    
    print("\n" + "=" * 70)
    print("✅ INFERENCE COMPLETE")
    print("=" * 70)
    print(f"\n📊 Results:")
    print(f"   Total images: {len(meta_df)}")
    print(f"   Successful predictions: {len(predictions) - len(failed_images)}")
    print(f"   Failed images: {len(failed_images)}")
    print(f"\n💾 Predictions saved to: {output_csv}")
    
    if failed_images:
        print(f"\n❌ Failed images:")
        for fname in failed_images:
            print(f"   - {fname}")
    
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
                       help='CSV file with image metadata (must have "filename" column)')
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
