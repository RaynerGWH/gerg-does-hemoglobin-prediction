"""
Export scikit-learn model to ONNX format for competition submission
Run from scripts/: python export_model_onnx.py
"""

import pickle
import numpy as np
from pathlib import Path
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def export_to_onnx():
    """Export trained model to ONNX format"""
    
    print("=" * 70)
    print("EXPORTING MODEL TO ONNX FORMAT")
    print("=" * 70)
    
    weights_dir = Path('../weights')
    
    # Load model and scaler
    print("\n📦 Loading trained model...")
    with open(weights_dir / 'final_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open(weights_dir / 'feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    print("   ✓ Model loaded")
    print("   ✓ Scaler loaded")
    
    # Define input shape (28 features)
    num_features = 28
    initial_type = [('float_input', FloatTensorType([None, num_features]))]
    
    # Convert scaler to ONNX
    print("\n🔧 Converting scaler to ONNX...")
    scaler_onnx = convert_sklearn(
        scaler, 
        initial_types=initial_type,
        target_opset=12
    )
    
    # Save scaler
    scaler_onnx_path = weights_dir / 'scaler.onnx'
    with open(scaler_onnx_path, 'wb') as f:
        f.write(scaler_onnx.SerializeToString())
    print(f"   ✓ Saved: {scaler_onnx_path}")
    
    # Convert model to ONNX
    print("\n🔧 Converting model to ONNX...")
    model_onnx = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=12
    )
    
    # Save model
    model_onnx_path = weights_dir / 'model.onnx'
    with open(model_onnx_path, 'wb') as f:
        f.write(model_onnx.SerializeToString())
    print(f"   ✓ Saved: {model_onnx_path}")
    
    # Check file sizes
    scaler_size_mb = scaler_onnx_path.stat().st_size / (1024 * 1024)
    model_size_mb = model_onnx_path.stat().st_size / (1024 * 1024)
    total_size_mb = scaler_size_mb + model_size_mb
    
    print("\n" + "=" * 70)
    print("✅ EXPORT COMPLETE")
    print("=" * 70)
    print(f"\n📊 Model Sizes:")
    print(f"   Scaler: {scaler_size_mb:.2f} MB")
    print(f"   Model:  {model_size_mb:.2f} MB")
    print(f"   Total:  {total_size_mb:.2f} MB")
    
    if total_size_mb <= 10:
        print(f"\n🏆 QUALIFIES FOR EDGE-LITE TRACK! (≤ 10 MB)")
    elif total_size_mb <= 50:
        print(f"\n✅ QUALIFIES FOR ACCURACY TRACK (≤ 50 MB)")
    else:
        print(f"\n⚠️  WARNING: Exceeds 50 MB limit!")
    
    # Save version info
    version_path = weights_dir / 'onnx_version.txt'
    with open(version_path, 'w') as f:
        f.write("ONNX Runtime Version: See requirements.txt\n")
        f.write("Opset Version: 12\n")
        f.write(f"Model Type: {type(model).__name__}\n")
        f.write(f"Input Shape: (batch_size, {num_features})\n")
        f.write(f"Output Shape: (batch_size, 1)\n")
    
    print(f"\n📄 Version info saved: {version_path}")
    print("\nNext: Test inference with ONNX model!")


if __name__ == '__main__':
    export_to_onnx()
