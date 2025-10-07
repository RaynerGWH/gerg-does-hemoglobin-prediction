"""
Compare inference predictions with ground truth labels
"""

import pandas as pd
import numpy as np
from pathlib import Path


def compare_predictions():
    """Compare predictions with ground truth"""
    
    print("=" * 80)
    print("COMPARING PREDICTIONS WITH GROUND TRUTH")
    print("=" * 80)
    
    # Load files
    labels_path = Path('test_inference_setup/labels.csv')
    predictions_path = Path('predictions.csv')
    
    if not labels_path.exists():
        print(f"❌ Ground truth not found: {labels_path}")
        return
    
    if not predictions_path.exists():
        print(f"❌ Predictions not found: {predictions_path}")
        return
    
    gt = pd.read_csv(labels_path)
    pred = pd.read_csv(predictions_path)
    
    print(f"\n📊 Loaded data:")
    print(f"   Ground truth: {len(gt)} samples")
    print(f"   Predictions: {len(pred)} samples")
    
    # Merge on image_id
    # Predictions use 'filename' (e.g., '001.jpg'), labels use 'image_id' (e.g., '001')
    pred['image_id'] = pred['filename'].str.replace('.jpg', '')
    
    merged = gt.merge(pred[['image_id', 'predicted_hgb']], on='image_id', how='left')
    
    # Calculate metrics
    valid_mask = ~merged['predicted_hgb'].isna()
    valid_merged = merged[valid_mask]
    
    if len(valid_merged) == 0:
        print("\n❌ No valid predictions found!")
        return
    
    mae = np.abs(valid_merged['hgb'] - valid_merged['predicted_hgb']).mean()
    rmse = np.sqrt(((valid_merged['hgb'] - valid_merged['predicted_hgb']) ** 2).mean())
    r2 = np.corrcoef(valid_merged['hgb'], valid_merged['predicted_hgb'])[0, 1] ** 2
    
    print("\n" + "=" * 80)
    print("📈 RESULTS")
    print("=" * 80)
    print(f"\n✅ Valid predictions: {len(valid_merged)}/{len(gt)}")
    
    if len(merged) > len(valid_merged):
        failed = len(merged) - len(valid_merged)
        print(f"❌ Failed predictions: {failed}/{len(gt)}")
    
    print(f"\n📊 Performance Metrics:")
    print(f"   MAE (Mean Absolute Error): {mae:.3f} g/dL")
    print(f"   RMSE (Root Mean Squared Error): {rmse:.3f} g/dL")
    print(f"   R² Score: {r2:.3f}")
    
    if mae <= 0.8:
        print(f"\n🎉 TARGET ACHIEVED! MAE ≤ 0.8 g/dL")
    else:
        print(f"\n⚠️  MAE > 0.8 g/dL (target not met)")
    
    # Show comparison table
    print("\n" + "=" * 80)
    print("📋 SAMPLE COMPARISONS")
    print("=" * 80)
    
    comparison = valid_merged[['image_id', 'hgb', 'predicted_hgb']].copy()
    comparison['error'] = comparison['hgb'] - comparison['predicted_hgb']
    comparison['abs_error'] = comparison['error'].abs()
    
    # Sort by absolute error
    comparison_sorted = comparison.sort_values('abs_error', ascending=False)
    
    print("\n🔴 Worst 5 predictions:")
    print(comparison_sorted.head(5).to_string(index=False))
    
    print("\n🟢 Best 5 predictions:")
    print(comparison_sorted.tail(5).to_string(index=False))
    
    # Error distribution
    print("\n" + "=" * 80)
    print("📊 ERROR DISTRIBUTION")
    print("=" * 80)
    
    bins = [0, 0.5, 1.0, 1.5, 2.0, float('inf')]
    labels = ['0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '>2.0']
    
    comparison['error_bin'] = pd.cut(comparison['abs_error'], bins=bins, labels=labels)
    error_dist = comparison['error_bin'].value_counts().sort_index()
    
    for bin_label, count in error_dist.items():
        pct = (count / len(comparison)) * 100
        bar = '█' * int(pct / 2)
        print(f"  {bin_label:>10s} g/dL: {count:3d} samples ({pct:5.1f}%) {bar}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    compare_predictions()
