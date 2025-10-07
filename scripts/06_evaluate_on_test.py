"""
Evaluate Model on Test Set (30 Images)
Run from project root: 
    python scripts/06_evaluate_on_test.py              # Evaluate default model
    python scripts/06_evaluate_on_test.py --model rf   # Evaluate Random Forest
    python scripts/06_evaluate_on_test.py --model gb   # Evaluate Gradient Boosting
    python scripts/06_evaluate_on_test.py --model both # Compare both models
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import sys
import os
import importlib.util
import argparse

# Load the enhanced feature extraction module
script_dir = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("extract_enhanced_features", 
                                                os.path.join(script_dir, "01_extract_enhanced_features.py"))
efe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(efe_module)
extract_enhanced_features = efe_module.extract_enhanced_features


def load_model_and_scaler(model_type=None):
    """
    Load trained model and feature scaler
    
    Args:
        model_type: 'rf', 'gb', or None for default
    
    Returns:
        model, scaler, model_name
    """
    weights_dir = Path('../weights')
    
    if model_type:
        # Load specific model type from subdirectory
        model_dir = weights_dir / model_type
        model_path = model_dir / 'final_model.pkl'
        scaler_path = model_dir / 'feature_scaler.pkl'
        model_name = model_type.upper()
    else:
        # Load default model
        model_path = weights_dir / 'final_model.pkl'
        scaler_path = weights_dir / 'feature_scaler.pkl'
        model_name = 'Default'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Scaler is optional (older models might not have it)
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    
    return model, scaler, model_name


def evaluate_on_test_set(model_type=None):
    """
    Evaluate model on 30 test images, showing validation and full results
    
    Args:
        model_type: 'rf', 'gb', or None for default
    
    Returns:
        results_df, mae, rmse, r2, model_name
    """
    
    print("=" * 70)
    print(f"EVALUATING ON TEST SET (30 IMAGES)")
    if model_type:
        print(f"Model: {model_type.upper()}")
    print("=" * 70)
    
    # Load model
    print("\n📦 Loading trained model...")
    model, scaler, model_name = load_model_and_scaler(model_type)
    print(f"   ✓ Model loaded ({model_name})")
    if scaler:
        print("   ✓ Feature scaler loaded")
    
    # Load test labels
    labels_path = Path('../data/starter/labels.csv')
    if not labels_path.exists():
        print(f"\n❌ Labels file not found: {labels_path}")
        return
    
    df = pd.read_csv(labels_path)
    print(f"\n📊 Found {len(df)} test images")
    
    # Try to load validation metadata to identify which images are in validation set
    processed_dir = Path('../data/processed')
    val_metadata_path = processed_dir / 'val_metadata.csv'
    val_images = set()
    
    if val_metadata_path.exists():
        try:
            meta_df = pd.read_csv(val_metadata_path)
            # Get validation image filenames (filename column has the image names)
            if 'filename' in meta_df.columns:
                val_images = set(meta_df['filename'].unique())
                print(f"   📋 Identified {len(val_images)} validation images:")
                for img in sorted(val_images):
                    print(f"      - {img}")
            elif 'original_filename' in meta_df.columns:
                val_images = set(meta_df['original_filename'].unique())
                print(f"   📋 Identified {len(val_images)} validation images: {sorted(val_images)}")
            else:
                print(f"   ⚠️  'filename' or 'original_filename' column not found in metadata")
                print(f"   Columns found: {list(meta_df.columns)}")
        except Exception as e:
            print(f"   ⚠️  Could not load validation metadata: {e}")
    else:
        print(f"   ℹ️  No validation metadata found at {val_metadata_path}")
        print(f"   All images will be treated as test set")
    
    # Extract features from test images
    print("\n🔍 Extracting features from test images...")
    features_list = []
    predictions = []
    actual_values = []
    filenames = []
    is_validation = []  # Track which are validation images
    failed_images = []
    
    for idx, row in df.iterrows():
        img_path = Path(row['filepath'])
        if not img_path.exists():
            img_path = Path('..') / img_path
        
        # Extract features
        features = extract_enhanced_features(img_path)
        
        if features is not None:
            # Convert dict to array (ensure same order as training)
            feature_array = np.array(list(features.values())).reshape(1, -1)
            
            # Scale features if scaler is available
            if scaler:
                feature_array = scaler.transform(feature_array)
            
            # Predict
            prediction = model.predict(feature_array)[0]
            
            predictions.append(prediction)
            actual_values.append(row['hgb'])
            filenames.append(row['filename'])
            is_validation.append(row['filename'] in val_images)
            
            
            val_status = " [VAL]" if row['filename'] in val_images else ""
            print(f"  ✓ [{idx+1}/{len(df)}] {row['filename']}{val_status}: "
                  f"Actual={row['hgb']:.1f}, Predicted={prediction:.1f} g/dL")
        else:
            failed_images.append(row['filename'])
            print(f"  ✗ [{idx+1}/{len(df)}] {row['filename']} - FAILED")
    
    # Convert to arrays
    predictions = np.array(predictions)
    actual_values = np.array(actual_values)
    is_validation = np.array(is_validation)
    
    # Calculate metrics for validation set (if we have validation images)
    if len(val_images) > 0 and any(is_validation):
        val_mask = is_validation
        val_predictions = predictions[val_mask]
        val_actuals = actual_values[val_mask]
        
        val_mae = mean_absolute_error(val_actuals, val_predictions)
        val_rmse = np.sqrt(mean_squared_error(val_actuals, val_predictions))
        val_r2 = r2_score(val_actuals, val_predictions)
        
        print("\n" + "=" * 70)
        print("VALIDATION SET RESULTS (Independent Images)")
        print("=" * 70)
        print(f"\n📊 Validation Set Performance ({len(val_predictions)} images):")
        print(f"   MAE:  {val_mae:.3f} g/dL")
        print(f"   RMSE: {val_rmse:.3f} g/dL")
        print(f"   R²:   {val_r2:.3f}")
        
        if val_mae <= 0.8:
            print(f"\n🎉 VALIDATION SUCCESS! Target achieved (MAE ≤ 0.8 g/dL)")
        else:
            print(f"\n⚠️  Validation MAE > 0.8 g/dL (target not met)")
    
    # Calculate metrics for ALL images
    print("\n" + "=" * 70)
    print("FULL TEST SET RESULTS (All 30 Images)")
    print("=" * 70)
    
    mae = mean_absolute_error(actual_values, predictions)
    rmse = np.sqrt(mean_squared_error(actual_values, predictions))
    r2 = r2_score(actual_values, predictions)
    
    print(f"\n📊 Full Dataset Performance ({len(predictions)} images):")
    print(f"   MAE:  {mae:.3f} g/dL")
    print(f"   RMSE: {rmse:.3f} g/dL")
    print(f"   R²:   {r2:.3f}")
    
    if mae <= 0.8:
        print(f"\n🎉 SUCCESS! Target achieved (MAE ≤ 0.8 g/dL)")
    else:
        print(f"\n⚠️  Target not reached. Need MAE ≤ 0.8 g/dL")
    
    print(f"\n✅ Successfully evaluated: {len(predictions)}/{len(df)} images")
    if failed_images:
        print(f"❌ Failed images: {len(failed_images)}")
        for fname in failed_images:
            print(f"   - {fname}")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'filename': filenames,
        'actual_hgb': actual_values,
        'predicted_hgb': predictions,
        'error': np.abs(actual_values - predictions),
        'is_validation': is_validation
    })
    
    # Sort by error
    results_df = results_df.sort_values('error', ascending=False)
    
    print(f"\n📋 Worst predictions (largest errors):")
    print(results_df.head(5)[['filename', 'actual_hgb', 'predicted_hgb', 'error', 'is_validation']].to_string(index=False))
    
    # Save results
    results_dir = Path('../results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = results_dir / 'test_evaluation.csv'
    
    # Try to save CSV with fallback filename if file is locked
    try:
        results_df.to_csv(results_path, index=False)
        print(f"\n💾 Saved results to: {results_path}")
    except PermissionError:
        # File might be open in Excel - try alternative name
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_path = results_dir / f'test_evaluation_{timestamp}.csv'
        results_df.to_csv(alt_path, index=False)
        print(f"\n⚠️  Original file was locked (close Excel if open)")
        print(f"💾 Saved results to: {alt_path}")
    
    # Create visualization
    print("\n📈 Creating visualization...")
    model_suffix = f"_{model_type}" if model_type else ""
    if len(val_images) > 0 and any(is_validation):
        create_evaluation_plot(actual_values, predictions, is_validation, mae, val_mae, r2, results_dir, model_name, model_suffix)
    else:
        create_evaluation_plot(actual_values, predictions, None, mae, None, r2, results_dir, model_name, model_suffix)
    
    return results_df, mae, rmse, r2, model_name


def create_evaluation_plot(actual, predicted, is_validation, mae, val_mae, r2, output_dir, model_name='Model', suffix=''):
    """Create scatter plot of predictions vs actual values, highlighting validation set"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Separate training and validation points
    if is_validation is not None and any(is_validation):
        train_mask = ~is_validation
        val_mask = is_validation
        
        # Training points
        ax.scatter(actual[train_mask], predicted[train_mask], 
                  alpha=0.6, s=100, edgecolors='black', linewidths=1, 
                  color='blue', label='Training images')
        
        # Validation points (highlighted)
        ax.scatter(actual[val_mask], predicted[val_mask], 
                  alpha=0.8, s=150, edgecolors='red', linewidths=2, 
                  color='orange', marker='D', label='Validation images')
    else:
        # All points (no validation split)
        ax.scatter(actual, predicted, alpha=0.6, s=100, edgecolors='black', linewidths=1)
    
    # Perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7, label='Perfect Prediction')
    
    # Labels and title
    ax.set_xlabel('Actual Hemoglobin (g/dL)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Hemoglobin (g/dL)', fontsize=12, fontweight='bold')
    
    if val_mae is not None:
        title = f'{model_name} Performance on Test Set (30 Images)\nFull MAE: {mae:.3f} | Val MAE: {val_mae:.3f} | R²: {r2:.3f}'
    else:
        title = f'{model_name} Performance on Test Set (30 Images)\nMAE: {mae:.3f} g/dL | R²: {r2:.3f}'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper left')
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Save plot with error handling
    plot_path = output_dir / f'test_evaluation_plot{suffix}.png'
    try:
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved plot to: {plot_path}")
    except PermissionError:
        # File might be open - try alternative name
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_plot_path = output_dir / f'test_evaluation_plot{suffix}_{timestamp}.png'
        plt.savefig(alt_plot_path, dpi=300, bbox_inches='tight')
        print(f"   ⚠️  Original plot file was locked")
        print(f"   ✓ Saved plot to: {alt_plot_path}")
    finally:
        plt.close()


def compare_model_results(results_list):
    """Compare results from multiple models"""
    print("\n" + "=" * 70)
    print("📊 MODEL COMPARISON ON TEST SET")
    print("=" * 70)
    
    comparison = []
    for result in results_list:
        comparison.append({
            'Model': result['model_name'],
            'MAE': result['mae'],
            'RMSE': result['rmse'],
            'R²': result['r2']
        })
    
    df = pd.DataFrame(comparison)
    print("\n" + df.to_string(index=False))
    
    # Find best model
    best_idx = df['MAE'].idxmin()
    best_model = df.iloc[best_idx]
    
    print("\n" + "=" * 70)
    print(f"🏆 BEST MODEL ON TEST SET: {best_model['Model']}")
    print("=" * 70)
    print(f"   MAE:  {best_model['MAE']:.3f} g/dL")
    print(f"   RMSE: {best_model['RMSE']:.3f} g/dL")
    print(f"   R²:   {best_model['R²']:.3f}")
    
    if best_model['MAE'] <= 0.8:
        print(f"\n🎉 TARGET ACHIEVED! MAE ≤ 0.8 g/dL")
    else:
        print(f"\n⚠️  Best MAE is {best_model['MAE']:.3f} g/dL (target: ≤ 0.8)")
    
    return df


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(
        description='Evaluate trained model(s) on test set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python 06_evaluate_on_test.py              # Evaluate default model
    python 06_evaluate_on_test.py --model rf   # Evaluate Random Forest
    python 06_evaluate_on_test.py --model gb   # Evaluate Gradient Boosting
    python 06_evaluate_on_test.py --model both # Compare both models
        """
    )
    parser.add_argument('--model', type=str, default=None, choices=['rf', 'gb', 'both'],
                       help='Model to evaluate: rf, gb, or both')
    args = parser.parse_args()
    
    try:
        if args.model == 'both':
            print("🔄 Evaluating both Random Forest and Gradient Boosting...\n")
            results_list = []
            
            for model_type in ['rf', 'gb']:
                results_df, mae, rmse, r2, model_name = evaluate_on_test_set(model_type)
                results_list.append({
                    'model_name': model_name,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'results_df': results_df
                })
                print("\n")
            
            # Compare results
            compare_model_results(results_list)
            
        else:
            results_df, mae, rmse, r2, model_name = evaluate_on_test_set(args.model)
        
        print("\n" + "=" * 70)
        print("✅ EVALUATION COMPLETE")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
