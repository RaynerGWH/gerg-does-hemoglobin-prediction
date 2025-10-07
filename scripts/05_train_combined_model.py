"""
Combined Model Training with Enhanced Features
Trains on augmented lip data + conjunctiva data with optional CNN features
Run from project root: python scripts/05_train_combined_mode    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    X_train_lip, y_train_lip, X_val_lip, y_val_lip, X_conj, y_conj, feature_names = load_features_and_labels(
        use_augmented=not args.no_augmentation,
        use_cnn=args.use_cnn,
        cnn_model=args.cnn_model
    )
    
    # Combine training data: augmented lip images + conjunctiva
    X_train = np.vstack([X_train_lip, X_conj])
    y_train = np.concatenate([y_train_lip, y_conj])
    
    # Validation data: only original lip images (NO conjunctiva)
    X_val = X_val_lip
    y_val = y_val_lip
    
    print(f"\n📈 Dataset composition:")
    print(f"   TRAINING:")
    print(f"     Lip samples (augmented): {len(X_train_lip)}")
    print(f"     Conjunctiva samples: {len(X_conj)}")
    print(f"     Total training: {len(X_train)}")
    print(f"   VALIDATION:")
    print(f"     Lip samples (original, no augmentation): {len(X_val)}")
    print(f"   Features per sample: {X_train.shape[1]}")
    print(f"   Training HgB range: {y_train.min():.1f} - {y_train.max():.1f} g/dL")
    print(f"   Validation HgB range: {y_val.min():.1f} - {y_val.max():.1f} g/dL")cnn-model resnet18]
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
import pickle
import argparse
import warnings
warnings.filterwarnings('ignore')


def load_features_and_labels(use_augmented=True, use_cnn=False, cnn_model='resnet18'):
    """
    Load all features and labels with train/val split
    
    Args:
        use_augmented: Use augmented training data
        use_cnn: Include CNN features
        cnn_model: Which CNN model features to use
    
    Returns:
        X_train, y_train, X_val, y_val, X_conj, y_conj, feature_names
    """
    processed_dir = Path('../data/processed')
    
    # Load lip features with train/val split
    if use_augmented and (processed_dir / 'train_features.npy').exists():
        print("📊 Loading augmented lip data with train/val split...")
        X_train_lip = np.load(processed_dir / 'train_features.npy')
        y_train_lip = np.load(processed_dir / 'train_labels.npy')
        X_val_lip = np.load(processed_dir / 'val_features.npy')
        y_val_lip = np.load(processed_dir / 'val_labels.npy')
        print(f"   Training samples (augmented): {len(X_train_lip)}")
        print(f"   Validation samples (original): {len(X_val_lip)}")
    else:
        print("📊 Loading original lip data...")
        X_lip = np.load(processed_dir / 'enhanced_features.npy')
        labels_df = pd.read_csv(processed_dir / 'labels_valid.csv')
        y_lip = labels_df['hgb'].values
        
        # Manual split if augmented data not available
        from sklearn.model_selection import train_test_split
        X_train_lip, X_val_lip, y_train_lip, y_val_lip = train_test_split(
            X_lip, y_lip, test_size=0.2, random_state=42
        )
        print(f"   Training samples: {len(X_train_lip)}")
        print(f"   Validation samples: {len(X_val_lip)}")
    
    # Load conjunctiva features
    print("📊 Loading conjunctiva data...")
    X_conj = np.load(processed_dir / 'conjunctiva_features.npy')
    y_conj = np.load(processed_dir / 'conjunctiva_labels.npy')
    print(f"   Conjunctiva samples: {len(X_conj)}")
    
    # Load feature names
    feature_names_path = processed_dir / 'feature_names.txt'
    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip() for line in f]
    
    # Optionally add CNN features
    if use_cnn:
        cnn_features_path = processed_dir / f'cnn_features_{cnn_model}.npy'
        if cnn_features_path.exists():
            print(f"📊 Loading CNN features ({cnn_model})...")
            X_cnn = np.load(cnn_features_path)
            
            # CNN features are only for original lip images
            print("   ⚠️  CNN features not available for augmented/conjunctiva images")
            print("   Using handcrafted features only")
        else:
            print(f"   ⚠️  CNN features not found at {cnn_features_path}")
            print("   Using handcrafted features only")
    
    return X_train_lip, y_train_lip, X_val_lip, y_val_lip, X_conj, y_conj, feature_names


def train_and_evaluate(X_train, y_train, X_val, y_val, feature_names, model_type='rf'):
    """
    Train model and evaluate on separate validation set
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        feature_names: Names of features
        model_type: 'rf' (Random Forest), 'gb' (Gradient Boosting), 'ridge'
    
    Returns:
        Trained model, scaler, train_metrics, val_metrics
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Select model
    if model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        print(f"   📋 Using Random Forest")
        print(f"      n_estimators=300, max_depth=20, max_features='sqrt'")
    elif model_type == 'gb':
        # More conservative GB settings to prevent overfitting
        model = GradientBoostingRegressor(
            n_estimators=100,           # Reduced from 200
            max_depth=4,                # Reduced from 8 (shallower trees)
            learning_rate=0.01,         # Reduced from 0.05 (slower learning)
            min_samples_split=10,       # Increased from 5 (more regularization)
            min_samples_leaf=5,         # Increased from 2 (more regularization)
            subsample=0.8,              # Add subsampling (80% of data per tree)
            max_features='sqrt',        # Limit features per split
            validation_fraction=0.1,    # Use 10% for early stopping
            n_iter_no_change=10,        # Stop if no improvement for 10 iterations
            random_state=42
        )
        print(f"   📋 Using REGULARIZED Gradient Boosting (anti-overfitting)")
        print(f"      max_depth=4, learning_rate=0.01, subsample=0.8")
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    print(f"\n🔨 Training {model_type.upper()} model on {len(X_train)} samples...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on training set
    y_train_pred = model.predict(X_train_scaled)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    print(f"\n📊 Training Set Performance:")
    print(f"   MAE: {train_mae:.3f} g/dL")
    print(f"   R²: {train_r2:.3f}")
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val_scaled)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"\n📊 Validation Set Performance (INDEPENDENT):")
    print(f"   MAE: {val_mae:.3f} g/dL")
    print(f"   R²: {val_r2:.3f}")
    
    # OOB score for Random Forest
    if model_type == 'rf' and hasattr(model, 'oob_score_'):
        print(f"   OOB Score: {model.oob_score_:.3f}")
    
    if val_mae <= 0.8:
        print(f"\n🎉 TARGET ACHIEVED on validation set! (MAE ≤ 0.8 g/dL)")
    else:
        print(f"\n⚠️  Validation MAE > 0.8 g/dL (target not met yet)")
    
    # Check for overfitting
    overfitting_ratio = val_mae / train_mae
    if train_mae < val_mae * 0.7:
        print(f"\n⚠️  WARNING: Possible overfitting detected")
        print(f"   Training MAE ({train_mae:.3f}) is much lower than validation MAE ({val_mae:.3f})")
        print(f"   Overfitting ratio: {overfitting_ratio:.1f}x")
    
    # Feature importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        
        print(f"\n📊 Top 10 Most Important Features:")
        for i in range(min(10, len(sorted_idx))):
            idx = sorted_idx[i]
            if idx < len(feature_names):
                print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    train_metrics = {'mae': train_mae, 'r2': train_r2}
    val_metrics = {'mae': val_mae, 'r2': val_r2}
    
    return model, scaler, train_metrics, val_metrics


def compare_models(results):
    """Compare multiple trained models"""
    print("\n" + "=" * 70)
    print("📊 MODEL COMPARISON")
    print("=" * 70)
    
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    
    # Find best model by validation MAE
    best_idx = df['val_mae'].idxmin()
    best_model = df.iloc[best_idx]
    
    print("\n" + "=" * 70)
    print(f"🏆 BEST MODEL: {best_model['model_type'].upper()}")
    print("=" * 70)
    print(f"   Validation MAE: {best_model['val_mae']:.3f} g/dL")
    print(f"   Validation R²:  {best_model['val_r2']:.3f}")
    print(f"   Overfitting ratio: {best_model['val_mae'] / best_model['train_mae']:.1f}x")


def main():
    """Main training pipeline"""
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cnn', action='store_true',
                       help='Include CNN features')
    parser.add_argument('--cnn-model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet50', 'mobilenet_v2'],
                       help='CNN model to use')
    parser.add_argument('--model', type=str, default='gb',
                       choices=['rf', 'gb', 'ridge', 'both'],
                       help='Model type: rf (Random Forest), gb (Gradient Boosting), ridge, both (train and compare)')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Do not use augmented data')
    args = parser.parse_args()
    
    print("=" * 70)
    print("COMBINED MODEL TRAINING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model.upper()}")
    print(f"  Use augmentation: {not args.no_augmentation}")
    print(f"  Use CNN features: {args.use_cnn}")
    if args.use_cnn:
        print(f"  CNN model: {args.cnn_model}")
    
    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    X_train_lip, y_train_lip, X_val_lip, y_val_lip, X_conj, y_conj, feature_names = load_features_and_labels(
        use_augmented=not args.no_augmentation,
        use_cnn=args.use_cnn,
        cnn_model=args.cnn_model
    )
    
    # Combine training data: augmented lip images + conjunctiva
    X_train = np.vstack([X_train_lip, X_conj])
    y_train = np.concatenate([y_train_lip, y_conj])
    
    # Validation data: only original lip images (NO conjunctiva)
    X_val = X_val_lip
    y_val = y_val_lip
    
    print(f"\n📈 Dataset composition:")
    print(f"   TRAINING:")
    print(f"     Lip samples (augmented): {len(X_train_lip)}")
    print(f"     Conjunctiva samples: {len(X_conj)}")
    print(f"     Total training: {len(X_train)}")
    print(f"   VALIDATION:")
    print(f"     Lip samples (original, no augmentation): {len(X_val)}")
    print(f"   Features per sample: {X_train.shape[1]}")
    print(f"   Training HgB range: {y_train.min():.1f} - {y_train.max():.1f} g/dL")
    print(f"   Validation HgB range: {y_val.min():.1f} - {y_val.max():.1f} g/dL")
    
    # Train model(s)
    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)
    
    if args.model == 'both':
        print("\n🔄 Training both Random Forest and Gradient Boosting for comparison...\n")
        results = []
        models_data = {}
        
        for model_type in ['rf', 'gb']:
            print(f"\n{'='*70}")
            print(f"Training {model_type.upper()}")
            print(f"{'='*70}")
            
            model, scaler, train_metrics, val_metrics = train_and_evaluate(
                X_train, y_train, X_val, y_val, feature_names, model_type
            )
            
            models_data[model_type] = {
                'model': model,
                'scaler': scaler,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            
            results.append({
                'model_type': model_type.upper(),
                'train_mae': train_metrics['mae'],
                'val_mae': val_metrics['mae'],
                'train_r2': train_metrics['r2'],
                'val_r2': val_metrics['r2']
            })
        
        # Compare models
        compare_models(results)
        
        # Save BOTH models with different names
        print(f"\n💾 Saving both models for comparison...")
        weights_dir = Path('../weights')
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        for model_type in ['rf', 'gb']:
            model_subdir = weights_dir / model_type
            model_subdir.mkdir(parents=True, exist_ok=True)
            
            model_path_type = model_subdir / 'final_model.pkl'
            scaler_path_type = model_subdir / 'feature_scaler.pkl'
            config_path_type = model_subdir / 'model_config.txt'
            
            with open(model_path_type, 'wb') as f:
                pickle.dump(models_data[model_type]['model'], f)
            
            with open(scaler_path_type, 'wb') as f:
                pickle.dump(models_data[model_type]['scaler'], f)
            
            # Save configuration
            with open(config_path_type, 'w') as f:
                f.write(f"Model type: {model_type}\n")
                f.write(f"Use augmentation: {not args.no_augmentation}\n")
                f.write(f"Use CNN: {args.use_cnn}\n")
                if args.use_cnn:
                    f.write(f"CNN model: {args.cnn_model}\n")
                f.write(f"Features: {len(feature_names)}\n")
                f.write(f"Training samples: {len(X_train)}\n")
                f.write(f"Validation samples: {len(X_val)}\n")
                f.write(f"Training MAE: {models_data[model_type]['train_metrics']['mae']:.3f}\n")
                f.write(f"Validation MAE: {models_data[model_type]['val_metrics']['mae']:.3f}\n")
            
            print(f"   ✓ Saved {model_type.upper()}: {model_subdir}")
        
        # Also save best model to default location
        best_idx = min(range(len(results)), key=lambda i: results[i]['val_mae'])
        best_model_type = ['rf', 'gb'][best_idx]
        
        print(f"\n💾 Saving best model ({best_model_type.upper()}) to default location...")
        model = models_data[best_model_type]['model']
        scaler = models_data[best_model_type]['scaler']
        train_metrics = models_data[best_model_type]['train_metrics']
        val_metrics = models_data[best_model_type]['val_metrics']
        selected_model_type = best_model_type
        
    else:
        model, scaler, train_metrics, val_metrics = train_and_evaluate(
            X_train, y_train, X_val, y_val, feature_names, args.model
        )
        selected_model_type = args.model
    
    # Save model and scaler
    weights_dir = Path('../weights')
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = weights_dir / 'final_model.pkl'
    scaler_path = weights_dir / 'feature_scaler.pkl'
    config_path = weights_dir / 'model_config.txt'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save configuration
    with open(config_path, 'w') as f:
        f.write(f"Model type: {selected_model_type}\n")
        f.write(f"Use augmentation: {not args.no_augmentation}\n")
        f.write(f"Use CNN: {args.use_cnn}\n")
        if args.use_cnn:
            f.write(f"CNN model: {args.cnn_model}\n")
        f.write(f"Features: {len(feature_names)}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Validation samples: {len(X_val)}\n")
        f.write(f"Training MAE: {train_metrics['mae']:.3f}\n")
        f.write(f"Validation MAE: {val_metrics['mae']:.3f}\n")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"📁 Saved files:")
    print(f"   Model: {model_path}")
    print(f"   Scaler: {scaler_path}")
    print(f"   Config: {config_path}")
    print(f"\n🎯 Final Performance:")
    print(f"   Training MAE: {train_metrics['mae']:.3f} g/dL (R² = {train_metrics['r2']:.3f})")
    print(f"   Validation MAE: {val_metrics['mae']:.3f} g/dL (R² = {val_metrics['r2']:.3f})")
    
    if val_metrics['mae'] <= 0.8:
        print(f"\n🎉 TARGET ACHIEVED on validation! MAE ≤ 0.8 g/dL")
        print(f"   Next: Run 06_evaluate_on_test.py to evaluate on ALL 30 images")
    else:
        print(f"\n⚠️  Validation MAE is {val_metrics['mae']:.3f} g/dL (target: ≤ 0.8)")
        print(f"   Consider trying:")
        if args.model != 'both':
            print(f"     - Compare models: --model both")
        if selected_model_type == 'rf':
            print(f"     - Try Gradient Boosting: --model gb")
        else:
            print(f"     - Try Random Forest: --model rf")
        print(f"     - Add CNN features: --use-cnn --cnn-model resnet50")
        print(f"     - Check image quality in validation set")


if __name__ == '__main__':
    main()
