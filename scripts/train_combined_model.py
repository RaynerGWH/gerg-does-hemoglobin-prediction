import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import pickle
from pathlib import Path

# Load Genesis lip data (with relative features)
print("Loading Genesis lip images with relative features...")
lip_X = np.load('data/starter/relative_features.npy')
lip_df = pd.read_csv('data/starter/labels_valid.csv')
lip_y = lip_df['hgb'].values

print(f"Lip features shape: {lip_X.shape}")

# Load Kaggle conjunctiva data (only RGB percentages available)
print("Loading Kaggle conjunctiva data...")
conj_df = pd.read_csv('data/external/kaggle_anemia/anemia_dataset.csv')

# Extract basic RGB percentages from Kaggle data
conj_rgb = conj_df[['%Red Pixel', '%Green pixel', '%Blue pixel']].values

# Calculate relative features for Kaggle data to match lip features
print("Computing relative features for conjunctiva data...")
conj_features = []
for rgb_pcts in conj_rgb:
    r_pct, g_pct, b_pct = rgb_pcts
    
    # Convert percentages back to approximate mean values (rough estimate)
    # This is approximate but maintains relative relationships
    total = 100
    r_ratio = r_pct / total
    g_ratio = g_pct / total
    b_ratio = b_pct / total
    
    # Estimate mean RGB (scale to 0-255 range)
    scale = 255 / 3  # Approximate scaling
    mean_r = r_pct * scale / 100
    mean_g = g_pct * scale / 100
    mean_b = b_pct * scale / 100
    
    # Calculate color ratios
    rg_ratio = mean_r / (mean_g + 1e-6)
    rb_ratio = mean_r / (mean_b + 1e-6)
    gb_ratio = mean_g / (mean_b + 1e-6)
    
    # Color dominance
    red_dominance = mean_r - mean_b
    
    # Normalized
    r_norm = mean_r / 255
    g_norm = mean_g / 255
    b_norm = mean_b / 255
    
    # Match the 13-feature format from extract_relative_features.py
    features = [
        r_pct, g_pct, b_pct,           # Absolute RGB%
        r_ratio, g_ratio, b_ratio,      # Relative ratios
        rg_ratio, rb_ratio, gb_ratio,   # Color ratios
        red_dominance,                  # Dominance
        r_norm, g_norm, b_norm          # Normalized
    ]
    conj_features.append(features)

conj_X = np.array(conj_features)
conj_y = conj_df['Hb'].values

print(f"Conjunctiva features shape: {conj_X.shape}")

# Combine datasets
X_combined = np.vstack([lip_X, conj_X])
y_combined = np.concatenate([lip_y, conj_y])

print(f"\n{'='*60}")
print("COMBINED DATASET:")
print(f"{'='*60}")
print(f"  Lip samples: {len(lip_X)} (with 13 relative features)")
print(f"  Conjunctiva samples: {len(conj_X)} (computed 13 features)")
print(f"  Total: {len(X_combined)} samples")
print(f"  Feature dimensions: {X_combined.shape[1]} features")
print(f"  HgB range: {y_combined.min():.1f} - {y_combined.max():.1f} g/dL")

# Train with cross-validation
print(f"\n{'='*60}")
print("TRAINING MODEL:")
print(f"{'='*60}")

model = RandomForestRegressor(
    n_estimators=200,      # Increased for better performance
    max_depth=15,          # Deeper trees for complex relationships
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)

cv_scores = cross_val_score(
    model, X_combined, y_combined, 
    cv=5, 
    scoring='neg_mean_absolute_error'
)
cv_mae = -cv_scores.mean()
cv_std = cv_scores.std()

print(f"\n5-Fold Cross-Validation Results:")
print(f"  MAE: {cv_mae:.2f} ± {cv_std:.2f} g/dL")
print(f"  Individual fold MAEs: {[-score for score in cv_scores]}")

# Train final model on all data
print(f"\nTraining final model on all {len(X_combined)} samples...")
model.fit(X_combined, y_combined)

# Feature importance
feature_names = [
    'Red%', 'Green%', 'Blue%',
    'R_ratio', 'G_ratio', 'B_ratio',
    'RG_ratio', 'RB_ratio', 'GB_ratio',
    'Red_dominance',
    'R_norm', 'G_norm', 'B_norm'
]
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print(f"\nTop 5 Most Important Features:")
for i in range(min(5, len(sorted_idx))):
    idx = sorted_idx[i]
    print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.3f}")

# Save model
Path('weights').mkdir(exist_ok=True)
with open('weights/final_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"\n{'='*60}")
print("✅ MODEL SAVED")
print(f"{'='*60}")
print(f"Model saved to: weights/final_model.pkl")

# Save combined training data record
combined_record = pd.DataFrame({
    'site': ['lip']*len(lip_X) + ['conjunctiva']*len(conj_X),
    'hgb': y_combined
})
combined_record.to_csv('data/combined_training_record.csv', index=False)
print(f"Training record saved to: data/combined_training_record.csv")

print(f"\n{'='*60}")
print("NEXT STEPS:")
print(f"{'='*60}")
print("1. Test inference: python inference_relative.py <image_path>")
print("2. Check results in weights/ folder")
print("3. Analyze feature importance above")