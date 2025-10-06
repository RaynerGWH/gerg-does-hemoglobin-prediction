import pickle
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from pillow_heif import register_heif_opener
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

register_heif_opener()

def extract_relative_color_features(img_path):
    """Extract relative color features from image"""
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
    
    return [
        red_pct, green_pct, blue_pct,
        r_ratio, g_ratio, b_ratio,
        rg_ratio, rb_ratio, gb_ratio,
        red_dominance,
        r_norm, g_norm, b_norm
    ]

print("=" * 60)
print("TESTING ALL IMAGES")
print("=" * 60)

# Load model
print("\n📦 Loading model...")
with open('weights/final_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load labels
print("📋 Loading labels...")
df = pd.read_csv('data/starter/labels_valid.csv')

# Extract features and predict
print(f"🔍 Processing {len(df)} images...\n")
predictions = []
true_values = []
errors = []

print(f"{'#':<3} {'Filename':<50} {'True':>6} {'Pred':>6} {'Error':>6}")
print("-" * 73)

for idx, row in df.iterrows():
    filepath = row['filepath']
    true_hgb = row['hgb']
    
    # Extract features
    features = np.array([extract_relative_color_features(filepath)])
    
    # Predict
    pred_hgb = model.predict(features)[0]
    error = abs(true_hgb - pred_hgb)
    
    predictions.append(pred_hgb)
    true_values.append(true_hgb)
    errors.append(error)
    
    # Print result
    filename = Path(filepath).name
    print(f"{idx+1:<3} {filename:<50} {true_hgb:>6.1f} {pred_hgb:>6.1f} {error:>6.2f}")

# Convert to arrays
y_true = np.array(true_values)
y_pred = np.array(predictions)

# Calculate metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n" + "=" * 60)
print("PERFORMANCE METRICS")
print("=" * 60)
print(f"Mean Absolute Error (MAE):  {mae:.2f} g/dL")
print(f"Root Mean Squared Error:     {rmse:.2f} g/dL")
print(f"R² Score:                    {r2:.3f}")
print(f"Max Error:                   {max(errors):.2f} g/dL")
print(f"Min Error:                   {min(errors):.2f} g/dL")

# Save results
results = pd.DataFrame({
    'filename': [Path(fp).name for fp in df['filepath']],
    'true_hgb': y_true,
    'predicted_hgb': y_pred,
    'absolute_error': errors
})

# Create results directory
Path('results').mkdir(exist_ok=True)
results.to_csv('results/all_predictions.csv', index=False)
print(f"\n💾 Results saved to: results/all_predictions.csv")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: True vs Predicted
axes[0].scatter(y_true, y_pred, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('True HgB (g/dL)', fontsize=12)
axes[0].set_ylabel('Predicted HgB (g/dL)', fontsize=12)
axes[0].set_title(f'Predictions vs True Values\nMAE: {mae:.2f} g/dL, R²: {r2:.3f}', fontsize=14)
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: Error distribution
axes[1].hist(errors, bins=15, edgecolor='black', alpha=0.7, color='coral')
axes[1].axvline(mae, color='red', linestyle='--', linewidth=2, label=f'MAE: {mae:.2f}')
axes[1].set_xlabel('Absolute Error (g/dL)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Error Distribution', fontsize=14)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
viz_path = 'results/prediction_analysis.png'
plt.savefig(viz_path, dpi=150, bbox_inches='tight')
print(f"📊 Visualization saved to: {viz_path}")
plt.close()

print("\n" + "=" * 60)
print("✅ TESTING COMPLETE!")
print("=" * 60)
print("\nCheck:")
print("  - results/all_predictions.csv (detailed results)")
print("  - results/prediction_analysis.png (visualizations)")