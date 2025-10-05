import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import pickle
from pillow_heif import register_heif_opener

register_heif_opener()

def extract_rgb_percentages(img_path):
    """Extract RGB percentages"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    pixels = np.array(img)
    
    total = pixels.sum()
    red_pct = (pixels[:,:,0].sum() / total) * 100
    green_pct = (pixels[:,:,1].sum() / total) * 100
    blue_pct = (pixels[:,:,2].sum() / total) * 100
    
    return [red_pct, green_pct, blue_pct]

# Load Genesis lip data
print("Loading Genesis lip images...")
lip_df = pd.read_csv('data/starter/labels.csv')
lip_features = []
for filepath in lip_df['filepath']:
    rgb = extract_rgb_percentages(filepath)
    lip_features.append(rgb)

lip_X = np.array(lip_features)
lip_y = lip_df['hgb'].values

# Load Kaggle conjunctiva data
print("Loading Kaggle conjunctiva data...")
conj_df = pd.read_csv('data/external/kaggle_anemia/anemia_dataset.csv')
conj_X = conj_df[['%Red Pixel', '%Green pixel', '%Blue pixel']].values
conj_y = conj_df['Hb'].values

# Combine datasets
X_combined = np.vstack([lip_X, conj_X])
y_combined = np.concatenate([lip_y, conj_y])

print(f"\nCombined dataset:")
print(f"  Lip samples: {len(lip_X)}")
print(f"  Conjunctiva samples: {len(conj_X)}")
print(f"  Total: {len(X_combined)} samples")
print(f"  HgB range: {y_combined.min():.1f} - {y_combined.max():.1f} g/dL")

# Train with cross-validation
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
cv_scores = cross_val_score(model, X_combined, y_combined, cv=5, scoring='neg_mean_absolute_error')
cv_mae = -cv_scores.mean()
cv_std = cv_scores.std()

print(f"\n5-Fold Cross-Validation:")
print(f"  MAE: {cv_mae:.2f} ± {cv_std:.2f} g/dL")

# Train final model on all data
model.fit(X_combined, y_combined)

# Save model
Path('weights').mkdir(exist_ok=True)
with open('weights/final_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"\nModel saved to weights/final_model.pkl")

# Save combined training data record
combined_record = pd.DataFrame({
    'site': ['lip']*len(lip_X) + ['conjunctiva']*len(conj_X),
    'hgb': y_combined
})
combined_record.to_csv('data/combined_training_record.csv', index=False)
print("Training record saved to data/combined_training_record.csv")