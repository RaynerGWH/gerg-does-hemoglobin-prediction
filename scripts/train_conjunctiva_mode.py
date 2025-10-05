import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import pickle
from pathlib import Path

# Load Kaggle conjunctiva data
df = pd.read_csv('data/external/kaggle_anemia/anemia_dataset.csv')

# Extract RGB features
X = df[['%Red Pixel', '%Green pixel', '%Blue pixel']].values
y = df['Hb'].values

print(f"Training on {len(X)} conjunctiva samples")
print(f"HgB range: {y.min():.1f} - {y.max():.1f} g/dL")

# Cross-validation to estimate performance
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_absolute_error')
cv_mae = -cv_scores.mean()

print(f"\nCross-validation MAE: {cv_mae:.2f} g/dL")

# Train on all data
model.fit(X, y)

# Save model
Path('weights').mkdir(exist_ok=True)
with open('weights/conjunctiva_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"\nModel saved to weights/conjunctiva_model.pkl")