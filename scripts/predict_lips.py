import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error

# Load trained conjunctiva model
with open('weights/conjunctiva_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load lip image features
X_lips = np.load('data/starter/lip_rgb_features.npy')

# Load true Hb values
df = pd.read_csv('data/starter/labels.csv')
y_true = df['hgb'].values

# Predict
y_pred = model.predict(X_lips)

# Calculate MAE
mae = mean_absolute_error(y_true, y_pred)

print(f"Predictions on lip images (trained on conjunctiva):")
print(f"MAE: {mae:.2f} g/dL")
print(f"\nSample predictions:")
for i in range(min(5, len(y_pred))):
    print(f"  True: {y_true[i]:.1f} g/dL, Predicted: {y_pred[i]:.1f} g/dL")

# Save results
results = pd.DataFrame({
    'image_id': df['image_id'],
    'true_hgb': y_true,
    'predicted_hgb': y_pred,
    'error': np.abs(y_true - y_pred)
})
results.to_csv('results/lip_predictions.csv', index=False)
print(f"\nFull results saved to results/lip_predictions.csv")