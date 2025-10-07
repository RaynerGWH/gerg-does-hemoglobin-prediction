# Model Card: Hemoglobin Prediction from Lip Images

## Model Details

**Model Name**: Random Forest Hemoglobin Predictor  
**Model Type**: Random Forest Regressor (scikit-learn)  
**Model Size**: 0.44 MB (ONNX format)  
**Competition Track**: Edge-Lite (≤10 MB)  
**Date**: October 2025

### Model Architecture

- **Algorithm**: Random Forest Regressor
- **Number of Estimators**: 300 trees (ensemble of decision trees)
- **Max Depth**: 20
- **Min Samples Split**: 5
- **Min Samples Leaf**: 2
- **Max Features**: sqrt (randomly selects features at each split)
- **Bootstrap**: True (samples with replacement)
- **Random State**: 42 (for reproducibility)

### Input Features

The model uses **28 handcrafted features** extracted from lip images:

**Color Features (13)**:

- RGB percentages (absolute): `red_pct`, `green_pct`, `blue_pct`
- RGB ratios (lighting-independent): `r_ratio`, `g_ratio`, `b_ratio`
- Inter-channel ratios: `rg_ratio`, `rb_ratio`, `gb_ratio`
- Color dominance: `red_dominance`, `redness_index`
- Normalized values: `r_norm`, `g_norm`, `b_norm`

**Image Quality Features (5)**:

- `brightness`: Average brightness (0-255 scale)
- `contrast`: RMS contrast
- `blur_score`: Laplacian variance (sharpness)
- `saturation`: Average HSV saturation
- `lighting_uniformity`: Standard deviation of grayscale

**Color Distribution Features (9)**:

- Statistical moments per RGB channel (mean, std, skewness)
- `r_mean`, `r_std`, `r_skew`
- `g_mean`, `g_std`, `g_skew`
- `b_mean`, `b_std`, `b_skew`

### Export Format

- **Primary Format**: ONNX (Open Neural Network Exchange)
- **Runtime**: ONNX Runtime
- **Opset Version**: 12
- **Components**:
  - `model.onnx`: Trained Random Forest model (0.44 MB)
  - `scaler.onnx`: StandardScaler for feature normalization

---

## Training Data

### Dataset Composition

**Total Training Samples**: 200 samples

- **Training Set**: 200 samples (augmented lip images)
- **Validation Set**: 7 samples (original lip images)

The current model was trained on a carefully selected subset of augmented lip images to balance model complexity and generalization. This smaller, curated dataset helps prevent overfitting while maintaining good performance.

### Preprocessing

- All images resized to 224×224 pixels
- RGB color space
- StandardScaler normalization applied to all features
- Feature extraction: Color statistics, image quality metrics, distribution moments

---

## Performance

### Metrics (Current Model)

- MAE: 0.919 g/dL
- RMSE: 1.443 g/dL
- R²: 0.796

### Target Performance

- **Competition Target**: MAE ≤ 0.8 g/dL
- **Status**: ⚠️ Current validation MAE exceeds target (further optimization needed)

### Known Limitations

1. **Limited Training Data**: Only 200 training samples may not capture full variability
2. **Validation Set Size**: Small validation set (7 samples) may not fully represent test performance
3. **Feature Engineering**: Handcrafted features may not capture all relevant biological signals
4. **Lighting Sensitivity**: Performance may vary with different lighting conditions and camera quality

---

## Intended Use

### Primary Use Case

Non-invasive hemoglobin level prediction from smartphone images of lips for anemia screening in resource-limited settings.

### Target Users

- Healthcare workers in low-resource environments
- Mobile health applications
- Point-of-care screening tools

### Out-of-Scope Uses

- **NOT for clinical diagnosis**: Should not replace laboratory blood tests
- **NOT for treatment decisions**: Results require confirmation with standard methods
- **NOT for diverse populations**: Model trained on limited demographic diversity

---

## Ethical Considerations

### Fairness & Bias

- **Demographic Bias**: Training data may not represent diverse skin tones, ages, and genders
- **Geographic Bias**: Limited to specific populations/regions
- **Device Bias**: Performance may vary across different smartphone cameras and lighting conditions

### Privacy

- Images contain biometric information (lip/face features)
- Proper data anonymization and consent required
- Metadata (device info, demographics) should be handled securely

### Safety Concerns

- **False Negatives**: Missing anemia cases could delay critical treatment
- **False Positives**: Unnecessary interventions and patient anxiety
- **Clinical Validation**: Model requires extensive clinical validation before deployment

---

## Training & Compute

### Training Environment

- **Framework**: scikit-learn 1.3.0
- **Export**: skl2onnx for ONNX conversion
- **Hardware**: Standard CPU (no GPU required)
- **Training Time**: < 5 minutes

### Inference

- **Runtime**: ONNX Runtime
- **Latency**: ~50-100ms per image (CPU)
- **Edge Deployment**: Suitable for mobile/edge devices (0.44 MB model)

---

## Maintenance & Updates

### Model Versioning

- **Current Version**: v1.0
- **Last Updated**: October 2025

### Future Improvements

1. **Data Collection**: Gather more diverse training samples
2. **Feature Engineering**: Explore deep learning features (CNN embeddings)
3. **Regularization**: Stronger techniques to reduce overfitting
4. **Calibration**: Post-processing to improve probability estimates
5. **Clinical Validation**: Field testing with medical professionals


### Technical Documentation

- ONNX Format: https://onnx.ai/
- scikit-learn Random Forest: https://scikit-learn.org/stable/modules/ensemble.html#forest

---

**Disclaimer**: This model is for research and educational purposes. It has not been approved for clinical use. Always consult qualified healthcare professionals for medical decisions.
