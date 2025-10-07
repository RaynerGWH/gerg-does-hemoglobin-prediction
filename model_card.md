# Model Card: Hemoglobin Prediction from Lip Images

## Model Details

**Model Name**: Gradient Boosting Hemoglobin Predictor  
**Model Type**: Gradient Boosting Regressor (scikit-learn)  
**Model Size**: 0.44 MB (ONNX format)  
**Competition Track**: Edge-Lite (≤10 MB)  
**Date**: October 2025  

### Model Architecture
- **Algorithm**: Gradient Boosting Regressor
- **Number of Estimators**: 100 trees
- **Max Depth**: 4
- **Learning Rate**: 0.01
- **Subsample**: 0.8
- **Min Samples Split**: 10
- **Min Samples Leaf**: 5
- **Max Features**: sqrt
- **Regularization**: Heavy (to prevent overfitting)

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
  - `model.onnx`: Trained Gradient Boosting model (0.44 MB)
  - `scaler.onnx`: StandardScaler for feature normalization

---

## Training Data

### Dataset Composition
**Total Training Samples**: ~1,620 samples

1. **Augmented Lip Images** (~1,080 samples)
   - 30 original lip images from starter dataset
   - 3x augmentation per image (rotation, brightness, contrast adjustments)
   - 36 samples per original image

2. **Conjunctiva Dataset** (~540 samples)
   - External dataset with conjunctiva images
   - Used for additional training diversity

### Data Split
- **Training Set**: ~1,440 samples (augmented + conjunctiva)
- **Validation Set**: ~180 samples (30 original images, 6x augmented)

### Preprocessing
- All images resized to 224×224 pixels
- RGB color space
- StandardScaler normalization applied to all features
- Feature extraction: Color statistics, image quality metrics, distribution moments

---

## Performance

### Metrics (Current Model)
- **Training MAE**: 0.043 g/dL
- **Validation MAE**: 1.416 g/dL
- **Training R²**: High (near 1.0)
- **Validation R²**: Moderate

### Target Performance
- **Competition Target**: MAE ≤ 0.8 g/dL
- **Status**: ⚠️ Current validation MAE exceeds target (overfitting observed)

### Known Limitations
1. **Overfitting**: Large gap between training (0.043) and validation (1.416) MAE
2. **Generalization**: Model memorizes training data but struggles with unseen samples
3. **Data Efficiency**: Limited original samples (30 images) leads to overfitting despite augmentation
4. **Feature Limitations**: Handcrafted features may not capture all relevant biological signals

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

### Contact
For questions, issues, or collaboration:
- GitHub: [RaynerGWH/gerg-does-hemoglobin-prediction](https://github.com/RaynerGWH/gerg-does-hemoglobin-prediction)

---

## References

### Related Work
- Non-invasive anemia detection using smartphone imaging
- Computer vision for healthcare in low-resource settings
- Hemoglobin estimation from conjunctival images

### Technical Documentation
- ONNX Format: https://onnx.ai/
- scikit-learn Gradient Boosting: https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting

---

**Disclaimer**: This model is for research and educational purposes. It has not been approved for clinical use. Always consult qualified healthcare professionals for medical decisions.
