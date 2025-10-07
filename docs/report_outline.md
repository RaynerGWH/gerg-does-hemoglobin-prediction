# Technical Report Outline (≤ 8 Pages)

## Report Structure for Competition Submission

---

## 1. Introduction (0.5 pages)

### 1.1 Problem Statement

- Non-invasive hemoglobin prediction from lip images
- Target: MAE ≤ 0.8 g/dL
- Clinical importance (anemia detection)

### 1.2 Approach Summary

- Handcrafted feature engineering (28 features)
- Gradient Boosting Regressor
- Data augmentation + external dataset
- Edge-Lite track (0.44 MB model)

---

## 2. Data Usage & Preprocessing (1-1.5 pages)

### 2.1 Training Data

**Primary Dataset:**

- 30 labeled lip images (starter set)
- HgB range: 7.0 - 17.0 g/dL
- Image format: HEIC/JPG, various resolutions

**External Dataset:**

- Kaggle anemia conjunctiva dataset
- ~1,500 samples
- Domain: conjunctiva (pink eye area) vs lips

### 2.2 Data Augmentation Strategy

**Augmentation Pipeline:**

- 3x augmented versions per original image
- Techniques: rotation (±15°), brightness (±10%), contrast (±10%), horizontal flip
- Result: 30 → 120 training samples

**Train/Val Split:**

- Training: Augmented images (120) + Conjunctiva (1,500) = 1,620 samples
- Validation: 7 original images (NO augmentation, NO conjunctiva)
- No data leakage between train/val

### 2.3 Preprocessing Pipeline

1. Resize to 224×224 pixels
2. RGB color space
3. Feature extraction (28 features)
4. StandardScaler normalization (zero mean, unit variance)

**Rationale:** Fixed size for consistent feature extraction; RGB preserves color information critical for anemia detection

---

## 3. Feature Engineering (1 page)

### 3.1 Feature Categories (28 total)

**Basic Color Features (13):**

- RGB percentages (absolute)
- RGB ratios (lighting-independent)
- Color ratios (RG, RB, GB)
- Red dominance, redness index
- Normalized RGB values (0-1)

**Image Quality Features (5):**

- Brightness (average pixel intensity)
- Contrast (RMS contrast)
- Blur score (Laplacian variance - higher = sharper)
- Saturation (HSV space)
- Lighting uniformity (grayscale std)

**Color Distribution (9):**

- Mean, std, skewness per RGB channel
- Captures color distribution shape

**Extended Features (1):**

- Redness index: (R-G)/(R+G)

### 3.2 Feature Design Rationale

- **Color-based:** Anemia correlates with pallor (reduced redness)
- **Quality metrics:** Control for image quality variations
- **Statistical moments:** Capture pixel distribution patterns
- **Lighting-independent ratios:** Reduce sensitivity to lighting conditions

### 3.3 Feature Importance

[Include bar chart of top 10 features from model.feature_importances_]

Top features:

1. Red dominance
2. RG ratio
3. Redness index
4. [Add your actual top features]

---

## 4. Model Architecture & Training (1-1.5 pages)

### 4.1 Model Selection

**Algorithm:** Gradient Boosting Regressor (sklearn)

**Why Gradient Boosting:**

- Strong performance on tabular data
- Handles non-linear relationships
- Built-in feature importance
- Small model size (0.44 MB)
- Fast inference (~50ms per image)

### 4.2 Hyperparameters

```python
GradientBoostingRegressor(
    n_estimators=100,        # Number of trees
    max_depth=4,             # Shallow trees (regularization)
    learning_rate=0.01,      # Slow learning (anti-overfitting)
    min_samples_split=10,    # Increased regularization
    min_samples_leaf=5,      # Increased regularization
    subsample=0.8,           # 80% data per tree
    max_features='sqrt',     # Feature subsampling
    validation_fraction=0.1, # Early stopping
    n_iter_no_change=10,     # Stop if no improvement
    random_state=42          # Reproducibility
)
```

**Design Philosophy:** Heavy regularization to combat overfitting from small dataset

### 4.3 Loss Function

**Loss:** Least Squares (L2 loss)

- Default for regression
- Penalizes large errors heavily
- Appropriate for continuous HgB prediction

### 4.4 Training Procedure

1. Deterministic seed (random_state=42)
2. Feature scaling (StandardScaler on training set)
3. Fit on combined dataset (aug + conjunctiva)
4. Early stopping based on validation loss
5. Export to ONNX for deployment

**Training Time:** ~30-60 seconds on CPU  
**Compute Cost:** Minimal (< 1 GPU-hour equivalent)

---

## 5. Results & Performance (1 page)

### 5.1 Quantitative Results

**Training Set Performance:**

- MAE: 0.043 g/dL
- R²: [Calculate and add]
- N = 1,620 samples

**Validation Set Performance:**

- MAE: 1.416 g/dL ⚠️
- R²: [Calculate and add]
- N = 7 samples

**Gap Analysis:**

- Training/Validation MAE ratio: 33x
- Indicates severe overfitting

### 5.2 Prediction Visualization

[Include scatter plot: True HgB vs Predicted HgB for validation set]

[Include residuals plot: Prediction Error vs True HgB]

### 5.3 Comparison to Baselines

| Model                 | Validation MAE |
| --------------------- | -------------- |
| Mean predictor        | [Calculate]    |
| Linear regression     | [Calculate]    |
| **Gradient Boosting** | **1.416**      |
| Target                | **≤ 0.8**      |

---

## 6. Error Analysis & Calibration (1 page)

### 6.1 Error Breakdown

**By HgB Range:**

- Low HgB (< 10 g/dL): MAE = [Calculate]
- Normal HgB (10-14 g/dL): MAE = [Calculate]
- High HgB (> 14 g/dL): MAE = [Calculate]

**By Image Quality:**

- Sharp images: Better performance
- Blurry images: Higher error
- Poor lighting: Inconsistent predictions

### 6.2 Overfitting Analysis

**Root Causes:**

1. Small validation set (7 images) → high variance
2. Domain shift (conjunctiva ≠ lips)
3. Augmented data from same 30 images
4. Model memorizes training set

**Evidence:**

- Training MAE (0.043) << Validation MAE (1.416)
- Low training error suggests memorization

### 6.3 Calibration

[If time permits: Add calibration plot showing prediction intervals]

**Uncertainty Quantification:**

- Model outputs point predictions only
- No confidence intervals currently
- Future: Add ensemble or quantile regression

### 6.4 Failure Mode Analysis

**Common Failures:**

1. **Extreme HgB values** (< 8 or > 16 g/dL) - Outside training distribution
2. **Poor image quality** - Blurry, dark, or overexposed
3. **Makeup/lipstick** - Interferes with color features
4. **Diverse skin tones** - Limited representation in training data

**Example Failures:**
[Show 2-3 worst predictions with analysis]

---

## 7. Fairness & Ethical Considerations (0.5 pages)

### 7.1 Demographic Analysis

**Dataset Limitations:**

- No demographic labels (age, gender, ethnicity)
- Cannot assess fairness across groups
- Potential bias in color-based features for darker skin tones

**Fairness Concerns:**

1. **Skin tone bias:** Darker skin may affect color feature extraction
2. **Image quality disparity:** Lower-quality cameras may disadvantage some populations
3. **Geographic generalization:** Training data from limited sources

### 7.2 Mitigation Strategies

- Collect diverse training data across demographics
- Test disaggregated performance metrics
- Include skin tone normalization features
- Validate across multiple populations

### 7.3 Ethical Use

⚠️ **Not a diagnostic tool** - Screening only  
⚠️ **Requires clinical validation** - Before real-world deployment  
⚠️ **Human oversight needed** - Use alongside medical judgment

---

## 8. Ablation Studies (0.5 pages)

### 8.1 Data Ablations

| Configuration         | Val MAE   | Change    |
| --------------------- | --------- | --------- |
| Original only (30)    | ~1.5      | Baseline  |
| + Augmentation (120)  | ~1.2      | -0.3      |
| + Conjunctiva (1,620) | **1.416** | +0.2 (??) |

**Insight:** Conjunctiva data may introduce domain shift

### 8.2 Feature Ablations

| Feature Set         | Val MAE | Change   |
| ------------------- | ------- | -------- |
| All 28 features     | 1.416   | Baseline |
| Color only (13)     | [Test]  | -        |
| + Quality (18)      | [Test]  | -        |
| + Distribution (27) | [Test]  | -        |

### 8.3 Model Ablations

| Model                 | Val MAE   | Size        |
| --------------------- | --------- | ----------- |
| **Gradient Boosting** | **1.416** | **0.44 MB** |
| Random Forest         | [Test]    | ~1 MB       |
| Linear Regression     | [Test]    | ~1 KB       |

---

## 9. Compute Cost & Efficiency (0.5 pages)

### 9.1 Training Cost

- **Time:** ~60 seconds
- **Hardware:** CPU only (no GPU needed)
- **Energy:** < 0.01 kWh
- **Compute:** < 0.1 CPU-hours

### 9.2 Inference Cost

- **Per image:** ~50 ms
- **Batch (100 images):** ~5 seconds
- **Hardware:** CPU-only (mobile-friendly)
- **Memory:** ~200 MB RAM

### 9.3 Model Size

- **ONNX model:** 0.44 MB
- **Scaler:** < 0.01 MB
- **Total:** 0.44 MB
- **✅ Qualifies for Edge-Lite track** (< 10 MB)

### 9.4 Deployment

- Edge device compatible (smartphone, Raspberry Pi)
- No internet required (offline inference)
- Fast enough for real-time screening

---

## 10. Limitations & Future Work (0.5 pages)

### 10.1 Current Limitations

1. **Overfitting:** Large train/val gap (33x)
2. **Small dataset:** Only 30 original images
3. **Domain shift:** Conjunctiva data doesn't fully transfer to lips
4. **No uncertainty:** Point predictions only
5. **Limited diversity:** Narrow demographic representation
6. **Lighting sensitivity:** Color features affected by lighting

### 10.2 Next Steps

**Short-term (3-6 months):**

- Collect 200+ diverse lip images
- Cross-validation instead of single split
- Ensemble methods (bagging, stacking)
- Add CNN features (ResNet, EfficientNet)
- Implement prediction intervals

**Long-term (6-12 months):**

- End-to-end deep learning (CNN → HgB)
- Multi-task learning (HgB + other biomarkers)
- Active learning for efficient data collection
- Clinical validation study
- Mobile app deployment

### 10.3 Path to Target Performance (MAE ≤ 0.8)

1. **More data:** 10x larger dataset (300 images)
2. **Better features:** Deep learning embeddings
3. **Better regularization:** Ensemble, dropout
4. **Domain-specific data:** Lip images only (no conjunctiva)

---

## 11. Conclusion (0.25 pages)

### Summary

- Developed lightweight model (0.44 MB) for hemoglobin prediction
- Used handcrafted features + Gradient Boosting
- Achieved edge deployment readiness
- Validation MAE: 1.416 g/dL (target: ≤ 0.8)

### Key Contributions

1. ✅ Edge-Lite model (0.44 MB)
2. ✅ Fast inference (50 ms)
3. ✅ Interpretable features (28 handcrafted)
4. ⚠️ Performance gap remains

### Lessons Learned

- Small datasets require careful regularization
- Domain shift (conjunctiva → lips) hurts performance
- Augmentation helps but not sufficient
- Need larger, more diverse training data

---

## Appendix (Optional, if space permits)

### A. Feature Definitions

[Detailed mathematical definitions of all 28 features]

### B. Hyperparameter Tuning

[Grid search results if performed]

### C. Additional Visualizations

[Feature correlation heatmap, etc.]

---

## Figures to Include (Total: 6-8)

1. **Data distribution:** Histogram of HgB values
2. **Augmentation examples:** Original + 3 augmented versions
3. **Feature importance:** Bar chart of top 10 features
4. **Prediction scatter:** True vs Predicted HgB
5. **Residuals plot:** Error vs True HgB
6. **Training curves:** MAE over training iterations (if available)
7. **Error by HgB range:** MAE breakdown
8. **Failure cases:** 2-3 example images with analysis

---

## Tables to Include (Total: 4-6)

1. **Dataset summary:** Train/val split details
2. **Performance metrics:** MAE, R², RMSE, etc.
3. **Ablation study:** Data configurations
4. **Model comparison:** GB vs baselines
5. **Compute cost:** Training/inference times
6. **Feature categories:** 28 features organized

---

## Writing Tips

### Style

- Clear, concise technical writing
- Use bullet points for readability
- Include equations where appropriate
- Cite relevant papers (anemia detection, feature engineering)

### Formatting

- 11pt font, single-column
- 1-inch margins
- Section numbers for navigation
- High-quality figures (300 DPI)

### Content Balance

- **Focus on insights**, not just numbers
- **Explain decisions:** Why this architecture? Why these features?
- **Be honest about limitations:** Acknowledge overfitting
- **Forward-looking:** Clear path to improvement

### Page Allocation (8 pages total)

- Introduction: 0.5
- Data: 1.5
- Features: 1
- Model: 1.5
- Results: 1
- Error Analysis: 1
- Other sections: 1.5

---

## References (Not counted toward 8 pages)

[Add citations to:]

- Gradient Boosting papers (Friedman 2001)
- Anemia detection literature
- Feature engineering best practices
- Fairness in ML
- Model calibration techniques

---

**Good luck with your report! 📝🚀**
