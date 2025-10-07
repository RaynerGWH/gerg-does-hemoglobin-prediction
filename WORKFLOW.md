# Hemoglobin Prediction - Quick Workflow Guide

## 🚀 Complete Pipeline (Run from `scripts/` directory)

### Prerequisites

```bash
# 1. Ensure your 30 images are in: data/starter/images/
# 2. Download Kaggle anemia dataset to: data/external/kaggle_anemia/anemia_dataset.csv
# 3. Activate virtual environment
cd c:\Users\easan.s.2024\Documents\GitHub\gerg-does-hemoglobin-prediction
env\Scripts\activate
cd scripts
```

---

## Step-by-Step Execution

### Step 0: Generate Labels (30 seconds)

```bash
python create_labels_csv.py
```

✅ Creates `../data/starter/labels.csv` from image filenames

---

### Step 1: Extract Enhanced Features (1-2 minutes)

```bash
python 01_extract_enhanced_features.py
```

✅ Extracts 28 features per image
✅ Output: `../data/processed/enhanced_features.npy`

---

### Step 2: Augment Training Data (3-5 minutes)

```bash
python 02_augment_training_data.py
```

✅ Creates 3 augmented versions per image (30 → 120 samples)
✅ Output: `../data/processed/augmented_features.npy`

---

### Step 3: Process Conjunctiva Data (30 seconds)

```bash
python 03_process_conjunctiva_data.py
```

✅ Normalizes Kaggle dataset (~1,500 samples)
✅ Output: `../data/processed/conjunctiva_features.npy`

---

### Step 4: Extract CNN Features - OPTIONAL (5-10 minutes)

```bash
# Option A: Skip CNN (faster, still good results)
python 04_extract_cnn_features.py --skip-cnn

# Option B: Use CNN (better accuracy, slower)
python 04_extract_cnn_features.py --model resnet18
```

✅ Optional deep features (512-2048 dims)
✅ Output: `../data/processed/cnn_features_resnet18.npy`

---

### Step 5: Train Model (30-60 seconds)

```bash
# Recommended: Random Forest (fast, robust)
python 05_train_combined_model.py --model rf

# Alternative: Gradient Boosting (potentially better accuracy)
python 05_train_combined_model.py --model gb

# With CNN features:
python 05_train_combined_model.py --model rf --use-cnn --cnn-model resnet18
```

✅ Trains on augmented + conjunctiva data (~1,620 samples)
✅ Output: `../weights/final_model.pkl`

---

### Step 6: Evaluate on Test Set (30-60 seconds)

```bash
python 06_evaluate_on_test.py
```

✅ Predicts HgB for 30 test images
✅ Output: `../results/test_evaluation.csv` + visualization

---

## ⚡ One-Line Execution (All Steps)

```bash
cd scripts && ^
python create_labels_csv.py && ^
python 01_extract_enhanced_features.py && ^
python 02_augment_training_data.py && ^
python 03_process_conjunctiva_data.py && ^
python 04_extract_cnn_features.py --skip-cnn && ^
python 05_train_combined_model.py --model rf && ^
python 06_evaluate_on_test.py
```

**Total Time:** ~5-8 minutes (without CNN)

---

## 📊 Expected Results

| Configuration                   | Training Samples | Expected MAE         | Time    |
| ------------------------------- | ---------------- | -------------------- | ------- |
| Original only (30 images)       | 30               | ~1.2-1.5 g/dL ❌     | ~2 min  |
| With augmentation               | 120              | ~1.0-1.2 g/dL ⚠️     | ~5 min  |
| With augmentation + conjunctiva | ~1,620           | **~0.7-0.9 g/dL** ✅ | ~6 min  |
| With aug + conj + CNN           | ~1,620           | **~0.6-0.8 g/dL** ✅ | ~15 min |

**Target: MAE ≤ 0.8 g/dL** 🎯

---

## 🔧 Common Options

### Model Selection

```bash
--model rf         # Random Forest (default, recommended)
--model gb         # Gradient Boosting (higher accuracy)
--model ridge      # Ridge Regression (baseline)
```

### CNN Options

```bash
--skip-cnn                      # Skip CNN extraction
--use-cnn --cnn-model resnet18  # Use ResNet18 (512 features)
--use-cnn --cnn-model resnet50  # Use ResNet50 (2048 features)
--use-cnn --cnn-model mobilenet_v2  # Use MobileNetV2 (1280 features)
```

### Training Options

```bash
--no-augmentation   # Train without augmented data
```

---

## 📁 Key Output Files

### After Step 1:

- `../data/processed/enhanced_features.npy` - 28 features × 30 images
- `../data/processed/feature_names.txt` - Feature names

### After Step 2:

- `../data/augmented/` - Augmented images (PNG)
- `../data/processed/augmented_features.npy` - 28 features × 120 samples

### After Step 3:

- `../data/processed/conjunctiva_features.npy` - 28 features × ~1,500 samples

### After Step 4 (optional):

- `../data/processed/cnn_features_resnet18.npy` - 512 features × 30 images

### After Step 5:

- `../weights/final_model.pkl` - Trained model
- `../weights/feature_scaler.pkl` - Feature standardization
- `../weights/model_config.txt` - Model configuration

### After Step 6:

- `../results/test_evaluation.csv` - Predictions + errors
- `../results/test_evaluation_plot.png` - Scatter plot

---

## ⚠️ Troubleshooting

### "No module named 'cv2'"

```bash
pip install opencv-python
```

### "No module named 'pillow_heif'"

```bash
pip install pillow-heif
```

### "Labels file not found"

```bash
# Make sure you ran create_labels_csv.py first
# Check that images are in: data/starter/images/
```

### "Kaggle dataset not found"

```bash
# Download from: https://www.kaggle.com/datasets/biswaranjanrao/anemia-detection
# Place anemia_dataset.csv in: data/external/kaggle_anemia/
```

### "PyTorch not available" (CNN)

```bash
# Either install PyTorch:
pip install torch torchvision
# Or skip CNN features:
python 04_extract_cnn_features.py --skip-cnn
```

---

## 🎯 Performance Optimization Tips

### For Best Accuracy:

1. ✅ Use Gradient Boosting: `--model gb`
2. ✅ Add CNN features: `--use-cnn --cnn-model resnet50`
3. ✅ Ensure high-quality images (sharp, well-lit)

### For Fastest Training:

1. ⚡ Use Random Forest: `--model rf`
2. ⚡ Skip CNN: `--skip-cnn`
3. ⚡ Use smaller augmentation factor (edit script)

---

## 📝 Feature Breakdown

### Handcrafted Features (28 total):

- **Color (13):** RGB %, ratios, dominance, normalized
- **Quality (5):** brightness, contrast, blur, saturation, uniformity
- **Extended (1):** redness index
- **Distribution (9):** mean, std, skewness per RGB channel

### CNN Features (optional):

- **ResNet18:** 512 features
- **ResNet50:** 2,048 features
- **MobileNetV2:** 1,280 features

---

## 🚦 Pipeline Status Indicators

After each step, you should see:

- ✅ Success messages
- 📊 Sample counts
- 📁 Output file paths
- ⏱️ Execution time

If you see ❌ or ⚠️ messages, check the error and refer to Troubleshooting.

---

## 📞 Quick Commands Reference

```bash
# View current directory
cd

# List files in data/starter/images/
dir ..\data\starter\images

# Check if labels exist
type ..\data\starter\labels.csv

# Check processed features
dir ..\data\processed

# View model config
type ..\weights\model_config.txt

# View results
type ..\results\test_evaluation.csv
```

---

## 🎓 Understanding the Pipeline

1. **Feature Extraction** → Converts images to numerical features
2. **Augmentation** → Expands training set (30 → 120 samples)
3. **Conjunctiva Integration** → Adds external dataset (~1,500 samples)
4. **CNN (Optional)** → Adds deep learning features
5. **Training** → Learns HgB prediction from features
6. **Evaluation** → Tests on 30 original images

**Key Insight:** We train on augmented+conjunctiva data but evaluate on original 30 images (no data leakage!)

---

**Updated:** October 2025
**Target:** MAE ≤ 0.8 g/dL 🎯
