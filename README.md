# Hemoglobin Prediction from Lip Images

Non-invasive hemoglobin level prediction using computer vision and machine learning for anemia screening.

---

## 🎯 Overview

This project predicts hemoglobin levels (g/dL) from smartphone images of lips using a Random Forest model with handcrafted features.

**Key Features:**

- 🚀 Lightweight model (0.44 MB) suitable for edge deployment
- 📱 Works with smartphone images
- 🔬 28 handcrafted features (color, quality, distribution metrics)
- 🎯 Random Forest with 300 estimators
- 📦 ONNX format for cross-platform inference

---

## 📋 Table of Contents

1. [Quick Start (Inference Only)](#quick-start-inference-only)
2. [Training from Scratch](#training-from-scratch)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [Performance](#performance)
6. [Citation](#citation)

---

## ⚡ Quick Start (Inference Only)

If you just want to use the pretrained model to predict hemoglobin levels from images.

### Prerequisites

- Python 3.8+
- ONNX Runtime

### Installation

```bash
# Clone repository
git clone https://github.com/RaynerGWH/gerg-does-hemoglobin-prediction.git
cd gerg-does-hemoglobin-prediction

# Create virtual environment (optional but recommended)
python -m venv env
source env/bin/activate  # Linux/macOS
# or
.\env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Inference

**Option 1: With competition-format images and metadata**

```bash
python inference.py --images path/to/images --meta meta.csv --out predictions.csv
```

Your images should be named: `001.jpg`, `002.jpg`, etc.  
Your `meta.csv` should have: `image_id`, `device`, `lighting` columns

**Option 2: Without metadata (any image format)**

```bash
python inference.py --images path/to/images --out predictions.csv
```

### Example

```bash
# Test with sample images
python inference.py --images test_inference_setup/images --meta test_inference_setup/meta.csv --out my_predictions.csv

# View results
cat my_predictions.csv
```

**Output:** A CSV file with columns `image_id` and `hemoglobin_gdl`

```csv
image_id,hemoglobin_gdl
001,12.34
002,13.45
003,11.23
```

---

## 🔨 Training from Scratch

If you want to train your own model or reproduce our results.

### Prerequisites

- Python 3.8+
- 4GB RAM minimum
- Your own lip images with known hemoglobin values

### Step-by-Step Guide

#### 1. Prepare Your Data

Place your lip images in `data/starter/images/` with this naming format:

```
data/starter/images/
    HgB_10.7gdl_Individual01.png
    HgB_12.3gdl_Individual02.png
    HgB_14.5gdl_Individual03.png
    ...
```

**Filename format:** `HgB_{value}gdl_Individual{id}.png`

#### 2. Create Labels

Extract hemoglobin values from filenames into a CSV:

```bash
python code/00_create_labels_csv.py
```

**Output:** `data/starter/labels.csv` with image paths and HgB values

#### 3. Extract Features

Extract 28 handcrafted features from images:

```bash
python code/01_extract_enhanced_features.py
```

**Output:**

- `data/processed/enhanced_features.npy` (features)
- `data/processed/labels_valid.csv` (labels)
- `data/processed/feature_names.txt` (feature names)

#### 4. Augment Data (Optional)

Increase training samples through augmentation:

```bash
python code/02_augment_training_data.py
```

**Output:** Augmented images and train/val splits in `data/processed/`

#### 5. Add Conjunctiva Dataset (Optional)

**Note:** The conjunctiva dataset used in our model is from our research and is **not included in this repository**. It is available in our accompanying research report.

If you have conjunctiva images with known hemoglobin values, you can add them for additional training diversity:

1. Place conjunctiva images in `data/external/conjunctiva_images/`
2. Create a CSV file `data/external/conjunctiva_labels.csv` with columns: `filename`, `hgb`
3. Process the conjunctiva data:

```bash
python code/03_process_conjunctiva_data.py
```

**What this does:**

- Reads conjunctiva images and labels
- Extracts same 28 features as lip images
- Saves to `data/processed/conjunctiva_features.npy` and `conjunctiva_labels.npy`
- Training script will automatically include these if available

**Conjunctiva Dataset Details:**

- Not included in this repository
- Available in our research report (see citation section)
- Optional: Model can be trained without it using only lip images

#### 6. Train Model

Train the Random Forest model:

```bash
python code/04_train_combined_model.py --model rf
```

**Options:**

- `--model rf`: Random Forest (recommended)
- `--model gb`: Gradient Boosting
- `--model both`: Train and compare both

**Output:**

- `weights/final_model.pkl` (trained model)
- `weights/feature_scaler.pkl` (feature scaler)
- `weights/model_config.txt` (configuration)

#### 7. Export to ONNX

Convert model to ONNX format for inference:

```bash
python code/export_model_onnx.py
```

**Output:**

- `weights/model.onnx` (0.43 MB)
- `weights/scaler.onnx` (0.01 MB)

#### 8. Evaluate (Optional)

Test model performance:

```bash
python code/05_evaluate_on_test.py --model-type rf
```

**Output:** `code/results/test_evaluation.csv`

### Complete Training Command Sequence

```bash
# Step 1: Create labels from filenames
python code/00_create_labels_csv.py

# Step 2: Extract features
python code/01_extract_enhanced_features.py

# Step 3: Augment data (optional)
python code/02_augment_training_data.py

# Step 4: Add conjunctiva data (optional - requires external dataset)
python code/03_process_conjunctiva_data.py

# Step 5: Train model
python code/04_train_combined_model.py --model rf

# Step 6: Export to ONNX
python code/export_model_onnx.py

# Step 7: Test inference
python inference.py --images test_images --out predictions.csv
```

---

## 📁 Project Structure

```
gerg-does-hemoglobin-prediction/
├── inference.py                    # Standalone inference script
├── requirements.txt                # Python dependencies
├── model_card.md                   # Model documentation
├── README.md                       # This file
│
├── data/
│   ├── starter/                    # Original starter dataset
│   │   ├── images/                 # Raw lip images (30 samples)
│   │   └── labels.csv              # Generated image labels
│   ├── external/                   # External datasets (not in repo)
│   │   ├── conjunctiva_images/     # Conjunctiva images (optional)
│   │   └── conjunctiva_labels.csv  # Conjunctiva labels (optional)
│   └── processed/                  # Generated during training
│       ├── enhanced_features.npy
│       ├── labels_valid.csv
│       ├── train_features.npy
│       ├── val_features.npy
│       ├── conjunctiva_features.npy  # If conjunctiva data added
│       └── feature_names.txt
│
├── weights/
│   ├── model.onnx                  # ONNX model (0.43 MB) ✓
│   ├── scaler.onnx                 # ONNX scaler (0.01 MB) ✓
│   ├── final_model.pkl             # Pickle model
│   ├── feature_scaler.pkl          # Pickle scaler
│   └── model_config.txt            # Model configuration
│
└── code/
    ├── 00_create_labels_csv.py             # Create labels from filenames
    ├── 01_extract_enhanced_features.py     # Feature extraction
    ├── 02_augment_training_data.py         # Data augmentation
    ├── 03_process_conjunctiva_data.py      # External conjunctiva data
    ├── 04_train_combined_model.py          # Model training
    ├── 05_evaluate_on_test.py              # Model evaluation
    └── export_model_onnx.py                # ONNX export
```

**Note:** Files/folders marked with ✓ are included in the repository. The `data/external/` conjunctiva dataset is **not included** and is referenced in our research report.

---

## 📦 Requirements

### Core Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
opencv-python>=4.8.0
Pillow>=10.0.0
onnx>=1.14.0
onnxruntime>=1.15.0
skl2onnx>=1.16.0
```

### Install All Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Performance

### Model Specifications

- **Model Type:** Random Forest Regressor
- **Architecture:** 300 estimators, max_depth=20
- **Features:** 28 handcrafted features
- **Model Size:** 0.44 MB (Edge-Lite track)
- **Training Samples:** 200
- **Validation Samples:** 7

### Metrics

| Metric | Training   | Validation |
| ------ | ---------- | ---------- |
| MAE    | 1.233 g/dL | 1.785 g/dL |
| Target | ≤ 0.8 g/dL | ≤ 0.8 g/dL |

### Features (28 Total)

**Color Features (13):**

- RGB percentages and ratios
- Inter-channel ratios
- Color dominance metrics

**Quality Features (5):**

- Brightness, contrast
- Blur score, saturation
- Lighting uniformity

**Distribution Features (9):**

- Statistical moments (mean, std, skewness)
- Per RGB channel

---

## 🚀 Creating Submission Bundle

To create a competition submission package:

```bash
python create_submission_bundle.py
```

**Output:** `submission.zip` containing:

- `inference.py`
- `weights/model.onnx` and `weights/scaler.onnx`
- `requirements.txt`
- `model_card.md`
- `README.md`

**Note:** The submission bundle includes only the essential files for inference. Training code is available in the `code/` directory.

---

## 🔧 Troubleshooting

### Issue: "Module not found"

```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

### Issue: "ONNX model not found"

```bash
# Solution: Train and export model
python code/04_train_combined_model.py --model rf
python code/export_model_onnx.py
```

### Issue: "Cannot open image"

- Ensure images are `.jpg`, `.jpeg`, or `.png`
- Check file paths are correct
- Verify images aren't corrupted

### Issue: "Conjunctiva dataset not found"

- The conjunctiva dataset is **optional** and not included in this repository
- Model can be trained using only lip images from `data/starter/`
- If you want to use conjunctiva data, see our research report for dataset details
- Skip step 4 (conjunctiva processing) if you don't have this data

### Issue: Poor predictions

- Use consistent lighting (natural daylight preferred)
- Ensure images are sharp and focused
- Crop to show lip region clearly
- Use similar camera as training data

---

## 📖 Documentation

- **[model_card.md](model_card.md)** - Complete model documentation and performance metrics

---

## 🎯 Dataset Information

### Starter Dataset (Included)

- **Location:** `data/starter/images/`
- **Size:** 30 lip images with known hemoglobin values
- **Format:** PNG images with filename format `HgB_{value}gdl_Individual{id}.png`
- **Included:** ✓ Yes

### Conjunctiva Dataset (Not Included)

- **Location:** Available in our research report (not in this repository)
- **Purpose:** Optional additional training data for improved model diversity
- **Format:** Conjunctiva images with known hemoglobin values
- **Included:** ✗ No (see research report for access)
- **Note:** Model can be trained without this dataset using only lip images

If you have access to conjunctiva data, place it in `data/external/conjunctiva_images/` and follow Step 5 in the training guide.

---

## 🎯 Competition Track

**Edge-Lite Track** (≤ 10 MB)

- Model Size: 0.44 MB ✅
- Target MAE: ≤ 0.8 g/dL ⚠️ (Current: 1.785 g/dL)

---

## 🔬 Technical Details

### Model Architecture

- **Algorithm:** Random Forest Regressor
- **Estimators:** 300 trees
- **Max Depth:** 20
- **Min Samples Split:** 5
- **Min Samples Leaf:** 2
- **Random State:** 42 (reproducible)

### Training Configuration

- **Training Data:** 200 samples (lip images only) or ~1,620 samples (with optional conjunctiva data)
- **Optimizer:** scikit-learn default (bootstrap aggregating)
- **Feature Scaling:** StandardScaler
- **Export Format:** ONNX (Opset 12)
- **Runtime:** ONNX Runtime (CPU)

**Note:** The current pretrained model (0.44 MB) was trained on 200 carefully selected samples for optimal generalization.

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is for educational and research purposes. See `LICENSE` file for details.

---

## 👥 Authors

- **GitHub:** [RaynerGWH](https://github.com/RaynerGWH)
- **Repository:** [gerg-does-hemoglobin-prediction](https://github.com/RaynerGWH/gerg-does-hemoglobin-prediction)

---

## 📧 Contact

For questions or issues:

- **GitHub Issues:** https://github.com/RaynerGWH/gerg-does-hemoglobin-prediction/issues
- **Discussions:** https://github.com/RaynerGWH/gerg-does-hemoglobin-prediction/discussions

---

## ⚠️ Disclaimer

This model is for research and educational purposes only. It has **not been approved for clinical use**. Always consult qualified healthcare professionals for medical decisions.

---

## 🌟 Acknowledgments

- Competition organizers for the dataset and challenge
- scikit-learn for machine learning framework
- ONNX for model portability

---

**Quick Links:**

- 📚 [Detailed Usage Guide](USAGE_GUIDE.md)
- 📋 [Model Card](model_card.md)
- 🔄 [Development Workflow](WORKFLOW.md)
- 📦 [Create Submission Bundle](create_submission_bundle.py)
