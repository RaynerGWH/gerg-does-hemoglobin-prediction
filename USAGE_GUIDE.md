# Usage Guide: Hemoglobin Prediction Model

This guide provides step-by-step instructions for training the model from scratch and running inference on new images.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup Instructions](#setup-instructions)
3. [Training the Model (From Scratch)](#training-the-model-from-scratch)
4. [Running Inference (Using Pretrained Model)](#running-inference-using-pretrained-model)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- Python 3.8 or higher
- 4GB RAM minimum
- Windows/Linux/macOS

### Required Python Packages

```
numpy
pandas
scikit-learn>=1.3.0
opencv-python
Pillow
onnx
onnxruntime
skl2onnx
```

---

## Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/RaynerGWH/gerg-does-hemoglobin-prediction.git
cd gerg-does-hemoglobin-prediction
```

### Step 2: Create Virtual Environment (Recommended)

**Windows (PowerShell):**

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

**Windows (CMD):**

```cmd
python -m venv env
.\env\Scripts\activate.bat
```

**Linux/macOS:**

```bash
python -m venv env
source env/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Training the Model (From Scratch)

Follow these steps to train the model from your own data or reproduce the existing model.

### Step 1: Prepare Your Data

Place your lip images in the `data/starter/images/` directory. Images should be named according to their hemoglobin level:

```
data/starter/images/
    HgB_10.7gdl_Individual01.png
    HgB_12.3gdl_Individual02.png
    HgB_14.5gdl_Individual03.png
    ...
```

**Filename format**: `HgB_{value}gdl_Individual{id}.png` or `Random_{value}gdl_Individual{id}.png`

### Step 2: Create Labels CSV

Generate a CSV file that maps image filenames to their hemoglobin values:

```bash
python scripts/create_labels_csv.py
```

**What this does:**

- Scans all images in `data/starter/images/`
- Extracts hemoglobin values from filenames (e.g., `HgB_10.7gdl` → 10.7)
- Creates `data/starter/labels.csv` with columns: `image_id`, `filename`, `filepath`, `hgb`
- Shows statistics: min/max/mean HgB values

**Expected output:**

```
✓ HgB_10.7gdl_Individual01.png → HgB: 10.7 g/dL
✓ HgB_12.3gdl_Individual02.png → HgB: 12.3 g/dL
...
✅ Created labels.csv with 30 samples
📁 Saved to: data/starter/labels.csv
```

**Important**: This step is **required** before feature extraction. The next script will exit with an error if `labels.csv` doesn't exist.

### Step 3: Extract Features

Extract handcrafted features (color statistics, image quality metrics) from your images:

```bash
python scripts/01_extract_enhanced_features.py
```

**What this does:**

- Reads `data/starter/labels.csv` to find images to process
- Extracts 28 features per image (color, quality, distribution statistics)
- Saves features to `data/processed/enhanced_features.npy`
- Saves labels to `data/processed/labels_valid.csv`
- Creates feature names list in `data/processed/feature_names.txt`

**Expected output:**

```
📊 Found 30 labeled images
🔍 Extracting features...
Processing 30 images...
✓ Extracted features: (30, 28)
✓ Saved to data/processed/
```

### Step 4: Augment Training Data (Optional)

Increase training samples through data augmentation:

```bash
python scripts/02_augment_training_data.py
```

**What this does:**

- Creates augmented versions of each image (rotation, brightness, contrast)
- Saves augmented images to `data/augmented/`
- Creates train/validation splits
- Saves to `data/processed/train_features.npy` and `val_features.npy`

**Expected output:**

```
Created 1080 augmented samples
Training samples: 972
Validation samples: 108
```

### Step 5: Add External Data (Optional)

If you have conjunctiva images, process them to add more training diversity:

```bash
python scripts/03_process_conjunctiva_data.py
```

**What this does:**

- Processes external conjunctiva dataset
- Extracts same 28 features
- Saves to `data/processed/conjunctiva_features.npy`

### Step 6: Train the Model

Train the Random Forest model with the original hyperparameters:

```bash
python scripts/05_train_combined_model.py --model rf
```

**Command options:**

- `--model rf`: Train Random Forest (recommended)
- `--model gb`: Train Gradient Boosting
- `--model ridge`: Train Ridge Regression
- `--model both`: Train and compare RF and GB
- `--no-augmentation`: Use only original images (not augmented data)
- `--use-cnn`: Include CNN features (if available)

**What this does:**

- Loads features and labels
- Trains Random Forest with 300 estimators, max_depth=20
- Evaluates on validation set
- Saves model to `weights/final_model.pkl`
- Saves scaler to `weights/feature_scaler.pkl`
- Saves configuration to `weights/model_config.txt`

### Step 6: Export to ONNX Format

Convert the trained model to ONNX for efficient inference:

```bash
python scripts/export_model_onnx.py
```

**What this does:**

- Loads `weights/final_model.pkl` and `weights/feature_scaler.pkl`
- Converts both to ONNX format
- Saves `weights/model.onnx` and `weights/scaler.onnx`
- Reports model size (should be ~0.44 MB total)

**Expected output:**

```
✅ EXPORT COMPLETE
📊 Model Sizes:
   Scaler: 0.01 MB
   Model:  0.43 MB
   Total:  0.44 MB

🏆 QUALIFIES FOR EDGE-LITE TRACK! (≤ 10 MB)
```

### Step 7: Export to ONNX Format

````

### Step 8: Verify Model Performance (Optional)

Evaluate the trained model on test images:

```bash
python scripts/06_evaluate_on_test.py --model-type rf
````

**What this does:**

- Loads trained model and test images
- Generates predictions
- Compares with ground truth
- Saves results to `scripts/results/test_evaluation.csv`

---

## Running Inference (Using Pretrained Model)

Use the pretrained ONNX model to predict hemoglobin levels from new lip images.

### Prerequisites

Ensure you have the model files in the `weights/` directory:

- `weights/model.onnx` (Random Forest model)
- `weights/scaler.onnx` (Feature scaler)

### Step 1: Prepare Your Images

Organize your images according to competition format:

**Option A: Use Competition Format (Recommended)**

```
your_images_folder/
    001.jpg
    002.jpg
    003.jpg
    ...
meta.csv
```

The `meta.csv` should contain:

```csv
image_id,device,lighting
001,smartphone_a,natural
002,smartphone_b,indoor
003,smartphone_a,outdoor
...
```

**Option B: Use Any Image Files**

Place your images in a folder with any naming format (e.g., `image1.jpg`, `sample.png`, etc.)

### Step 2: Run Inference

**Option A: Competition Format with Metadata**

```bash
python inference.py --images path/to/images --meta meta.csv --out predictions.csv
```

**Option B: Images Only (No Metadata)**

```bash
python inference.py --images path/to/images --out predictions.csv
```

**Command options:**

- `--images`: Path to folder containing images
- `--meta`: Path to metadata CSV file (optional)
- `--out`: Output CSV file for predictions (default: `predictions.csv`)

### Step 3: View Results

The predictions will be saved to your specified output file (e.g., `predictions.csv`):

```csv
image_id,hemoglobin_gdl
001,12.34
002,13.45
003,11.23
...
```

**Columns:**

- `image_id`: Image identifier (from filename without extension)
- `hemoglobin_gdl`: Predicted hemoglobin level in g/dL

### Example: Complete Inference Workflow

```bash
# 1. Prepare test images in competition format
mkdir test_images
# Copy your images as 001.jpg, 002.jpg, etc.

# 2. Create metadata file
echo "image_id,device,lighting" > test_meta.csv
echo "001,smartphone,natural" >> test_meta.csv
echo "002,smartphone,indoor" >> test_meta.csv

# 3. Run inference
python inference.py --images test_images --meta test_meta.csv --out my_predictions.csv

# 4. View results
cat my_predictions.csv  # Linux/macOS
type my_predictions.csv  # Windows CMD
```

### Expected Output

```
🔍 Loading ONNX models from weights/...
   ✓ Scaler loaded
   ✓ Model loaded

📊 Processing 30 images...
   ✓ Extracted features from 30 images

🎯 Running inference...
   ✓ Generated 30 predictions

💾 Saving predictions to predictions.csv...

✅ INFERENCE COMPLETE!
   Predictions saved to: predictions.csv
   Total images processed: 30
   Hemoglobin range: 10.2 - 15.7 g/dL
```

---

## Troubleshooting

### Issue 1: Module Not Found Error

**Error:**

```
ModuleNotFoundError: No module named 'sklearn'
```

**Solution:**
Make sure you've activated your virtual environment and installed dependencies:

```bash
pip install -r requirements.txt
```

### Issue 2: ONNX Model Not Found

**Error:**

```
FileNotFoundError: weights/model.onnx not found
```

**Solution:**

1. Make sure you've trained the model and exported to ONNX:
   ```bash
   python scripts/05_train_combined_model.py --model rf
   python scripts/export_model_onnx.py
   ```
2. Or download the pretrained model from the repository's releases page

### Issue 3: Image Format Issues

**Error:**

```
Cannot open image: path/to/image.jpg
```

**Solution:**

- Ensure images are in supported formats: `.jpg`, `.jpeg`, `.png`
- Check that image paths are correct
- Verify images are not corrupted (try opening in image viewer)

### Issue 4: Feature Extraction Fails

**Error:**

```
ValueError: invalid literal for int() with base 10: 'HgB'
```

**Solution:**

- Check filename format: `HgB_{value}gdl_Individual{id}.png`
- Ensure hemoglobin value uses period (.) not comma (,): `10.7` not `10,7`
- Remove any spaces in filenames

### Issue 5: Poor Prediction Accuracy

**Possible causes:**

1. **Different lighting conditions**: Model is sensitive to lighting
2. **Different camera quality**: Trained on specific device types
3. **Image quality**: Blurry or low-resolution images
4. **Lip region**: Ensure clear view of lips without obstruction

**Solutions:**

- Use consistent lighting (natural daylight preferred)
- Ensure images are sharp and well-focused
- Crop images to show primarily lip region
- Use similar camera/device as training data

### Issue 6: Windows PowerShell Execution Policy

**Error:**

```
cannot be loaded because running scripts is disabled on this system
```

**Solution:**

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Advanced Usage

### Training with Different Hyperparameters

Edit `scripts/05_train_combined_model.py` to modify Random Forest parameters:

```python
model = RandomForestRegressor(
    n_estimators=300,      # Number of trees (increase for better fit)
    max_depth=20,          # Tree depth (increase for complexity)
    min_samples_split=5,   # Min samples to split (decrease for more splits)
    min_samples_leaf=2,    # Min samples in leaf (decrease for smaller leaves)
    max_features='sqrt',   # Features per split
    random_state=42,       # For reproducibility
    n_jobs=-1             # Use all CPU cores
)
```

### Using Different Model Types

Compare Random Forest vs Gradient Boosting:

```bash
# Train both models and compare
python scripts/05_train_combined_model.py --model both

# This will show side-by-side comparison and save the best model
```

### Batch Inference on Multiple Folders

```bash
# Process multiple test folders
for folder in test_set_1 test_set_2 test_set_3; do
    python inference.py --images $folder --out ${folder}_predictions.csv
done
```

---

## Performance Benchmarks

### Training Performance

- **Training time**: ~2-5 minutes (200 samples, CPU)
- **Memory usage**: ~500 MB
- **CPU cores**: Uses all available cores (n_jobs=-1)

### Inference Performance

- **Latency**: ~50-100ms per image (CPU)
- **Throughput**: ~10-20 images/second
- **Memory**: ~100 MB
- **Batch size**: Processes all images at once for efficiency

### Model Size

- **ONNX Model**: 0.43 MB
- **Scaler**: 0.01 MB
- **Total**: 0.44 MB (qualifies for Edge-Lite track ≤ 10 MB)

---

## Contact & Support

For questions, issues, or contributions:

- **GitHub Issues**: https://github.com/RaynerGWH/gerg-does-hemoglobin-prediction/issues
- **Repository**: https://github.com/RaynerGWH/gerg-does-hemoglobin-prediction

---

## License

See `LICENSE` file in the repository root.
