"""
Data Augmentation Module
Creates augmented versions of training images to improve model generalization
Run from project root: python scripts/02_augment_training_data.py
"""

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from pathlib import Path
from pillow_heif import register_heif_opener
import random
from typing import List, Tuple
import sys
import os

# Add scripts directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import the enhanced feature extraction module
import importlib.util
spec = importlib.util.spec_from_file_location("extract_enhanced_features", 
                                                os.path.join(script_dir, "01_extract_enhanced_features.py"))
efe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(efe_module)
extract_enhanced_features = efe_module.extract_enhanced_features

register_heif_opener()


class ImageAugmenter:
    """Applies various augmentation techniques to images"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
    
    def rotate(self, img: Image.Image, angle_range: Tuple[int, int] = (-15, 15)) -> Image.Image:
        """Rotate image by random angle"""
        angle = random.uniform(*angle_range)
        return img.rotate(angle, fillcolor=(255, 255, 255))
    
    def adjust_brightness(self, img: Image.Image, factor_range: Tuple[float, float] = (0.7, 1.3)) -> Image.Image:
        """Adjust brightness - MORE AGGRESSIVE"""
        factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)
    
    def adjust_contrast(self, img: Image.Image, factor_range: Tuple[float, float] = (0.7, 1.4)) -> Image.Image:
        """Adjust contrast - MORE AGGRESSIVE"""
        factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    
    def adjust_saturation(self, img: Image.Image, factor_range: Tuple[float, float] = (0.7, 1.3)) -> Image.Image:
        """Adjust color saturation - MORE AGGRESSIVE"""
        factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)
    
    def adjust_sharpness(self, img: Image.Image, factor_range: Tuple[float, float] = (0.5, 1.5)) -> Image.Image:
        """Adjust image sharpness"""
        factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(factor)
    
    def adjust_hue(self, img: Image.Image, hue_shift_range: Tuple[int, int] = (-10, 10)) -> Image.Image:
        """Shift hue (color temperature) in HSV space"""
        img_array = np.array(img)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Shift hue
        hue_shift = random.randint(*hue_shift_range)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        # Convert back
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(rgb)
    
    def adjust_color_temperature(self, img: Image.Image, temp_shift_range: Tuple[float, float] = (-20, 20)) -> Image.Image:
        """Adjust color temperature (warm/cool)"""
        img_array = np.array(img).astype(np.float32)
        temp_shift = random.uniform(*temp_shift_range)
        
        # Warm: increase red/yellow, Cool: increase blue
        if temp_shift > 0:  # Warmer
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] + temp_shift, 0, 255)  # More red
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] + temp_shift * 0.5, 0, 255)  # Slight yellow
        else:  # Cooler
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] - temp_shift, 0, 255)  # More blue
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def adjust_gamma(self, img: Image.Image, gamma_range: Tuple[float, float] = (0.7, 1.3)) -> Image.Image:
        """Apply gamma correction"""
        gamma = random.uniform(*gamma_range)
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.power(img_array, gamma)
        img_array = (img_array * 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def add_gaussian_noise(self, img: Image.Image, std_range: Tuple[int, int] = (3, 20)) -> Image.Image:
        """Add Gaussian noise - MORE AGGRESSIVE"""
        img_array = np.array(img)
        std = random.uniform(*std_range)
        noise = np.random.normal(0, std, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    
    def add_shadow(self, img: Image.Image, intensity_range: Tuple[float, float] = (0.3, 0.7)) -> Image.Image:
        """Add random shadow overlay"""
        img_array = np.array(img).astype(np.float32)
        height, width = img_array.shape[:2]
        
        # Create random shadow mask
        shadow_intensity = random.uniform(*intensity_range)
        
        # Random shadow shape (ellipse or rectangle)
        if random.random() > 0.5:
            # Ellipse shadow
            center_x = random.randint(width // 4, 3 * width // 4)
            center_y = random.randint(height // 4, 3 * height // 4)
            axes_x = random.randint(width // 4, width // 2)
            axes_y = random.randint(height // 4, height // 2)
            
            mask = np.ones((height, width), dtype=np.float32)
            cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, shadow_intensity, -1)
        else:
            # Gradient shadow
            mask = np.linspace(1, shadow_intensity, width, dtype=np.float32)
            mask = np.tile(mask, (height, 1))
        
        # Apply shadow
        for c in range(3):
            img_array[:, :, c] *= mask
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    def apply_blur(self, img: Image.Image, radius_range: Tuple[float, float] = (0.5, 1.5)) -> Image.Image:
        """Apply slight Gaussian blur"""
        radius = random.uniform(*radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def horizontal_flip(self, img: Image.Image) -> Image.Image:
        """Flip image horizontally"""
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    
    def zoom(self, img: Image.Image, zoom_range: Tuple[float, float] = (0.9, 1.1)) -> Image.Image:
        """Zoom in/out slightly"""
        zoom_factor = random.uniform(*zoom_range)
        width, height = img.size
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)
        
        # Resize
        img_zoomed = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Crop or pad to original size
        if zoom_factor > 1:  # Crop
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            img_zoomed = img_zoomed.crop((left, top, left + width, top + height))
        else:  # Pad
            new_img = Image.new('RGB', (width, height), (255, 255, 255))
            paste_x = (width - new_width) // 2
            paste_y = (height - new_height) // 2
            new_img.paste(img_zoomed, (paste_x, paste_y))
            img_zoomed = new_img
        
        return img_zoomed
    
    def augment(self, img: Image.Image, num_augmentations: int = 3) -> List[Image.Image]:
        """
        Apply random augmentations to create multiple variants
        
        Args:
            img: Input PIL Image
            num_augmentations: Number of augmented versions to create
        
        Returns:
            List of augmented images
        """
        augmented_images = []
        
        # All augmentation methods (EXPANDED!)
        augmentation_methods = [
            self.rotate,
            self.adjust_brightness,
            self.adjust_contrast,
            self.adjust_saturation,
            self.adjust_sharpness,
            self.adjust_hue,
            self.adjust_color_temperature,
            self.adjust_gamma,
            self.add_gaussian_noise,
            self.add_shadow,
            self.apply_blur,
            self.horizontal_flip,
            self.zoom
        ]
        
        for i in range(num_augmentations):
            # Start with original image
            aug_img = img.copy()
            
            # Apply 3-5 random augmentations (MORE transforms per image)
            num_transforms = random.randint(3, 5)
            transforms = random.sample(augmentation_methods, num_transforms)
            
            for transform in transforms:
                aug_img = transform(aug_img)
            
            augmented_images.append(aug_img)
        
        return augmented_images


def main():
    """Augment training data and extract features"""
    
    print("=" * 70)
    print("DATA AUGMENTATION FOR TRAINING SET")
    print("=" * 70)
    
    # Configuration
    NUM_AUGMENTATIONS = 3  # Create 3 augmented versions per image
    TRAIN_SPLIT = 0.8  # 80% for training, 20% for validation
    RANDOM_SEED = 42
    
    # Load labels
    labels_path = Path('../data/starter/labels.csv')
    if not labels_path.exists():
        print(f"\n❌ Labels file not found: {labels_path}")
        return
    
    df = pd.read_csv(labels_path)
    
    # Split into train/validation
    np.random.seed(RANDOM_SEED)
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    
    n_train = int(len(df) * TRAIN_SPLIT)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    df_train = df.iloc[train_indices].reset_index(drop=True)
    df_val = df.iloc[val_indices].reset_index(drop=True)
    
    print(f"\n📊 Dataset split:")
    print(f"   Total images: {len(df)}")
    print(f"   Training: {len(df_train)} images (will be augmented)")
    print(f"   Validation: {len(df_val)} images (NO augmentation)")
    print(f"\n🔄 Creating {NUM_AUGMENTATIONS} augmented versions per training image")
    print(f"📈 Training samples after augmentation: {len(df_train) * (1 + NUM_AUGMENTATIONS)}")
    
    # Initialize augmenter
    augmenter = ImageAugmenter(seed=RANDOM_SEED)
    
    # Store all features and labels
    train_features = []
    train_labels = []
    train_metadata = []
    
    val_features = []
    val_labels = []
    val_metadata = []
    
    print("\n🔍 Processing TRAINING images (with augmentation)...")
    
    for idx, row in df_train.iterrows():
        img_path = Path(row['filepath'])
        if not img_path.exists():
            img_path = Path('..') / img_path
        
        try:
            # Load original image
            img = Image.open(img_path).convert('RGB')
            
            # Extract features from original
            original_features = extract_enhanced_features(img_path)
            if original_features is None:
                print(f"  ✗ [{idx+1}/{len(df_train)}] {img_path.name} - FAILED")
                continue
            
            # Add original
            train_features.append(original_features)
            train_labels.append(row['hgb'])
            train_metadata.append({
                'image_id': row['image_id'],
                'filename': row['filename'],
                'augmentation': 'original',
                'hgb': row['hgb'],
                'split': 'train'
            })
            
            # Create and process augmented versions
            augmented_imgs = augmenter.augment(img, NUM_AUGMENTATIONS)
            
            # Save augmented images and extract features
            aug_dir = Path('../data/augmented')
            aug_dir.mkdir(parents=True, exist_ok=True)
            
            for aug_idx, aug_img in enumerate(augmented_imgs):
                # Save augmented image
                aug_filename = f"{row['image_id']}_aug{aug_idx+1}.png"
                aug_path = aug_dir / aug_filename
                aug_img.save(aug_path)
                
                # Extract features from augmented image
                aug_features = extract_enhanced_features(aug_path)
                if aug_features is not None:
                    train_features.append(aug_features)
                    train_labels.append(row['hgb'])
                    train_metadata.append({
                        'image_id': f"{row['image_id']}_aug{aug_idx+1}",
                        'filename': aug_filename,
                        'augmentation': f'aug_{aug_idx+1}',
                        'hgb': row['hgb'],
                        'split': 'train'
                    })
            
            print(f"  ✓ [{idx+1}/{len(df_train)}] {img_path.name} + {NUM_AUGMENTATIONS} augmentations")
            
        except Exception as e:
            print(f"  ✗ [{idx+1}/{len(df_train)}] {img_path.name} - ERROR: {e}")
            continue
    
    # Process VALIDATION images (NO augmentation)
    print("\n🔍 Processing VALIDATION images (NO augmentation)...")
    
    for idx, row in df_val.iterrows():
        img_path = Path(row['filepath'])
        if not img_path.exists():
            img_path = Path('..') / img_path
        
        try:
            # Extract features from original only (no augmentation for validation)
            original_features = extract_enhanced_features(img_path)
            if original_features is None:
                print(f"  ✗ [{idx+1}/{len(df_val)}] {img_path.name} - FAILED")
                continue
            
            # Add to validation set
            val_features.append(original_features)
            val_labels.append(row['hgb'])
            val_metadata.append({
                'image_id': row['image_id'],
                'filename': row['filename'],
                'augmentation': 'original',
                'hgb': row['hgb'],
                'split': 'validation'
            })
            
            print(f"  ✓ [{idx+1}/{len(df_val)}] {img_path.name}")
            
        except Exception as e:
            print(f"  ✗ [{idx+1}/{len(df_val)}] {img_path.name} - ERROR: {e}")
            continue
    
    # Convert to arrays and DataFrames
    train_features_df = pd.DataFrame(train_features)
    train_feature_array = train_features_df.values
    train_labels_array = np.array(train_labels)
    train_metadata_df = pd.DataFrame(train_metadata)
    
    val_features_df = pd.DataFrame(val_features)
    val_feature_array = val_features_df.values
    val_labels_array = np.array(val_labels)
    val_metadata_df = pd.DataFrame(val_metadata)
    
    # Save outputs
    output_dir = Path('../data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training data (augmented)
    train_feature_path = output_dir / 'train_features.npy'
    train_labels_path = output_dir / 'train_labels.npy'
    train_metadata_path = output_dir / 'train_metadata.csv'
    
    np.save(train_feature_path, train_feature_array)
    np.save(train_labels_path, train_labels_array)
    train_metadata_df.to_csv(train_metadata_path, index=False)
    
    # Save validation data (NOT augmented)
    val_feature_path = output_dir / 'val_features.npy'
    val_labels_path = output_dir / 'val_labels.npy'
    val_metadata_path = output_dir / 'val_metadata.csv'
    
    np.save(val_feature_path, val_feature_array)
    np.save(val_labels_path, val_labels_array)
    val_metadata_df.to_csv(val_metadata_path, index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ DATA AUGMENTATION COMPLETE")
    print("=" * 70)
    print(f"\n📊 TRAINING SET (with augmentation):")
    print(f"  Original images: {len(df_train)}")
    print(f"  After augmentation: {len(train_features)}")
    print(f"  Features per sample: {train_feature_array.shape[1]}")
    print(f"  HgB range: {train_labels_array.min():.1f} - {train_labels_array.max():.1f} g/dL")
    
    print(f"\n📊 VALIDATION SET (NO augmentation):")
    print(f"  Images: {len(val_features)}")
    print(f"  Features per sample: {val_feature_array.shape[1]}")
    print(f"  HgB range: {val_labels_array.min():.1f} - {val_labels_array.max():.1f} g/dL")
    
    print(f"\n📁 Output files:")
    print(f"  TRAINING:")
    print(f"    - Features: {train_feature_path}")
    print(f"    - Labels: {train_labels_path}")
    print(f"    - Metadata: {train_metadata_path}")
    print(f"  VALIDATION:")
    print(f"    - Features: {val_feature_path}")
    print(f"    - Labels: {val_labels_path}")
    print(f"    - Metadata: {val_metadata_path}")
    print(f"  Augmented images: {aug_dir}")
    
    print(f"\n⚠️  NOTE: Validation set is INDEPENDENT - not used during training!")
    print(f"   This prevents data leakage from augmentation.")


if __name__ == '__main__':
    main()
