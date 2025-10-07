"""
Optional CNN Feature Extractor
Uses a pre-trained CNN to extract deep features from images
Can be used in addition to handcrafted features for improved performance
Run from project root: python scripts/04_extract_cnn_features.py [--skip-cnn]
"""

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from pillow_heif import register_heif_opener
import argparse
import warnings
warnings.filterwarnings('ignore')

# Try to import deep learning libraries
try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available. CNN features will be skipped.")

register_heif_opener()


class CNNFeatureExtractor:
    """Extract features using pre-trained CNN"""
    
    def __init__(self, model_name: str = 'resnet18'):
        """
        Initialize CNN feature extractor
        
        Args:
            model_name: Name of pre-trained model ('resnet18', 'resnet50', 'mobilenet_v2')
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed. Cannot use CNN features.")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load pre-trained model
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            # Remove final classification layer
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 512
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 2048
        elif model_name == 'mobilenet_v2':
            self.model = models.mobilenet_v2(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 1280
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"Loaded {model_name} model (feature dim: {self.feature_dim})")
    
    def extract_features(self, img_path: Path) -> np.ndarray:
        """
        Extract CNN features from image
        
        Args:
            img_path: Path to image file
        
        Returns:
            Feature vector as numpy array
        """
        try:
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(img_tensor)
            
            # Convert to numpy and flatten
            features = features.cpu().numpy().flatten()
            
            return features
            
        except Exception as e:
            print(f"Error extracting CNN features from {img_path}: {e}")
            return None


def extract_cnn_features_from_dataset(
    image_paths: list,
    output_path: Path,
    model_name: str = 'resnet18'
) -> np.ndarray:
    """
    Extract CNN features from a list of images
    
    Args:
        image_paths: List of image file paths
        output_path: Where to save features
        model_name: CNN model to use
    
    Returns:
        Array of CNN features
    """
    if not TORCH_AVAILABLE:
        print("⚠️  PyTorch not available. Skipping CNN feature extraction.")
        return None
    
    print(f"\n🔍 Extracting CNN features using {model_name}...")
    
    # Initialize extractor
    extractor = CNNFeatureExtractor(model_name)
    
    # Extract features
    features_list = []
    valid_indices = []
    
    for idx, img_path in enumerate(image_paths):
        features = extractor.extract_features(img_path)
        if features is not None:
            features_list.append(features)
            valid_indices.append(idx)
            print(f"  ✓ [{idx+1}/{len(image_paths)}] {img_path.name}")
        else:
            print(f"  ✗ [{idx+1}/{len(image_paths)}] {img_path.name} - FAILED")
    
    # Convert to array
    if len(features_list) > 0:
        features_array = np.array(features_list)
        np.save(output_path, features_array)
        print(f"\n✅ Saved CNN features to {output_path}")
        print(f"   Shape: {features_array.shape}")
        return features_array, valid_indices
    else:
        print("\n❌ No CNN features extracted")
        return None, []


def main():
    """Extract CNN features from all images"""
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-cnn', action='store_true', 
                       help='Skip CNN feature extraction')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet50', 'mobilenet_v2'],
                       help='CNN model to use')
    args = parser.parse_args()
    
    print("=" * 70)
    print("CNN FEATURE EXTRACTION (OPTIONAL)")
    print("=" * 70)
    
    if args.skip_cnn:
        print("\n⏭️  Skipping CNN feature extraction (--skip-cnn flag set)")
        print("The pipeline will continue with handcrafted features only.")
        return
    
    if not TORCH_AVAILABLE:
        print("\n⚠️  PyTorch is not installed.")
        print("To use CNN features, install PyTorch:")
        print("  pip install torch torchvision")
        print("\nThe pipeline will continue with handcrafted features only.")
        return
    
    # Load labels
    labels_path = Path('../data/starter/labels.csv')
    if not labels_path.exists():
        print(f"\n❌ Labels file not found: {labels_path}")
        return
    
    df = pd.read_csv(labels_path)
    
    # Get image paths
    image_paths = []
    for filepath in df['filepath']:
        img_path = Path(filepath)
        if not img_path.exists():
            img_path = Path('..') / img_path
        image_paths.append(img_path)
    
    print(f"\n📊 Found {len(image_paths)} images")
    
    # Extract CNN features
    output_dir = Path('../data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f'cnn_features_{args.model}.npy'
    features, valid_indices = extract_cnn_features_from_dataset(
        image_paths,
        output_path,
        args.model
    )
    
    if features is not None:
        # Save valid indices
        valid_indices_path = output_dir / f'cnn_valid_indices_{args.model}.npy'
        np.save(valid_indices_path, valid_indices)
        
        # Summary
        print("\n" + "=" * 70)
        print("✅ CNN FEATURE EXTRACTION COMPLETE")
        print("=" * 70)
        print(f"  Model: {args.model}")
        print(f"  Successfully processed: {len(features)}/{len(image_paths)} images")
        print(f"  Feature dimension: {features.shape[1]}")
        print(f"\n📁 Output files:")
        print(f"  - Features: {output_path}")
        print(f"  - Valid indices: {valid_indices_path}")
        print(f"\n💡 These features can be combined with handcrafted features during training.")
    else:
        print("\n❌ CNN feature extraction failed")
        print("The pipeline will continue with handcrafted features only.")


if __name__ == '__main__':
    main()
