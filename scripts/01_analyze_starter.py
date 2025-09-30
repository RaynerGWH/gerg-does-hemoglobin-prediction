"""
Analyze the 30 starter images
Run: python scripts/01_analyze_starter.py
"""

import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_starter_images():
    # Configuration
    STARTER_DIR = Path('data/starter/images')
    OUTPUT_DIR = Path('results/figures')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STARTER DATASET ANALYSIS")
    print("=" * 60)
    
    # Check if directory exists
    if not STARTER_DIR.exists():
        print(f"\n❌ Directory not found: {STARTER_DIR}")
        print("Please add your 30 images to data/starter/images/")
        return
    
    # Get all image files
    image_files = sorted([f for f in STARTER_DIR.iterdir() 
                         if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    print(f"\n📊 Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print("❌ No images found!")
        return
    
    # Analyze each image
    stats = []
    
    print("\n" + "=" * 60)
    print("IMAGE DETAILS:")
    print("=" * 60)
    
    for idx, img_path in enumerate(image_files, 1):
        img = Image.open(img_path)
        img_array = np.array(img)
        
        # Get dimensions
        width, height = img.size
        
        # Calculate color statistics
        if len(img_array.shape) == 3:
            mean_r = np.mean(img_array[:, :, 0])
            mean_g = np.mean(img_array[:, :, 1])
            mean_b = np.mean(img_array[:, :, 2])
        else:
            mean_r = mean_g = mean_b = np.mean(img_array)
        
        stats.append({
            'filename': img_path.name,
            'width': width,
            'height': height,
            'aspect_ratio': width / height,
            'mean_red': mean_r,
            'mean_green': mean_g,
            'mean_blue': mean_b,
            'file_size_kb': img_path.stat().st_size / 1024
        })
        
        print(f"{idx:2d}. {img_path.name:40s} | {width:4d}x{height:4d} | "
              f"RGB: ({mean_r:5.1f}, {mean_g:5.1f}, {mean_b:5.1f})")
    
    # Convert to DataFrame
    df = pd.DataFrame(stats)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS:")
    print("=" * 60)
    print(f"\n📐 Image Dimensions:")
    print(f"  Width  - Min: {df['width'].min()}, Max: {df['width'].max()}, Mean: {df['width'].mean():.0f}")
    print(f"  Height - Min: {df['height'].min()}, Max: {df['height'].max()}, Mean: {df['height'].mean():.0f}")
    
    print(f"\n🎨 Color Statistics (0-255):")
    print(f"  Red   - Mean: {df['mean_red'].mean():.1f} ± {df['mean_red'].std():.1f}")
    print(f"  Green - Mean: {df['mean_green'].mean():.1f} ± {df['mean_green'].std():.1f}")
    print(f"  Blue  - Mean: {df['mean_blue'].mean():.1f} ± {df['mean_blue'].std():.1f}")
    
    print(f"\n💾 File Sizes:")
    print(f"  Min: {df['file_size_kb'].min():.1f} KB")
    print(f"  Max: {df['file_size_kb'].max():.1f} KB")
    print(f"  Mean: {df['file_size_kb'].mean():.1f} KB")
    
    # Save statistics
    stats_path = OUTPUT_DIR / 'starter_images_stats.csv'
    df.to_csv(stats_path, index=False)
    print(f"\n💾 Saved statistics to: {stats_path}")
    
    # Visualize sample images
    n_samples = min(12, len(image_files))
    rows = 3
    cols = 4
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx in range(n_samples):
        img = Image.open(image_files[idx])
        axes[idx].imshow(img)
        axes[idx].set_title(f"{image_files[idx].name}\n{img.size[0]}x{img.size[1]}", 
                           fontsize=8)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    viz_path = OUTPUT_DIR / 'starter_images_overview.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved visualization to: {viz_path}")
    plt.close()
    
    # Color distribution plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(df['mean_red'], bins=20, color='red', alpha=0.7, edgecolor='black')
    axes[0].set_title('Red Channel Distribution')
    axes[0].set_xlabel('Mean Red Value')
    
    axes[1].hist(df['mean_green'], bins=20, color='green', alpha=0.7, edgecolor='black')
    axes[1].set_title('Green Channel Distribution')
    axes[1].set_xlabel('Mean Green Value')
    
    axes[2].hist(df['mean_blue'], bins=20, color='blue', alpha=0.7, edgecolor='black')
    axes[2].set_title('Blue Channel Distribution')
    axes[2].set_xlabel('Mean Blue Value')
    
    plt.tight_layout()
    color_path = OUTPUT_DIR / 'color_distributions.png'
    plt.savefig(color_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved color analysis to: {color_path}")
    plt.close()
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check results/figures/ for visualizations")
    print("2. Review the statistics CSV")
    print("3. Start searching for external datasets")
    
    return df

if __name__ == "__main__":
    analyze_starter_images()