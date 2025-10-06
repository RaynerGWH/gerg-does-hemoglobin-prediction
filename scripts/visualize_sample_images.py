import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from pillow_heif import register_heif_opener

register_heif_opener()

# Get sample images
images_dir = Path('data/starter/images')
image_files = sorted([f for f in images_dir.iterdir() 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.heic']])

# Show first 6 images
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx in range(min(6, len(image_files))):
    img = Image.open(image_files[idx])
    axes[idx].imshow(img)
    axes[idx].set_title(f"{image_files[idx].name}\nSize: {img.size[0]}x{img.size[1]}", 
                       fontsize=10)
    axes[idx].axis('off')

# Hide unused subplots
for idx in range(len(image_files), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('results/sample_images_check.png', dpi=150, bbox_inches='tight')
print("✅ Saved visualization to: results/sample_images_check.png")
print("\nCheck this image to see if your photos show:")
print("  - Full face (need cropping)")
print("  - Just lips (already good)")
plt.show()