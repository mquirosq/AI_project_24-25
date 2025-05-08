import os
import numpy as np
import matplotlib.pyplot as plt
from helper import (
    load_image,
    convert_image_to_palette, 
    display_image, 
    display_image_with_palette,
    generate_random_palette
)
from heuristics import compute_palette_kmeans

def test_helpers():
    """Test the helper functions."""
    
    # 1. Load an image from the images directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(project_dir, "images")
    
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return False
    
    # Find image files
    image_files = [f for f in os.listdir(images_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print("No image files found.")
        return False
    
    print(f"Testing with image: {image_files[0]}")
    image_path = os.path.join(images_dir, image_files[0])
    
    # Load the image
    image = load_image(image_path)
    if image is None:
        print("Failed to load the image.")
        return False
    
    print(f"Image loaded successfully with shape {image.shape}")
    
    # 2. Create a small color palette (manually or from the image)
    # Method 1: Create a small test palette manually
    test_palette = np.array([
        [255, 0, 0],   # red
        [0, 255, 0],   # green
        [0, 0, 255],   # blue
        [255, 255, 0], # yellow
        [255, 0, 255], # magenta
        [0, 255, 255]  # cyan    
    ], dtype=np.uint8)

    # Method 2: Compute a limited palette from the image
    # Use k-means clustering to find a limited set of representative colors
    
    # Number of colors to use
    n_colors = 8
    
    print(f"Computing a palette with {n_colors} colors using K-means clustering...")

    # Get the colors from the cluster centers
    kmeans_palette = compute_palette_kmeans(image, n_colors=n_colors)

    # Randomly generate a palette for testing
    random_palette = generate_random_palette(n_colors)

    
    # 3. Convert the image to both palettes
    print("Converting image to test palette...")
    converted_test = convert_image_to_palette(image, test_palette)
    
    print("Converting image to K-means palette...")
    converted_kmeans = convert_image_to_palette(image, kmeans_palette)

    print("Converting image to random palette...")
    converted_random = convert_image_to_palette(image, random_palette)
    
    # 4. Display the results
    # Original image
    print("Displaying original image...")
    display_image(image, "Original Image")
    
    # Test palette and converted image
    print("Displaying test palette...")
    display_image_with_palette(converted_test, test_palette, "Image with Test Palette")
    
    # K-means palette and converted image
    print("Displaying K-means palette...")
    display_image_with_palette(converted_kmeans, kmeans_palette, "Image with K-means Palette")

    # Random palette and converted image
    print("Displaying random palette...")
    display_image_with_palette(converted_random, random_palette, "Image with Random Palette")
    
    # 5. Display side-by-side comparison
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(converted_test)
    axes[1].set_title("Test Palette")
    axes[1].axis('off')
    
    axes[2].imshow(converted_kmeans)
    axes[2].set_title("K-means Palette (8 colors)")
    axes[2].axis('off')

    axes[3].imshow(converted_random)
    axes[3].set_title("Random Palette")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Conversion test completed successfully!")
    return True

if __name__ == "__main__":
    result = test_helpers()
    print(f"\nPalette conversion test {'passed' if result else 'failed'}")