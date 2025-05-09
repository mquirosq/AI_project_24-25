import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_image(image_path):
    """Load an image from the specified path and convert it to RGB format."""
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    # Convert the image from BGR to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def load_all_images(images_dir):
    """Load all images from the specified directory."""
    images = []
    
    # Iterate through all files in the directory
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(images_dir, filename)
            image = load_image(image_path)
            if image is not None:
                images.append((filename, image))
    
    return images


def display_image (image, palette = None, window_name='Image'):
    """Display an image in a window."""
    if palette != None:
        window_name = f"{window_name} ({len(palette)} colors)"
    plt.imshow(image)
    plt.axis('off')  # Hide the axis
    plt.title(window_name)
    plt.show()


def compute_palette(image):
    """Compute the color palette of an image."""
    # Reshape the image to be a list of pixels
    # Each row represents one pixel and each pixel has 3 values (R, G, B)
    pixels = image.reshape(-1, 3)
    
    unique_colors = np.unique(pixels, axis=0)
    return unique_colors

def display_palette(palette):
    """Display the color palette."""
    # Create a blank image to display the palette
    palette_image = np.zeros((100, 100 * len(palette), 3), dtype=np.uint8)
    
    # Fill the palette image with the colors
    for i in range(len(palette)):
        palette_image[:, i * 100:(i + 1) * 100] = palette[i]
    
    # Display the palette image
    plt.imshow(palette_image)
    plt.axis('off')  # Hide the axis
    plt.title('Color Palette')
    plt.show()


def display_image_with_palette(image, palette=None, title="Image with Color Palette"):
    """
    Display an image and its color palette in the same window.
    
    Args:
        image: Input image to display
        palette: Color palette to display (if None, computes from image)
        title: Title for the figure
    """
    
    # Compute palette if not provided
    if palette is None:
        palette = compute_palette(image)
    
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})
    
    # Display the image in the first subplot
    axs[0].imshow(image)
    axs[0].set_title("Image")
    axs[0].axis('off')
    
    # Create palette display image
    num_colors = len(palette)
    palette_height = image.shape[0]  # Match image height
    box_height = palette_height // num_colors
    
    # Create palette image with one color per row
    palette_width = 100
    palette_img = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
    
    for i in range(num_colors):
        start_y = i * box_height
        end_y = min(start_y + box_height, palette_height)
        palette_img[start_y:end_y, :] = palette[i]
    
    # Display the palette in the second subplot
    axs[1].imshow(palette_img)
    axs[1].set_title(f"Color Palette ({num_colors} colors)")
    axs[1].axis('off')
    
    # Set the overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def display_image_with_palette_comparison(original, image, palette=None, title="Image with Color Palette", save_path=None):
    """
    Display an original image and a converted image with its color palette.
    
    Args:
        original: Original image to display
        image: Converted image to display
        palette: Color palette to display (if None, computes from image)
        title: Title for the figure
    """
    
    # Compute palette if not provided
    if palette is None:
        palette = compute_palette(image)

    originalPalette = compute_palette(original)
    
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 3, 1]})
    
    # Display the original image in the first subplot
    axs[0].imshow(original)
    axs[0].set_title("Original Image ({} colors)".format(len(originalPalette)))
    axs[0].axis('off')
    
    # Create palette display image
    num_colors = len(palette)
    box_height = original.shape[0] // num_colors
    
    # Create palette image with one color per row
    palette_width = 100
    palette_img = np.zeros((original.shape[0], palette_width, 3), dtype=np.uint8)
    
    for i in range(num_colors):
        start_y = i * box_height
        end_y = min(start_y + box_height, original.shape[0])
        palette_img[start_y:end_y, :] = palette[i]

    image = convert_image_to_palette(image, palette)  # Convert image to palette colors
    
    # Display the converted image in the second subplot
    axs[1].imshow(image)
    axs[1].set_title(f"Converted Image ({num_colors} colors)")
    axs[1].axis('off')

    axs[2].imshow(palette_img)
    axs[2].set_title(f"Color Palette ({num_colors} colors)")
    axs[2].axis('off')
    
    # Set the overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    if save_path:
        # Save the figure to the specified path
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")

def convert_image_to_palette(image, palette):
    """
    Convert an image to a given color palette using Euclidean distance.
    
    Args:
        image: Input image to convert
        palette: Color palette to use for conversion
    
    Returns:
        Converted image with colors from the palette
    """
    # Ensure we have the same data type and that we won't get an overflow
    image = image.astype(np.float32)
    palette = palette.astype(np.float32)
    
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)
    
    # Initialize an array to hold the converted pixels
    converted_pixels = np.zeros_like(pixels)
    
    # For each pixel in the image, find the closest color in the palette
    for i, pixel in enumerate(pixels):
        distances = np.linalg.norm(palette - pixel, axis=1)  # Compute Euclidean distance along rows
        closest_color_index = np.argmin(distances)  # Find index of closest color
        converted_pixels[i] = palette[closest_color_index]  # Assign closest color
    
    # Reshape the converted pixels back to the original image shape
    converted_image = converted_pixels.reshape(image.shape)
    
    return converted_image.astype(np.uint8)  # Convert back to uint8 for display


def generate_random_palette(num_colors):
    """Generate a random color palette."""
    # Generate random colors in the range [0, 255]
    palette = np.random.randint(0, 256, size=(num_colors, 3), dtype=np.uint8)
    return palette

def generate_random_palette_with_colors(num_colors, colors):
    """Generate a random color palette with specified colors."""
    num_available = len(colors)
    
    if num_available == 0:
        raise ValueError("No colors provided to choose from")
    
    # Generate random indices to select colors
    indices = np.random.randint(0, num_available, size=num_colors)
    
    # Use the indices to select colors from the provided set
    palette = np.array([colors[i] for i in indices], dtype=np.uint8)
    return palette

def euclidean_distance(image1, image2):
    """
    Compute the Euclidean distance between two images.
    
    Args:
        image1: First image
        image2: Second image
    
    Returns:
        Euclidean distance between the two images
    """
    # Ensure both images are the same size
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions to compute Euclidean distance.")
    
    # Compute the Euclidean distance
    distance = np.sqrt(np.sum((image1 - image2) ** 2))
    
    return distance

def save_image(image, filename):
    """Save an image to a file."""
    # Convert the image from RGB to BGR format for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Save the image using OpenCV
    cv2.imwrite(filename, image_bgr)

def save_palette(palette, filename):
    """Save a color palette to a file."""
    # Create a blank image to display the palette
    palette_image = np.zeros((100, 100 * len(palette), 3), dtype=np.uint8)
    
    # Fill the palette image with the colors
    for i in range(len(palette)):
        palette_image[:, i * 100:(i + 1) * 100] = palette[i]
    
    # Save the palette image using OpenCV
    save_image(palette_image, filename)