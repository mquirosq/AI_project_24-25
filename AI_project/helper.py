import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ----- Image loading and processing
def load_image(image_path):
    """
    Load an image from the specified path and convert it to RGB format.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        RGB image as a numpy array
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: '{image_path}'")
    
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Failed to load image: '{image_path}'. The file may be corrupted or in an unsupported format.")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def load_all_images(images_dir):
    """
    Load all images from the specified directory.
    
    Args:
        images_dir: Directory containing image files
    
    Returns:
            List of tuples containing image filenames and their corresponding RGB images
    """
    images = []
    
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_dir, filename)
            image = load_image(image_path)
            if image is not None:
                images.append((filename, image))
    
    return images

# ----- Operations on images

def compute_palette(image):
    """
    Compute the color palette of an image.
    
    Args:    
        image: Input image to compute the palette from
        
    Returns:
        unique_colors: Array of unique colors in the image
    """
    # Reshape the image to be a list of pixels
    # Each row represents one pixel and each pixel has 3 values (R, G, B)
    pixels = image.reshape(-1, 3)
    
    unique_colors = np.unique(pixels, axis=0)
    return unique_colors

def convert_image_to_palette(image, palette):
    """
    Convert an image to a given color palette using Euclidean distance.
    
    Args:
        image: Input image to convert
        palette: Color palette to use for conversion
    
    Returns:
        Converted image with colors from the palette
    """
    # Convert to float to avoid overflow
    image = image.astype(np.float32)
    palette = palette.astype(np.float32)
    
    pixels = image.reshape(-1, 3)
    
    # Use numpy to compute the Euclidean distance between each pixel and each color in the palette
    # By using this expression we get a x10 speedup compared to the for loop
    # np.newaxis adds a new axis to the array. Numpy boradcasts the arrays to make them compatible for subtraction
    # So we get (n_pixels, n_colors, 3), that with the sqrt give us (n_pixels, n_colors)
    # That is, one distance for each pixel to each color in the palette
    distances = np.sqrt(np.sum((pixels[:, np.newaxis, :] - palette[np.newaxis, :, :]) ** 2, axis=2))
    
    closest_color_indices = np.argmin(distances, axis=1)
    
    converted_pixels = palette[closest_color_indices]
    
    converted_image = converted_pixels.reshape(image.shape)
    
    return converted_image.astype(np.uint8)  # Convert back to uint8 for display


def generate_random_palette(num_colors):
    """
    Generate a random color palette.
    
    Args:
        num_colors: Number of colors in the palette
    Returns:
        palette: Random color palette as a numpy array
    """
    
    palette = np.random.randint(0, 256, size=(num_colors, 3), dtype=np.uint8)
    return palette

def generate_random_palette_with_colors(num_colors, colors):
    """
    Generate a random color palette from a set of specified colors.

    Args:
        num_colors: Number of colors in the palette
        colors: List of available colors to choose from
    Returns:
        palette: Random color palette as a numpy array
    """
    num_available = len(colors)
    
    if num_available < num_colors:
        print(f"Warning: Requested {num_colors} unique colors, but only {num_available} are available.")
        print("Some colors will be duplicated in the palette.")
        # Fall back to original method if not enough colors
        return generate_random_palette(num_colors)
    else:
        indices = np.random.choice(num_available, num_colors, replace=False)

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
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions to compute Euclidean distance.")
    
    distance = np.sqrt(np.sum((image1 - image2) ** 2))
    
    return distance


# ----- Display functions

def display_image (image, palette = None, window_name='Image'):
    """
    Display an image in a window.
    
    Args:
        image: Input image to display
        palette: Color palette to display (if None, does not display)
        window_name: Title for the window

    Returns:
        None. Displays the image in a window.
    """
    if palette != None:
        window_name = f"{window_name} ({len(palette)} colors)"
    
    plt.imshow(image)
    plt.axis('off')
    plt.title(window_name)
    plt.show()

def display_palette(palette):
    """
    Display the color palette in a window.
    
    Args:
        palette: Color palette to display
    
    Returns:
        None. Displays the palette in a window.
    """
    # Create a blank image to display the palette
    palette_image = np.zeros((100, 100 * len(palette), 3), dtype=np.uint8)
    
    # Fill the palette image with the colors
    for i in range(len(palette)):
        palette_image[:, i * 100:(i + 1) * 100] = palette[i]
    
    plt.imshow(palette_image)
    plt.axis('off') 
    plt.title('Color Palette')
    plt.show()

def display_image_with_palette(image, palette=None, title="Image with Color Palette"):
    """
    Display an image and its color palette in the same window.
    
    Args:
        image: Input image to display
        palette: Color palette to display (if None, computes from image)
        title: Title for the figure
    
    Returns:
        None. Displays the image and palette in a window.
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
    palette_height = image.shape[0]
    box_height = palette_height // num_colors
    
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
    
    fig.suptitle(title, fontsize=16)
    plt.show()

def display_image_with_palette_comparison(original, image, palette=None, title="Image with Color Palette", save_path=None):
    """
    Display an original image and a converted image with its color palette.
    
    Args:
        original: Original image to display
        image: Converted image to display
        palette: Color palette to display (if None, computes from image)
        title: Title for the figure
        save_path: Path to save the figure (if None, does not save)

    Returns:
        None. Displays a figure with two rows:
            - The first row contains the original image and the converted image.
            - The second row contains the color palette.
    """
    
    # Prepare palettes
    if palette is None:
        palette = compute_palette(image)

    originalPalette = compute_palette(original)
    num_colors = len(palette)
    
    # Create figures
    color_width = min(200, max(30, 800 // num_colors))
    palette_width = color_width * num_colors
    palette_height = 50 
    
    fig = plt.figure(figsize=(12, 8))
    
    gs = plt.GridSpec(2, 2, height_ratios=[4, 1], hspace=0.3)
    
    # Display the original image in top-left
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(original)
    ax_orig.set_title(f"Original Image ({len(originalPalette)} colors)")
    ax_orig.axis('off')

    # Display the converted image in top-right
    ax_conv = fig.add_subplot(gs[0, 1])
    ax_conv.imshow(image)
    ax_conv.set_title(f"Converted Image ({num_colors} colors)")
    ax_conv.axis('off')
    
    # Create palette display in the second row spanning both columns
    ax_palette = fig.add_subplot(gs[1, :])
    
    palette_img = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
    
    for i in range(num_colors):
        start_x = i * color_width
        end_x = min(start_x + color_width, palette_width)
        palette_img[:, start_x:end_x] = palette[i]

    ax_palette.imshow(palette_img)
    ax_palette.set_title(f"Color Palette ({num_colors} colors)")
    ax_palette.axis('off')
    

    fig.suptitle(title, fontsize=16, y=0.98)
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_fitness_evolution(fitness_history, save_path=None, display=True):
    """
    Plot the evolution of fitness over generations.
    
    Args:
        fitness_history: List of fitness values for each generation
        save_path: Path to save the plot (if None, does not save)
        display: Whether to display the plot (default: True)
    Returns:
        None. Displays the plot of fitness evolution.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(fitness_history) + 1), fitness_history)
    plt.title('Fitness Evolution')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path + "/fitness_history.png", bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")
    
    if display:
        plt.show()
    else:
        plt.close()


# ----- Saving images and palettes

def save_image(image, filename):
    """
    Save an image to a file.
    
    Args:
        image: Input image to save
        filename: Path to save the image file
    Returns:
        None. Saves the image to the specified file.
    """
    # Convert the image from RGB to BGR format for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(filename, image_bgr)

def save_palette(palette, filename):
    """
    Save a color palette to an image file.
    Args:
        palette: Color palette to save
        filename: Path to save the palette image file
    Returns:
        None. Saves the palette to the specified file.
    """
    # Create palette image
    palette_image = np.zeros((100, 100 * len(palette), 3), dtype=np.uint8)
    
    for i in range(len(palette)):
        palette_image[:, i * 100:(i + 1) * 100] = palette[i]
    
    # Add the rgb values as text
    for i, color in enumerate(palette):
        text = f"{color[0]}, {color[1]}, {color[2]}"
        cv2.putText(palette_image, text, (i * 100 + 10, 50), cv2.FONT_HERSHEY_PLAIN, 0.75, (255, 255, 255), 1)

    save_image(palette_image, filename)

def save_image_with_palette(image, palette, filename="image_with_palette.png", fitness=None):
    """
    Save an image with a color palette to a file
    
    Args:
        image: Input image to save
        palette: Color palette to display
        filename: Path to save the image file
        fitness: Fitness value to display in the title (if None, does not display)
    Returns:
        None. Saves the image and palette to the specified file.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})
    
    # Display the image in the first subplot
    axs[0].imshow(image)
    axs[0].set_title("Image")
    axs[0].axis('off')
    
    # Create palette display image
    num_colors = len(palette)
    palette_height = image.shape[0]
    box_height = palette_height // num_colors
    
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

    if fitness is not None:
        axs[0].set_title(f"Image (Fitness: {fitness:.2f})")

    fig.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)