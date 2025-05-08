from sklearn.cluster import KMeans
import numpy as np

def compute_palette_kmeans(image, n_colors=8, random_state=42, n_init=10):
    """
    Compute the color palette of an image using k-means clustering.
    
    Args:
        image: Input image to compute the palette for.
        n_colors: Number of colors in the palette.
        random_state: Random state for reproducibility.
        n_init: Number of times the k-means algorithm will be run with different centers.
    Returns:
        palette: Computed color palette.
    """
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)
    
    # Use k-means clustering to find a limited set of representative colors
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Get the colors from the cluster centers
    palette = kmeans.cluster_centers_.astype(np.uint8)
    
    return palette