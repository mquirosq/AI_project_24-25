from sklearn.cluster import KMeans
import numpy as np

def compute_palette_kmeans(image, n_colors=8):
    """
    Compute the color palette of an image using k-means clustering.
    
    Args:
        image: Input image to compute the palette for.
        n_colors: Number of colors in the palette.
    Returns:
        palette: Computed color palette.
    """
    pixels = image.reshape(-1, 3)
    
    # Use k-means clustering to find colors
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Get the colors from the cluster centers
    palette = kmeans.cluster_centers_.astype(np.uint8)
    
    return palette