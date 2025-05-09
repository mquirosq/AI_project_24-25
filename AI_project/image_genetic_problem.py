from genetic_problem import GeneticProblem 
from helper import (
    load_image, save_image, convert_image_to_palette,
    generate_random_palette, save_palette, euclidean_distance,
    display_image_with_palette, display_image_with_palette_comparison
)
import os
import numpy as np
import random
from skimage import color
from heuristics import compute_palette_kmeans

class ImagePaletteGeneticProblem(GeneticProblem):
    """
    Genetic algorithm for finding an optimal color palette for an image.
    """
    
    def __init__(self, image_path, num_colors=5, cache_size=1000, kMeans=False):
        """
        Initialize the problem with an image and palette size.
        
        Args:
            image_path: Path to the image file
            num_colors: Number of colors in each palette
            cache_size: Maximum size of fitness cache
        """
        super().__init__()
        self.image = load_image(image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        
        if kMeans:
            palette = compute_palette_kmeans(self.image, num_colors)
            kmeans_image = convert_image_to_palette(self.image, palette)
            save_image(kmeans_image, os.path.join(results_dir, "kmeans_image.png"))
            save_palette(palette, os.path.join(results_dir, "kmeans_palette.png"))
            display_image_with_palette_comparison(self.image, kmeans_image, palette, "Initial Palette")

            print(f"Population size: {population_size}, Colors per palette: {num_colors}, Generations: {generations}")
            print(f"KMeans fitness: {self.compute_fitness(palette)}")
            
        self.image_path = image_path
        self.num_colors = num_colors
        self._image_name = os.path.basename(image_path)
        self.kMeans = kMeans
        self.cache_size = cache_size
    
    def generate_individual(self):
        """Generate a random color palette."""
        return generate_random_palette(self.num_colors)
    
    def compute_fitness_rgb(self, individual):
        """
        Calculate fitness of a palette based on how well it represents the image.
        Lower distance means higher fitness.
        """
        # Convert the image using the palette
        converted_image = convert_image_to_palette(self.image, individual)
        
        # Calculate distance between original and converted image
        distance = euclidean_distance(self.image, converted_image)
        
        # Convert distance to fitness (lower distance = higher fitness)
        fitness = - distance
        
        return fitness
    
    def compute_fitness(self, individual):
        # Convert image and palette to LAB color space
        image_lab = color.rgb2lab(self.image / 255.0)
        
        # Convert image using palette
        converted_image = convert_image_to_palette(self.image, individual)
        converted_lab = color.rgb2lab(converted_image / 255.0)
        
        # Calculate distance in LAB space (perceptually uniform)
        distance = np.mean(np.sqrt(np.sum((image_lab - converted_lab)**2, axis=2)))
        
        return 1 / (1 + distance)
    
    def crossover(self, parent1, parent2):
        """
        Perform one-point crossover between two palettes.
        """
        # Convert to numpy arrays if they aren't already
        parent1_array = np.array(parent1)
        parent2_array = np.array(parent2)
        
        # Choose a crossover point
        crossover_point = random.randint(1, self.num_colors - 1)
        
        # Create child by combining parts of both parents
        child = np.concatenate([
            parent1_array[:crossover_point], 
            parent2_array[crossover_point:self.num_colors]
        ])
        
        return child
    
    def uniform_crossover(self, parent1, parent2):
        """
        Perform uniform crossover between two palettes.
        Each color has a 50% chance of coming from either parent.
        """
        # Convert to numpy arrays
        parent1_array = np.array(parent1)
        parent2_array = np.array(parent2)
        
        # Create mask for uniform crossover (50% chance for each parent)
        mask = np.random.randint(0, 2, size=self.num_colors, dtype=bool)
        
        # Create child array
        child = np.zeros((self.num_colors, 3), dtype=np.uint8)
        
        # Apply mask to select colors from parents
        child[mask] = parent1_array[mask]
        child[~mask] = parent2_array[~mask]
        
        return child
    
    def mutate(self, individual, mutation_rate):
        """
        Mutate a palette by replacing some colors with random ones.
        """
        individual = np.array(individual)  # Ensure it's a numpy array
        
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                # Replace with a random color
                individual[i] = np.random.randint(0, 256, size=3, dtype=np.uint8)
        
        return individual
    
    def mutate_rgb_channels(self, individual, mutation_rate, max_change=30):
        """
        Mutate a palette by modifying RGB channels slightly.
        """
        individual = np.array(individual)  # Ensure it's a numpy array
        
        for i in range(len(individual)):
            # For each color, decide if we want to mutate it
            if random.random() < mutation_rate:
                # For each channel (R,G,B), make a small random change
                delta = np.random.randint(-max_change, max_change+1, size=3)
                individual[i] = np.clip(individual[i] + delta, 0, 255).astype(np.uint8)
        
        return individual
    
    def save_generation_results(self, best_palette, generation, results_dir):
        """Save the best palette and converted image for this generation."""
        try:
            # Create numbered filenames with leading zeros
            gen_num = f"{generation + 1:03d}"  # e.g., 001, 002, etc.
            
            images_dir = os.path.join(results_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

            # Convert the image using this palette
            converted_image = convert_image_to_palette(self.image, best_palette)
            
            # Save the palette
            palette_filename = os.path.join(images_dir, f"palette_gen_{gen_num}.png")
            save_palette(best_palette, palette_filename)
            
            # Save the converted image
            image_filename = os.path.join(images_dir, f"image_gen_{gen_num}.png")
            save_image(converted_image, image_filename)
            
            print(f"Saved results for generation {generation + 1}")
        except Exception as e:
            print(f"Error saving generation results: {e}")
    
    def save_best_results(self, best_palette, results_dir):
        """Save the best overall palette and converted image."""
        try:
            # Save the best palette
            palette_filename = os.path.join(results_dir, "best_palette_overall.png")
            save_palette(best_palette, palette_filename)
            
            # Convert and save the image
            final_image = convert_image_to_palette(self.image, best_palette)
            image_filename = os.path.join(results_dir, "best_image_overall.png")
            save_image(final_image, image_filename)
            
            # Show the final result
            display_image_with_palette_comparison(self.image, final_image, best_palette, "Best Overall Palette")
            
            print("Saved overall best results")
        except Exception as e:
            print(f"Error saving best results: {e}")

    def initialize_population(self, size):
        """Initialize a population of random palettes."""
        population = []
        if self.kMeans:
            # Use KMeans to generate a palette
            kmeans_palette = compute_palette_kmeans(self.image, self.num_colors)
            population.append(kmeans_palette)
            size = size-1
        for _ in range(size):
            individual = self.generate_individual()
            population.append(individual)
        return population

    def getBestResult(self, best_individual):
        """Get the best result from the best individual."""
        converted_image = convert_image_to_palette(self.image, best_individual)
        return converted_image

# Usage example
if __name__ == "__main__":
    # Example configuration
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "nasa.jpg")
    num_colors = 16
    population_size = 20
    generations = 40
    
    # Create results directory based on image name
    # This will create a directory named "tests/imageName/unrestricted" in the same directory as the image
    results_dir = os.path.join(os.path.dirname(image_path), "tests", os.path.splitext(os.path.basename(image_path))[0], "unrestricted")

    # Create and run the genetic algorithm
    problem = ImagePaletteGeneticProblem(image_path, num_colors, kMeans=False)
    
    best_palette, best_fitness, fitness_history, bestImage = problem.run(
        population_size=population_size,
        generations=generations,
        mutation_rate=0.2,
        crossover_rate=0.8,
        elitism=2,
        selection_method='tournament',
        save_results=True,
        results_dir=results_dir
    )
    
    print("\nGenetic algorithm completed.")
    print(f"Best palette: {best_palette}")
    print(f"Best palette fitness: {best_fitness:.6f}")
    
    # Plot fitness evolution
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, generations+1), fitness_history)
    plt.title('Fitness Evolution')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "fitness_evolution.png"))
    plt.show()