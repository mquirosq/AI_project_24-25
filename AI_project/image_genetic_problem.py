from genetic_problem import GeneticProblem 
from helper import (
    load_image, save_image_with_palette, convert_image_to_palette,
    generate_random_palette, save_palette, euclidean_distance,
    plot_fitness_evolution, display_image_with_palette_comparison
)
import os
import numpy as np
import random
from skimage import color
from heuristics import compute_palette_kmeans
import cv2

class ImagePaletteGeneticProblem(GeneticProblem):
    """
    Genetic algorithm for finding an optimal color palette for an image.
    """
    
    def __init__(self, image_path, num_colors=5, cache_size=1000, kMeans=False, mutate_diverse=False, crossover_method='uniform', save_results=True):
        """
        Initialize the problem with an image and palette size.
        
        Args:
            image_path: Path to the image file
            num_colors: Number of colors in each palette
            cache_size: Maximum size of fitness cache
        """
        if not hasattr(self, 'results_dir'):
            self.results_dir = os.path.join(os.path.dirname(image_path), "tests", os.path.splitext(os.path.basename(image_path))[0], "unrestricted")
        super().__init__(self.results_dir, save_results=save_results)
        
        self.image = load_image(image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        
        if kMeans:
            palette = compute_palette_kmeans(self.image, num_colors)
            kmeans_image = convert_image_to_palette(self.image, palette)
            save_image_with_palette(kmeans_image, palette, os.path.join(self.results_dir, "kmeans_image.png"))
            display_image_with_palette_comparison(self.image, kmeans_image, palette, "Initial kMeans Palette")

            print(f"Population size: {population_size}, Colors per palette: {num_colors}, Generations: {generations}")
            print(f"KMeans fitness: {self.compute_fitness(palette)}")
            
        self.image_path = image_path
        self.num_colors = num_colors
        self._image_name = os.path.basename(image_path)
        self.kMeans = kMeans
        self.cache_size = cache_size
        self.use_mutate_diverse = mutate_diverse
        self.crossover_method = crossover_method
    
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
        """
        Calculate fitness of a palette based on how well it represents the image.
        Lower distance means higher fitness.
        Add a penalty for repeated colors in the palette, as they should be minimized.
        """
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
        Perform crossover between two parents to create a child.
        The method of crossover is determined by the crossover_method attribute.
        """
        if self.crossover_method == 'uniform':
            return self.uniform_crossover(parent1, parent2)
        elif self.crossover_method == 'one_point':
            return self.one_point_crossover(parent1, parent2)
        elif self.crossover_method == 'closest_pairs':
            return self.crossover_closest_pairs(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover method: {self.crossover_method}")

    def one_point_crossover(self, parent1, parent2):
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
    
    def crossover_closest_pairs(self, parent1, parent2):
        """
        Perform crossover between two parents to create a child.
        Match the closest colors from each parent, for each pair choose one color randomly.
        Ensure no duplicates in the child.
        """
        child = np.zeros_like(parent1)
        for i in range(len(parent1)):
            # Find the closest color in parent2 to the current color in parent1
            distances = np.linalg.norm(parent2 - parent1[i], axis=1)
            closest_index = np.argmin(distances)
            
            # Randomly choose between the two colors
            if random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[closest_index]

        # Try to avoid duplicates in the child
        unique_colors = np.unique(child, axis=0)
        if len(unique_colors) < len(child):
            # If duplicates exist, randomly select from colors in any palette that are not in the child
            available_colors = np.array([color for color in self.colors if not np.any(np.all(color == child, axis=1))])
            if len(available_colors) < len(child):
                # If not enough unique colors, fill with random colors from the original image palette
                child = unique_colors[np.random.choice(len(unique_colors), len(child), replace=True)]
            else:
                child = np.array([available_colors[np.random.choice(len(available_colors))] for _ in range(len(child))])
        
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
    
    def mutate_diverse(self, individual, mutation_rate):
        """
        Mutate a palette by replacing the closest two colors with random ones.
        """
        # Get the two closest colors in the palette using LAB color space
        individual_lab = cv2.cvtColor(individual.reshape(1, -1, 3), cv2.COLOR_RGB2LAB)[0]
        distances = np.linalg.norm(individual_lab[:, np.newaxis] - individual_lab, axis=2)
        np.fill_diagonal(distances, np.inf)
        closest_indices = np.unravel_index(np.argmin(distances), distances.shape)
        # Mutate the closest colors with a certain probability
        for i in closest_indices:
            if random.random() < mutation_rate:
                individual[i] = np.random.randint(0, 256, size=3, dtype=np.uint8)

        return individual
    
    def save_generation_results(self, best_palette, generation):
        """Save the best palette and converted image for this generation."""
        try:
            # Create numbered filenames with leading zeros
            gen_num = f"{generation + 1:03d}"  # e.g., 001, 002, etc.
            
            images_dir = os.path.join(self.results_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

            # Convert the image using this palette
            converted_image = convert_image_to_palette(self.image, best_palette)
            
            # Save the palette and converted image
            image_filename = os.path.join(images_dir, f"image_gen_{gen_num}.png")
            save_image_with_palette(converted_image, best_palette, image_filename)
            
            print(f"Saved results for generation {generation + 1}")
        except Exception as e:
            print(f"Error saving generation results: {e}")
    
    def save_best_results(self, best_palette):
        """Save the best overall palette and converted image."""
        try:
            # Convert and save the image and the best palette
            final_image = convert_image_to_palette(self.image, best_palette)
            image_filename = os.path.join(self.results_dir, "best_image_overall.png")
            save_image_with_palette(final_image, best_palette, image_filename)
            
            # Show the final result
            display_image_with_palette_comparison(self.image, final_image, best_palette, "Best Overall Palette", self.results_dir + "/final_comparison.png")
            
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
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "houses.jpg")
    num_colors = 16
    population_size = 20
    generations = 40

    # Create and run the genetic algorithm
    problem = ImagePaletteGeneticProblem(image_path, num_colors, kMeans=False, mutate_diverse=True, crossover_method='closest_pairs')
    
    best_palette, best_fitness, fitness_history, bestImage = problem.run(
        population_size=population_size,
        generations=generations,
        mutation_rate=0.2,
        crossover_rate=0.8,
        elitism=2,
        selection_method='tournament',
        save_results=True
    )
    
    print("\nGenetic algorithm completed.")
    print(f"Best palette: {best_palette}")
    print(f"Best palette fitness: {best_fitness:.6f}")
    
    # Plot fitness evolution
    plot_fitness_evolution(fitness_history, save_path = problem.results_dir)