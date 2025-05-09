from image_genetic_problem import ImagePaletteGeneticProblem 
from helper import (
    generate_random_palette_with_colors, plot_fitness_evolution
)
import os
import numpy as np
import random
import cv2

# Restricts the colors on the palette to be in the original image

class ImagePaletteGeneticProblemRestricted(ImagePaletteGeneticProblem):
    """
    Genetic algorithm for finding an optimal color palette for an image where the colors must be part of the image.
    """
    
    def __init__(self, image_path, num_colors=5, cache_size=1000, kMeans=False, mutate_diverse=False, crossover_method='one-point', save_results=True):
        """
        Initialize the problem with an image and palette size.
        
        Args:
            image_path: Path to the image file
            num_colors: Number of colors in each palette
            cache_size: Maximum size of fitness cache
            kMeans: Whether to use KMeans for initial palette generation
        """
        if not hasattr(self, 'results_dir'):
            self.results_dir = os.path.join(os.path.dirname(image_path), "tests", os.path.splitext(os.path.basename(image_path))[0], "restricted")
        super().__init__(image_path, num_colors, cache_size, kMeans, mutate_diverse, crossover_method, save_results=save_results)
        self.colors = np.unique(self.image.reshape(-1, self.image.shape[2]), axis=0)
    
    def generate_individual(self):
        """Generate a random color palette.""" 
        return generate_random_palette_with_colors(self.num_colors, self.colors)
    
    def compute_fitness(self, individual):
        """
        Calculate fitness of a palette based on how well it represents the image.
        Lower distance means higher fitness.
        """
        # Penalize for repeated colors in the palette
        distance = 0.1 * (self.num_colors - len(np.unique(individual, axis=0)))
        return super().compute_fitness(individual) - distance
    
    def mutate(self, individual, mutation_rate):
        """
        Mutate a palette by replacing some colors with random ones from the original image palette.
        """
        individual = np.array(individual)  # Ensure it's a numpy array
        
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                # Replace with a random color
                individual[i] = random.choice(self.colors)
        return individual
    
    def mutate_diverse(self, individual, mutation_rate):
        """
        Mutate a palette by replacing the closest two colors with random ones from the original image palette.
        """
        # Get the two closest colors in the palette using LAB color space
        individual_lab = cv2.cvtColor(individual.reshape(1, -1, 3), cv2.COLOR_RGB2LAB)[0]
        distances = np.linalg.norm(individual_lab[:, np.newaxis] - individual_lab, axis=2)
        np.fill_diagonal(distances, np.inf)
        closest_indices = np.unravel_index(np.argmin(distances), distances.shape)
        # Mutate the closest colors with a certain probability
        for i in closest_indices:
            if random.random() < mutation_rate:
                individual[i] = random.choice(self.colors)

        return individual


# Usage example
if __name__ == "__main__":
    # Example configuration
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "squareFour.jpg")
    num_colors = 10
    population_size = 20
    generations = 3

    # Create and run the genetic algorithm
    problem = ImagePaletteGeneticProblemRestricted(image_path, num_colors, kMeans=False, mutate_diverse=True, crossover_method='closest_pairs', )
    
    best_palette, best_fitness, fitness_history, bestImage = problem.run(
        population_size=population_size,
        generations=generations,
        mutation_rate=0.2,
        crossover_rate=0.8,
        elitism=2,
        selection_method='tournament',
        save_results=True,
        adaptation_rate=1.2,
        adaptation_threshold=10, 
        halting_stagnation_threshold=20
    )
    
    print("\nGenetic algorithm completed.")
    print(f"Best palette: {best_palette}")
    print(f"Best palette fitness: {best_fitness:.6f}")
    
    # Plot fitness evolution
    plot_fitness_evolution(fitness_history, save_path = problem.results_dir)