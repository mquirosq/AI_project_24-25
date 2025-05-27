from image_genetic_problem import ImagePaletteGeneticProblem 
from helper import (
    generate_random_palette_with_colors, plot_fitness_evolution
)
import os
import numpy as np
import random
import cv2

class ImagePaletteGeneticProblemRestricted(ImagePaletteGeneticProblem):
    """
    Genetic algorithm for finding an optimal color palette for an image where the colors must exist in the originalimage.
    """
    
    def __init__(self, image_path, num_colors=5, kMeans=False, mutate_diverse=False, crossover_method='one-point', save_results=True, display=True, use_caching=True, results_dir=None):
        """
        Initialize the problem with an image and palette size.
        
        Args:
            image_path: Path to the image file
            num_colors: Number of colors in each palette
            kMeans: Whether to use KMeans for initial palette generation
            mutate_diverse: Whether to use diverse mutation
            crossover_method: Method for crossover ('one-point', 'two-point', 'uniform')
            save_results: Whether to save results
            display: Whether to display results
            use_caching: Whether to use caching for results
            results_dir: Directory to save results, defaults to a subdirectory in the image's directory
        """
        if results_dir is not None:
            self.results_dir = results_dir
        else:
            self.results_dir = os.path.join(os.path.dirname(image_path), "tests", os.path.splitext(os.path.basename(image_path))[0], "restricted")

        super().__init__(image_path, num_colors, kMeans, mutate_diverse, crossover_method, save_results=save_results, display=display, use_caching=use_caching, results_dir=self.results_dir)

        self.colors = np.unique(self.image.reshape(-1, self.image.shape[2]), axis=0)
        
    
    def generate_individual(self):
        """
        Generate a random color palette with the colors in the original image.
        
        Returns:
            A random color palette with the specified number of colors from the original image.
        """ 
        return generate_random_palette_with_colors(self.num_colors, self.colors)
    
    def mutate_random(self, individual):
        """
        Mutate a palette by replacing some colors with random ones from the original image palette.

        Args:
            individual: A color palette (array of RGB colors).
        Returns:
            The given palette individual mutated.
        """

        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = random.choice(self.colors)
        
        return individual
    
    def mutate_diverse(self, individual):
        """
        Mutate a palette by replacing the closest two colors with random ones from the palette of the original image.
        This method uses LAB color space to compute the distance between colors.

        Args:
            individual: A color palette (array of RGB colors).
        Returns:
            The given palette individual mutated.
        """

        # Get the two closest colors in the palette using LAB color space
        individual_lab = cv2.cvtColor(individual.reshape(1, -1, 3), cv2.COLOR_RGB2LAB)[0]
        distances = np.linalg.norm(individual_lab[:, np.newaxis] - individual_lab, axis=2)
        np.fill_diagonal(distances, np.inf)
        closest_indices = np.unravel_index(np.argmin(distances), distances.shape)
        
        # Mutate the closest colors with a certain probability
        for i in closest_indices:
            if random.random() < self.mutation_rate:
                individual[i] = random.choice(self.colors)

        return individual


# Usage example
if __name__ == "__main__":
    
    # Example configuration
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "squareFour.jpg")
    num_colors = 16
    population_size = 20
    generations = 40

    # Create and run the genetic algorithm
    problem = ImagePaletteGeneticProblemRestricted(image_path, num_colors, kMeans=False, mutate_diverse=True, crossover_method='closest_pairs', save_results=True, display=True, use_caching=True)
    
    best_palette, best_fitness, fitness_history, average_fitness_history, best_image, palette, kmeans_image, statistics = problem.run(
        population_size=population_size,
        generations=generations,
        mutation_rate=0.2,
        crossover_rate=0.8,
        elitism=2,
        selection_method='tournament',
        tournament_size=3,
        save_results=True,
        adaptation_rate=1,
        adaptation_threshold=10,
        halting_stagnation_threshold=None
    )
    
    print("\nGenetic algorithm completed.")
    print(f"Best palette: {best_palette}")
    print(f"Best palette fitness: {best_fitness:.6f}")
    
    plot_fitness_evolution(fitness_history, save_path = problem.results_dir)