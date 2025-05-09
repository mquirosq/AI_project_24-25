from image_genetic_problem import ImagePaletteGeneticProblem 
from helper import (
    generate_random_palette_with_colors
)
import os
import numpy as np
import random

# Restricts the colors on the palette to be in the original image

class ImagePaletteGeneticProblemRestricted(ImagePaletteGeneticProblem):
    """
    Genetic algorithm for finding an optimal color palette for an image where the colors must be part of the image.
    """
    
    def __init__(self, image_path, num_colors=5, cache_size=1000, kMeans=False):
        """
        Initialize the problem with an image and palette size.
        
        Args:
            image_path: Path to the image file
            num_colors: Number of colors in each palette
            cache_size: Maximum size of fitness cache
            kMeans: Whether to use KMeans for initial palette generation
        """
        super().__init__(image_path, num_colors, cache_size, kMeans)
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
        Mutate a palette by replacing some colors with random ones.
        """
        individual = np.array(individual)  # Ensure it's a numpy array
        
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                # Replace with a random color
                individual[i] = random.choice(self.colors)
        return individual


# Usage example
if __name__ == "__main__":
    # Example configuration
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "houses.jpg")
    num_colors = 16
    population_size = 10
    generations = 50
    
    # Create results directory based on image name
    # This will create a directory named "tests/imageName/restricted" in the same directory as the image (images folder)
    results_dir = os.path.join(os.path.dirname(image_path), "tests", os.path.splitext(os.path.basename(image_path))[0], "restricted")

    # Create and run the genetic algorithm
    problem = ImagePaletteGeneticProblemRestricted(image_path, num_colors, kMeans=False)
    
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