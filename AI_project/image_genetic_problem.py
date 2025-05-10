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
import datetime

class ImagePaletteGeneticProblem(GeneticProblem):
    """
    Genetic algorithm for finding an optimal color palette for an image.
    """
    
    def __init__(self, image_path, num_colors=5, cache_size=1000, kMeans=False, mutate_diverse=False, crossover_method='uniform', save_results=True, display=True):
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
        
        self.display = display
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
        If duplicates exist, fill with random color from any of the parents not already on child.
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
            parent_colors = np.concatenate([parent1, parent2], axis=0)
            parent_colors = np.unique(parent_colors, axis=0)
            available_colors = np.array([color for color in parent_colors if not np.any(np.all(color == child, axis=1))])
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
            if self.display: display_image_with_palette_comparison(self.image, final_image, best_palette, "Best Overall Palette", self.results_dir + "/final_comparison.png")
            
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
    
    def save_performance_report(self, performance_log, best_individual, total_time, population_size):
        """
        Save a detailed performance report of the genetic algorithm run.
        
        Args:
            performance_log: List of dictionaries with performance metrics for each generation
            best_individual: The overall best individual found
            total_time: Total execution time in seconds
            results_dir: Directory to save the report
        """
        # Create report filename with timestamp
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        report_filename = os.path.join(self.results_dir, f"performance_report_{timestamp}.txt")
        
        with open(report_filename, 'w') as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write(f"GENETIC ALGORITHM PERFORMANCE REPORT\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Basic information
            f.write("BASIC INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Image Name: {self._image_name}\n")
            f.write(f"Image Path: {self.image_path}\n")
            f.write(f"Number of Colors: {self.num_colors}\n")

            # Write configuration
            f.write("CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Population Size: {population_size}\n")
            f.write(f"Generations: {len(performance_log)}\n")

            f.write(f"Mutation Rate: {self.mutation_rate}\n")
            f.write(f"Crossover Rate: {self.crossover_rate}\n")
            f.write(f"Elitism: {self.elitism}\n")
            f.write(f"Selection Method: {self.selection_method}\n")
            f.write(f"Adaptation Rate: {self.adaptation_rate}\n")
            f.write(f"Adaptation Threshold: {self.adaptation_threshold}\n")
            f.write(f"Halting Stagnation Threshold: {self.halting_stagnation_threshold}\n")
            f.write(f"Population Size: {population_size}\n\n")

            f.write(f"Uses KMeans: {'Yes' if self.kMeans else 'No'}\n")
            f.write(f"Uses Diverse Mutation: {'Yes' if self.use_mutate_diverse else 'No'}\n")
            f.write(f"Crossover Method: {self.crossover_method}\n\n")

            f.write(f"Total Execution Time: {total_time:.2f} seconds\n")
            f.write(f"Average Time per Generation: {total_time/len(performance_log):.2f} seconds\n\n")
            
            # Write best solution
            f.write("BEST SOLUTION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Best Fitness: {max(entry['best_fitness'] for entry in performance_log):.6f}\n")
            f.write(f"Found in Generation: {next(i+1 for i, entry in enumerate(performance_log) if entry['best_fitness'] == max(e['best_fitness'] for e in performance_log))}\n")
            f.write(f"Best Chromosome:\n{self._format_individual(best_individual)}\n\n")
            f.write(f"Total Generations: {len(performance_log)}\n")

            # Write generation statistics with stagnation info
            f.write("GENERATION STATISTICS\n")
            f.write("-" * 110 + "\n")
            f.write("Gen | Best Fitness | Avg Fitness | Min Fitness | Stagnation | Mutation Rate | Adapted | Time (s)\n")
            f.write("-" * 110 + "\n")
            
            for entry in performance_log:
                mutation_adapted = "Yes" if entry.get('mutation_adapted', False) else "No"
                f.write(f"{entry['generation']:3d} | {entry['best_fitness']:12.6f} | {entry['avg_fitness']:11.6f} | "
                    f"{entry['min_fitness']:11.6f} | {entry.get('stagnation_count', 0):10d} | "
                    f"{entry.get('current_mutation_rate', 0.0):12.6f} | {mutation_adapted:7s} | {entry['time_seconds']:8.2f}\n")
            
            # Write cache statistics
            last_entry = performance_log[-1]
            total_evaluations = last_entry['cache_hits'] + last_entry['cache_misses']
            hit_rate = last_entry['cache_hits'] / total_evaluations * 100 if total_evaluations > 0 else 0
            
            f.write("\nCACHE STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Evaluations: {total_evaluations}\n")
            f.write(f"Cache Hits: {last_entry['cache_hits']} ({hit_rate:.2f}%)\n")
            f.write(f"Cache Misses: {last_entry['cache_misses']}\n")
        
        print(f"Performance report saved to {report_filename}")

    def run(self, population_size, generations, mutation_rate=0.1, 
            crossover_rate=0.8, elitism=2, selection_method='roulette',
            save_results=False, adaptation_rate=1, 
            adaptation_threshold=10, halting_stagnation_threshold=None):
        """
        Run the genetic algorithm for a specified number of generations.
        """
        palette = None
        kmeans_image = None
        if self.kMeans:
            palette = compute_palette_kmeans(self.image, self.num_colors)
            kmeans_image = convert_image_to_palette(self.image, palette)
            save_image_with_palette(kmeans_image, palette, os.path.join(self.results_dir, "kmeans_image.png"))
            if self.display: display_image_with_palette_comparison(self.image, kmeans_image, palette, "Initial kMeans Palette")

            print(f"KMeans fitness: {self.compute_fitness(palette)}")
        
        parent_results = super().run(
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elitism=elitism,
            selection_method=selection_method,
            adaptation_rate=adaptation_rate,
            adaptation_threshold=adaptation_threshold,
            halting_stagnation_threshold=halting_stagnation_threshold,
            save_results=save_results
        )
        best_palette, best_fitness, fitness_history, best_image = parent_results
    
        return best_palette, best_fitness, fitness_history, best_image, palette, kmeans_image
        

    def _format_individual(self, individual):
        """Format an individual for readable output in the report."""
        if isinstance(individual, np.ndarray):
            if individual.ndim == 2 and individual.shape[1] == 3:  # RGB colors
                return "\n".join([f"  Color {i+1}: RGB({c[0]}, {c[1]}, {c[2]})" 
                            for i, c in enumerate(individual)])
            return str(individual)
        return str(individual)
    
class ImagePaletteGeneticProblemRGBDistance(ImagePaletteGeneticProblem):
    """
    Genetic algorithm for finding an optimal color palette for an image using RGB distance.
    This class has been created to ilustrate the problem of computing the distance in RGB space 
    and justify the use of lab color space for distance computations.
    """

    def __init__(self, image_path, num_colors=5, cache_size=1000, kMeans=False, mutate_diverse=False, crossover_method='uniform', save_results=True, display=True):
        """
        Initialize the problem with an image and palette size.
        
        Args:
            image_path: Path to the image file
            num_colors: Number of colors in each palette
            cache_size: Maximum size of fitness cache
        """
        if not hasattr(self, 'results_dir'):
            self.results_dir = os.path.join(os.path.dirname(image_path), "tests", os.path.splitext(os.path.basename(image_path))[0], "unrestricted_rgb_distance")
        super().__init__(image_path, num_colors, cache_size, kMeans, mutate_diverse, crossover_method, save_results, display)
    
    def compute_fitness(self, individual):
        """
        Calculate fitness of a palette based on how well it represents the image.
        Lower distance means higher fitness. This method used eculidean distance in RGB space.
        Add a penalty for repeated colors in the palette, as they should be minimized.
        """
        # Convert the image using the palette
        converted_image = convert_image_to_palette(self.image, individual)
        
        # Calculate distance between original and converted image
        distance = euclidean_distance(self.image, converted_image)
        
        # Convert distance to fitness (lower distance = higher fitness)
        fitness = 1 / (1 + distance)
        # Add a penalty for repeated colors in the palette
        unique_colors = np.unique(individual, axis=0)
        penalty = 1 - (len(unique_colors) / self.num_colors)
        
        return fitness - penalty

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