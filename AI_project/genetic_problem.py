# Genetic Algorithm Implementation
# This module contains a class-based implementation of a genetic algorithm 
# with fitness caching to avoid redundant calculations.

import random
import os
import numpy as np

class GeneticProblem:
    """
    Base class for genetic algorithm problems.
    Includes fitness caching to avoid redundant calculations.
    """
    
    def __init__(self):
        # Initialize fitness cache
        self.fitness_cache = {}
    
    def generate_individual(self):
        """Generate a single individual for the initial population."""
        pass
    
    def compute_fitness(self, individual):
        """Calculate the fitness of an individual."""
        pass
    
    def crossover(self, parent1, parent2):
        """Perform crossover between two parents to create offspring."""
        pass
    
    def mutate(self, individual, mutation_rate):
        """Mutate an individual."""
        pass
    
    def fitness(self, individual):
        """
        Get fitness with caching to avoid redundant calculations.
        Uses a hash of the individual as the cache key.
        """
        # Convert to a hashable representation
        hashable_individual = self._make_hashable(individual)
        
        # Check if we've already calculated this fitness
        if hashable_individual in self.fitness_cache:
            return self.fitness_cache[hashable_individual]
        
        # Calculate fitness
        fitness_value = self.compute_fitness(individual)
        
        # Store in cache
        self.fitness_cache[hashable_individual] = fitness_value
        
        return fitness_value
    
    def _make_hashable(self, individual):
        """Convert an individual to a hashable representation."""
        # For numpy arrays, convert to tuple of tuples
        if isinstance(individual, np.ndarray):
            return tuple(map(tuple, individual))
        # For lists, convert to tuple
        elif isinstance(individual, list):
            if isinstance(individual[0], list) or isinstance(individual[0], np.ndarray):
                return tuple(tuple(row) for row in individual)
            else:
                return tuple(individual)
        # Already hashable
        return individual
    
    def selection(self, population, fitness_scores, num_selections=1, method='roulette'):
        """
        Select individuals from the population based on their fitness.
        
        Args:
            population: List of individuals
            fitness_scores: List of fitness scores corresponding to population
            num_selections: Number of individuals to select
            method: Selection method ('roulette', 'tournament', 'rank')
            
        Returns:
            List of selected individuals
        """
        if method == 'roulette':
            total_fitness = sum(fitness_scores)
            if total_fitness == 0:
                # If all fitness scores are zero, select randomly
                return random.choices(population, k=num_selections)
            
            selection_probs = [f / total_fitness for f in fitness_scores]
            selected_indices = random.choices(range(len(population)), weights=selection_probs, k=num_selections)
            return [population[i] for i in selected_indices]
            
        elif method == 'tournament':
            tournament_size = min(3, len(population))
            selected = []
            
            for _ in range(num_selections):
                # Select random individuals for tournament
                tournament_indices = random.sample(range(len(population)), tournament_size)
                # Find the best individual in the tournament
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
                selected.append(population[winner_idx])
                
            return selected
            
        elif method == 'rank':
            # Sort population by fitness
            ranked_pairs = sorted(zip(population, fitness_scores), key=lambda x: x[1])
            ranked_population = [pair[0] for pair in ranked_pairs]
            # Select with probability proportional to rank
            ranks = list(range(1, len(population) + 1))
            selected_indices = random.choices(range(len(population)), weights=ranks, k=num_selections)
            return [ranked_population[i] for i in selected_indices]
        
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
    def initialize_population(self, size):
        """
        Generate an initial population of individuals.
        
        Args:
            size: Number of individuals in the population
            
        Returns:
            List of individuals
        """
        return [self.generate_individual() for _ in range(size)]
    
    def run(self, population_size, generations, mutation_rate=0.1, 
            crossover_rate=0.8, elitism=2, selection_method='roulette',
            save_results=False, results_dir="results"):
        """
        Run the genetic algorithm.
        
        Args:
            population_size: Size of the population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism: Number of best individuals to keep unchanged
            selection_method: Method for selection ('roulette', 'tournament', 'rank')
            save_results: Whether to save results
            results_dir: Directory to save results
            
        Returns:
            best_individual, best_fitness, fitness_history
        """
        # Create results directory if saving is enabled
        if save_results:
            os.makedirs(results_dir, exist_ok=True)
        
        # Initialize population
        population = self.initialize_population(population_size)
        
        # Track the best individual and fitness
        best_individual = None
        best_fitness = float('-inf')
        fitness_history = []
        cache_hits = 0
        cache_misses = 0
        
        for generation in range(generations):
            print(f"Generation {generation + 1}/{generations}")
            
            # Evaluate fitness - with caching
            fitness_scores = []
            for individual in population:
                hashable = self._make_hashable(individual)
                if hashable in self.fitness_cache:
                    fitness_scores.append(self.fitness_cache[hashable])
                    cache_hits += 1
                else:
                    fitness_value = self.compute_fitness(individual)
                    self.fitness_cache[hashable] = fitness_value
                    fitness_scores.append(fitness_value)
                    cache_misses += 1
            
            # Find best individual in this generation
            gen_best_idx = fitness_scores.index(max(fitness_scores))
            gen_best_individual = population[gen_best_idx]
            gen_best_fitness = fitness_scores[gen_best_idx]
            
            # Update overall best if needed
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_individual = gen_best_individual
                print(f"New best solution found! Fitness: {best_fitness:.6f}")
            
            # Save fitness history
            fitness_history.append(gen_best_fitness)
            print(f"Best fitness: {gen_best_fitness:.6f} (Cache hits: {cache_hits}, misses: {cache_misses})")
            
            # Optional: Save results for this generation
            if save_results:
                self.save_generation_results(gen_best_individual, generation, results_dir)
            
            # Create next generation
            if generation < generations - 1:  # Don't evolve after the last generation
                population = self.evolve_population(
                    population, 
                    fitness_scores, 
                    elitism=elitism, 
                    crossover_rate=crossover_rate, 
                    mutation_rate=mutation_rate,
                    selection_method=selection_method
                )
        
        # Save final results
        if save_results:
            self.save_best_results(best_individual, results_dir)
        
        # Cache statistics
        print(f"\nCache statistics: {cache_hits} hits, {cache_misses} misses")
        if cache_hits + cache_misses > 0:
            hit_rate = cache_hits / (cache_hits + cache_misses) * 100
            print(f"Cache hit rate: {hit_rate:.2f}%")
        
        return best_individual, best_fitness, fitness_history
    
    def evolve_population(self, population, fitness_scores, elitism=2, 
                        crossover_rate=0.8, mutation_rate=0.1, selection_method='roulette'):
        """
        Create a new generation through selection, crossover, and mutation.
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for the population
            elitism: Number of best individuals to keep unchanged
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            selection_method: Method for selection
            
        Returns:
            New population
        """
        population_size = len(population)
        
        # Sort by fitness (best first)
        sorted_indices = sorted(range(len(fitness_scores)), 
                              key=lambda i: fitness_scores[i], reverse=True)
        
        # Keep the best individuals (elitism)
        new_population = [population[i].copy() if hasattr(population[i], 'copy') 
                         else np.array(population[i]) for i in sorted_indices[:elitism]]
        
        # Fill the rest of the population
        while len(new_population) < population_size:
            # Select parents
            if random.random() < crossover_rate:
                # Crossover
                parents = self.selection(population, fitness_scores, 2, selection_method)
                child = self.crossover(parents[0], parents[1])
            else:
                # No crossover, just select an individual
                child = self.selection(population, fitness_scores, 1, selection_method)[0]
                # Make a copy to avoid modifying the original
                if hasattr(child, 'copy'):
                    child = child.copy()
                else:
                    child = np.array(child)
            
            # Mutate
            child = self.mutate(child, mutation_rate)
            
            # Add to new population
            new_population.append(child)
        
        # Ensure population size remains constant
        return new_population[:population_size]
    
    def save_generation_results(self, best_individual, generation, results_dir):
        """
        Save the best individual and its fitness for this generation.
        Override in subclasses if needed.
        Default implementation does nothing.
        """
        pass
    
    def save_best_results(self, best_individual, results_dir):
        """
        Save the best results. Override in subclasses if needed.
        Default implementation does nothing.
        """
        pass