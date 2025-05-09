# Genetic Algorithm Implementation
# This module contains a class-based implementation of a genetic algorithm 
# with fitness caching to avoid redundant calculations.

import random
import os
import numpy as np
import time
import datetime

class GeneticProblem:
    """
    Base class for genetic algorithm problems.
    Includes fitness caching to avoid redundant calculations.
    """
    
    def __init__(self, results_dir=None, save_results=False):
        # Initialize fitness cache
        if save_results:
            timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            self.results_dir = f"{results_dir}_{timestamp}"
            os.makedirs(self.results_dir, exist_ok=True)
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
            save_results=False, adaptation_rate=1, 
            adaptation_threshold=10, halting_stagnation_threshold=None):
        """
        Run the genetic algorithm.

        Args:
            population_size: Number of individuals in the population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism: Number of best individuals to keep unchanged
            selection_method: Method for selection ('roulette', 'tournament', 'rank')
            save_results: Whether to save results
            results_dir: Directory to save results
            adaptation_rate: Rate of mutation adaptation (0 for no adaptation)
            adaptation_threshold: Number of generations to wait before adapting mutation rate
            halting_stagnation_threshold: Number of generations to wait before halting if no improvement is detected
    
        Returns:
            best_individual: The best individual found
            best_fitness: The fitness of the best individual
            fitness_history: List of fitness scores for each generation
            bestResult: The best result from the best individual (if applicable)
        """
        # Save parameters
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.selection_method = selection_method
        self.adaptation_rate = adaptation_rate
        self.adaptation_threshold = adaptation_threshold
        self.halting_stagnation_threshold = halting_stagnation_threshold

        # Start timer
        start_time = time.time()
        
        # Initialize population
        population = self.initialize_population(population_size)
        
        # Track the best individual and fitness
        best_individual = None
        best_fitness = float('-inf')
        fitness_history = []
        average_fitness_history = []
        cache_hits = 0
        cache_misses = 0
        performance_log = []

        stagnation_count = 0
        current_mutation_rate = mutation_rate

        for generation in range(generations):
            gen_start_time = time.time()
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
            
            #Calculate stats
            average_fitness = sum(fitness_scores) / len(fitness_scores)
            max_fitness = max(fitness_scores)
            min_fitness = min(fitness_scores)

            # Find best individual in this generation
            gen_best_idx = fitness_scores.index(max(fitness_scores))
            gen_best_individual = population[gen_best_idx]
            gen_best_fitness = fitness_scores[gen_best_idx]
            
            # Update overall best if needed
            improved = False
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_individual = gen_best_individual
                improved = True
                stagnation_count = 0
                print(f"New best solution found! Fitness: {best_fitness:.6f}")
            else:
                stagnation_count += 1
            
            # Adapt mutation rate if stagnation occurs
            mutation_adapted = False
            if adaptation_rate > 1:
                if stagnation_count >= adaptation_threshold:
                    previous_mutation_rate = current_mutation_rate
                    current_mutation_rate = min(0.5, current_mutation_rate * adaptation_rate)
                    mutation_adapted = True
                    print(f"Stagnation detected ({stagnation_count} generations). "
                      f"Adjusting mutation rate from {previous_mutation_rate:.4f} to {current_mutation_rate:.4f}")
                elif improved and current_mutation_rate != mutation_rate:
                    current_mutation_rate = mutation_rate
                    mutation_adapted = True
                    print(f"Improvement detected. Reducing mutation rate to {current_mutation_rate:.4f}")


            # Save fitness history
            fitness_history.append(gen_best_fitness)
            average_fitness_history.append(average_fitness)
        
            print(f"Best fitness: {gen_best_fitness:.6f} (Cache hits: {cache_hits}, misses: {cache_misses})")
            
            # Calculate generation time
            gen_time = time.time() - gen_start_time

            # Log performance metrics
            performance_log.append({
                'generation': generation + 1,
                'best_fitness': max_fitness,
                'avg_fitness': average_fitness,
                'min_fitness': min_fitness,
                'time_seconds': gen_time,
                'cache_hits': cache_hits,
                'cache_misses': cache_misses,
                'improved': improved,
                'stagnation_count': stagnation_count,
                'current_mutation_rate': current_mutation_rate,
                'mutation_adapted': mutation_adapted
            })
            
            # Optional: Save results for this generation only if it improved
            if save_results and improved:
                self.save_generation_results(gen_best_individual, generation)
            
            # Create next generation
            if generation < generations - 1 and (halting_stagnation_threshold == None or halting_stagnation_threshold > stagnation_count):
                population = self.evolve_population(
                    population, 
                    fitness_scores, 
                    elitism=elitism, 
                    crossover_rate=crossover_rate, 
                    mutation_rate=current_mutation_rate,
                    selection_method=selection_method
                )
            else:
                print("Halting evolution due to stagnation.")
                break

        # Calculate total execution time
        total_time = time.time() - start_time
            
        # Save final results
        if save_results:
            self.save_best_results(best_individual)
            self.save_performance_report(performance_log, best_individual, total_time, population_size)
            
        # Cache statistics
        print(f"\nCache statistics: {cache_hits} hits, {cache_misses} misses")
        if cache_hits + cache_misses > 0:
            hit_rate = cache_hits / (cache_hits + cache_misses) * 100
            print(f"Cache hit rate: {hit_rate:.2f}%")

        bestResult = None
        bestResult = self.getBestResult(best_individual)

        return best_individual, best_fitness, fitness_history, bestResult
        
    def save_performance_report(self, performance_log, best_individual, total_time, population_size):
        """
        Save a detailed performance report of the genetic algorithm run.
        
        Args:
            performance_log: List of dictionaries with performance metrics for each generation
            best_individual: The overall best individual found
            total_time: Total execution time in seconds
            results_dir: Directory to save the report
        """
        pass

    def getBestResult(self, best_individual):
        """
        Get the best result from the best individual.
        Override in subclasses if needed.
        Default implementation returns None.
        """
        return None
    
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
            # if we have the attribute mutate_diverse then use the self.mutate_diverse method
            if hasattr(self, 'use_mutate_diverse'):
                child = self.mutate_diverse(child, mutation_rate)
            else:
                child = self.mutate(child, mutation_rate)
            
            # Add to new population
            new_population.append(child)
        
        # Ensure population size remains constant
        return new_population[:population_size]
    
    def save_generation_results(self, best_individual, generation):
        """
        Save the best individual and its fitness for this generation.
        Override in subclasses if needed.
        Default implementation does nothing.
        """
        pass
    
    def save_best_results(self, best_individual):
        """
        Save the best results. Override in subclasses if needed.
        Default implementation does nothing.
        """
        pass