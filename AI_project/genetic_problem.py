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
    Args:
        results_dir: Directory to save results
        save_results: Whether to save results
        use_caching: Whether to use caching for fitness calculations
    """
    
    def __init__(self, results_dir=None, save_results=True, use_caching=True):
        
        if save_results:
            if results_dir is None:
                timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                self.results_dir = f"{results_dir}_{timestamp}"
            else:
                self.results_dir = results_dir
            os.makedirs(self.results_dir, exist_ok=True)
        
        self.fitness_cache = {}
        self.use_caching = use_caching
        self.save_results = save_results
    
    def generate_individual(self):
        """
        Generate a single individual for the initial population.
        
        Args:
            None

        Returns:
            A randomly generated individual
        """
        pass
    
    def compute_fitness(self, individual):
        """
        Calculate the fitness of an individual.
        
        Args:
            individual: The individual to evaluate
        
        Returns:
                Fitness score (higher is better)
        """
        pass
    
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to create offspring.
        
        Args:
            parent1: First parent
            parent2: Second parent
        
        Returns:
            Offspring created from the parents
        """
        pass
    
    def mutate(self, individual):
        """
        Mutate an individual.
        
        Args:
            individual: The individual to mutate
        Returns:
            Mutated individual according to the mutation rate
        """
        pass
    
    def fitness(self, individual):
        """
        Get fitness with caching to avoid redundant calculations.
        Uses a hash of the individual as the cache key.

        Args:
            individual: The individual to evaluate
        Returns:
            Fitness score (higher is better)
        """
        if not self.use_caching:
            return self.compute_fitness(individual)
        
        # Check if the fitness for the individual is already cached
        hashable_individual = self._make_hashable(individual)
        
        if hashable_individual in self.fitness_cache:
            return self.fitness_cache[hashable_individual]
        
        # If not cached, compute fitness
        else:
            fitness_value = self.compute_fitness(individual)

            # Store the computed fitness in the cache for future use
            self.fitness_cache[hashable_individual] = fitness_value
            
            return fitness_value
    
    def _make_hashable(self, individual):
        """
        Convert an individual to a hashable type for caching.
        
        Args:
            individual: The individual to convert
        
        Returns:
            A hashable representation of the individual
        """
        
        # Convert numpy to tuple of tuples
        if isinstance(individual, np.ndarray):
            return tuple(map(tuple, individual))

        return individual
    
    def selection(self, population, fitness_scores, num_selections=1):
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

        if self.selection_method == 'roulette':
            # We may have negative fitness scores, so we need to shift them
            min_fitness = min(fitness_scores)
            
            shifted_fitness = fitness_scores
            if min_fitness < 0:
                shifted_fitness = [f - min_fitness + 1 for f in fitness_scores]
            
            # Select individuals based on shifted fitness
            total_fitness = sum(shifted_fitness)
            if total_fitness == 0:
                # If all fitness scores are zero, select randomly
                return random.choices(population, k=num_selections)
            
            selection_probs = [f / total_fitness for f in shifted_fitness]
            selected_indices = random.choices(range(len(population)), weights=selection_probs, k=num_selections)
            return [population[i] for i in selected_indices]
            
        elif self.selection_method == 'tournament':
            tournament_size = min(self.tournament_size, len(population))
            selected = []
            
            for _ in range(num_selections):
                tournament_indices = random.sample(range(len(population)), tournament_size)

                fitnesses_in_tournament = [fitness_scores[i] for i in tournament_indices]
                max_fitness_in_tournament = max(fitnesses_in_tournament)
                winner_index = tournament_indices[fitnesses_in_tournament.index(max_fitness_in_tournament)]
                selected.append(population[winner_index])
                
            return selected
            
        elif self.selection_method == 'rank':
            # Sort population by fitness
            ranked_pairs = sorted(zip(population, fitness_scores), key=lambda x: x[1])
            ranked_population = [pair[0] for pair in ranked_pairs]

            # Select with probability proportional to rank
            ranks = list(range(1, len(population) + 1))
            selected_indices = random.choices(range(len(population)), weights=ranks, k=num_selections)
            return [ranked_population[i] for i in selected_indices]
        
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
        
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
            tournament_size=3, adaptation_rate=1, adaptation_threshold=10, 
            halting_stagnation_threshold=None):
        """
        Run the genetic algorithm.

        Args:
            population_size: Number of individuals in the population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism: Number of best individuals to keep unchanged between generations
            selection_method: Method for selection ('roulette', 'tournament', 'rank')
            tournament_size: Size of the tournament for selection (only used if selection_method is 'tournament')
            adaptation_rate: Rate of mutation adaptation (1 for no adaptation)
            adaptation_threshold: Number of generations to wait before adapting mutation rate
            halting_stagnation_threshold: Number of generations to wait before halting if no improvement is detected. If using adaptation, it is recommended to set it over the adaptation_threshold
    
        Returns:
            best_individual: The best individual found
            best_fitness: The fitness of the best individual
            fitness_history: List of best fitness score for each generation
            average_fitness_history: List of average fitness score for each generation
            bestResult: The best result from the best individual (if applicable)
            statistics: Dictionary with statistics of the run, including best fitness, average fitness, and diversity metrics
        """

        # Save parameters for later use
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.adaptation_rate = adaptation_rate
        self.adaptation_threshold = adaptation_threshold
        self.halting_stagnation_threshold = halting_stagnation_threshold

        # Start tracking
        start_time = time.time()
        
        best_individual = None
        best_fitness = float('-inf')
        fitness_history = []
        average_fitness_history = []
        cache_hits = 0
        cache_misses = 0
        performance_log = []

        stagnation_count = 0
        current_mutation_rate = mutation_rate


        population = self.initialize_population(population_size)
        

        for generation in range(generations):
            gen_start_time = time.time()
            print(f"Generation {generation + 1}/{generations}")
            
            # Evaluate fitness of the population
            fitness_scores = []
            for individual in population:
                
                if self.use_caching:    
                    # Track cache statistics
                    hashable = self._make_hashable(individual)
                    if hashable in self.fitness_cache:
                        cache_hits += 1
                    else:
                        cache_misses += 1
                
                fitness_value = self.fitness(individual)
                fitness_scores.append(fitness_value)
                
            
            #Calculate stats
            average_fitness = sum(fitness_scores) / len(fitness_scores)
            max_fitness = max(fitness_scores)
            min_fitness = min(fitness_scores)

            # Find best individual in this generation
            gen_best_index = fitness_scores.index(max(fitness_scores))
            gen_best_individual = population[gen_best_index]
            gen_best_fitness = fitness_scores[gen_best_index]
            
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
        
            if self.use_caching:
                print(f"Best fitness: {gen_best_fitness:.6f} (Cache hits: {cache_hits}, misses: {cache_misses})")
            else:
                print(f"Best fitness: {gen_best_fitness:.6f}")
            
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
            
            # Save results for this generation only if it improved
            if self.save_results and improved:
                self.save_generation_results(gen_best_individual, generation)
            
            # Create next generation
            if generation < generations - 1:
                if halting_stagnation_threshold == None or halting_stagnation_threshold > stagnation_count:
                    population = self.evolve_population(population, fitness_scores)
                else: 
                    print("Halting evolution due to stagnation.")
                    break

        # Calculate total execution time
        total_time = time.time() - start_time
            
        # Save final results
        if self.save_results:
            self.save_best_results(best_individual)
            self.save_performance_report(performance_log, best_individual, total_time)
            
        # Cache statistics
        if self.use_caching:
            print(f"\nCache statistics: {cache_hits} hits, {cache_misses} misses")
            if cache_hits + cache_misses > 0:
                hit_rate = cache_hits / (cache_hits + cache_misses) * 100
                print(f"Cache hit rate: {hit_rate:.2f}%")
        else:
            print("\nCaching was disabled for this run.")

        bestResult = self.getBestResult(best_individual)

        return best_individual, best_fitness, fitness_history, average_fitness_history, bestResult, performance_log
        
    def save_performance_report(self, performance_log, best_individual, total_time):
        """
        Save a detailed performance report of the genetic algorithm run.
        
        Args:
            performance_log: List of dictionaries with performance metrics for each generation
            best_individual: The overall best individual found
            total_time: Total execution time in seconds
        
        Returns:
            Nothing. A report is saved to the results directory.
        """
        pass

    def getBestResult(self, best_individual):
        """
        Get the interpretation of the best individual.
        Default implementation returns None.

        Args:
            best_individual: The best individual found

        Returns:
            The result obtained by interpreting the best individual
        """

        return None
    
    def evolve_population(self, population, fitness_scores):
        """
        Create a new generation through selection, crossover, and mutation.
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for the population
            
        Returns:
            New population
        """
        population_size = len(population)
        
        # Sort by fitness (best first)
        sorted_indices = sorted(range(len(fitness_scores)), 
                              key=lambda i: fitness_scores[i], reverse=True)
        
        # Keep the best individuals (elitism)
        new_population = [np.array(population[i]) for i in sorted_indices[:self.elitism]]
        

        while len(new_population) < population_size:
            
            # Crossover
            if random.random() < self.crossover_rate:
                parents = self.selection(population, fitness_scores, 2)
                child = self.crossover(parents[0], parents[1])
            else:
                # No crossover, just select an individual
                child = self.selection(population, fitness_scores, 1)[0]
                child = np.array(child)
            
            # Mutate
            child = self.mutate(child)
            
            # Add to new population
            new_population.append(child)
        
        # Make sure the population size is constant
        return new_population[:population_size]
    
    def save_generation_results(self, best_individual, generation):
        """
        Save the best individual and its fitness for this generation.

        Args:
            best_individual: The best individual found in this generation
            generation: The current generation number
        
        Returns:
            Nothing. Generation results are saved to the results directory.
        """
        pass
    
    def save_best_results(self, best_individual):
        """
        Save the best results.

        Args:
            best_individual: The best individual found in the entire run
        Returns:
            Nothing. Best results are saved to the results directory.
        """
        pass