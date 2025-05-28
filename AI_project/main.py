import os
from datetime import datetime
import argparse
from run_algorithm import run_image_quantization
import matplotlib.pyplot as plt
from helper import (save_image, load_image, save_image_with_palette)
import statistics

CONFIGURATIONS = [
    # Add new configuration here
    {
        'name': 'basic',                        # To identify the configuration
        
        'restricted': False,                    # Whether to restrict color selection to colors in the original image  
        
        'population_size': 10,                  # Size of the population for the genetic algorithm
        'generations': 30,                      # Number of generations to run the algorithm
        'mutation_rate': 0.2,                   # Probability of mutation for each individual
        'crossover_rate': 0.8,                  # Probability of crossover between individuals
        'elitism': 2,                           # Number of best individuals to carry over to the next generation   
        'halting_stagnation_threshold': 20,     # Threshold for halting if no improvement is seen for this many generations

        'adaptation_rate': 1.1,                 # Rate at which mutation rate is adapted
        'adaptation_threshold': 10,             # Threshold for adaptation to be applied (if None, adaptation is disabled)

        'kMeans': True,                        # Whether to use KMeans clustering for initial population generation    

        'selection_method': 'tournament',       # Method for selecting individuals for reproduction
        'tournament_size': 3,                   # Size of the tournament for selection methods that use tournaments
        'mutate_diverse': False,                # Whether to use diverse mutation strategy (if True, the custom mutation operator is used)
        'crossover_method': 'one_point',        # Method for crossover between individuals ('one_point', 'uniform', 'closest_pairs' - custom crossover)
        
        'use_caching': False,                   # Whether to use caching for fitness evaluations (can speed up the algorithm significantly)
        'display': True                         # Whether to results after each run
    }
]

# Number of repetitions for each configuration to get statistical significance
REPETITIONS = 3
# Default color counts to test
BASIC_COLOR_CHANGES = [4, 10, 20]
# Default image to test
DEFAULT_IMAGE = 'cat.jpg'

def run_multiple_images_test(
    # Test configuration
    test_type,                              # Type of test: 'caching', 'restriction_kmeans', 'selection', 'rgb', 'mutation', 'crossover', 'adaptation', 'early_stopping', 'hyperparameter_tuning', 'new'
    image_names = [DEFAULT_IMAGE],          # List of image names to test
    color_counts = BASIC_COLOR_CHANGES,     # List of color counts to test
    repetitions = REPETITIONS,              # Number of repetitions per configuration
    configurations = None,                  # Override default configurations for the test
    output_dir = None,                      # Custom output directory
    verbose = True                          # Print progress information
):
    """
    Run a specific configuration test for the genetic algorithm on multiple images.

    Args:
        test_type (str): Type of test to run ('caching', 'restriction_kmeans', 'selection', 'rgb', 'mutation', 'crossover', 'adaptation', 'early_stopping', 'hyperparameter_tuning', 'new')
        image_names (list): List of image names to test
        color_counts (list): List of color counts to test
        repetitions (int): Number of repetitions for each configuration
        configurations (list): Custom configurations to use for the test, if None, default configurations are used
        output_dir (str): Directory to save results, if None, a timestamped directory will be created
        verbose (bool): Whether to print progress information

    Returns:
        dict: Dictionary containing the output directory
    """

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    if output_dir is None:
        output_dir = os.path.join("analysis", f"{test_type}_comparison_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"Running {test_type} comparison test with {repetitions} repetitions for each configuration")
        print(f"Testing color counts: {color_counts}")
        print(f"Testing images: {image_names}")
        print(f"Results will be saved to: {output_dir}")

    for image_name in image_names:
        image_output_dir = os.path.join(output_dir, image_name.replace('.', '_'))
        os.makedirs(image_output_dir, exist_ok=True)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"TESTING IMAGE: {image_name}")
            print(f"{'='*60}")
        
        # Run the configuration test for this image
        run_configuration_test(
            test_type=test_type,
            image_name=image_name,
            color_counts=color_counts,
            repetitions=repetitions,
            configurations=configurations,
            output_dir=image_output_dir,
            verbose=verbose
        )
    
        if verbose:
            print(f"Completed tests for image: {image_name}")

    if verbose:
        print(f"\nComparison complete. Results saved to {output_dir}")

    return {
        'output_dir': output_dir
    }


def run_configuration_test(
    # Test configuration
    test_type,                              # Type of test: 'caching', 'restriction_kmeans', 'selection', 'rgb', 'mutation', 'crossover', 'adaptation', 'early_stopping', 'hyperparameter_tuning', 'new'
    image_name = DEFAULT_IMAGE,             # Name of image to test
    color_counts = BASIC_COLOR_CHANGES,     # List of color counts to test
    repetitions = REPETITIONS,              # Number of repetitions per configuration
    configurations = None,                  # Override default configurations for the test
    output_dir = None,                      # Custom output directory
    verbose = True                          # Print progress information
):
    """
    Run a specific configuration test for the genetic algorithm on a single image.

    Args:
        test_type (str): Type of test to run ('caching', 'restriction_kmeans', 'selection', 'rgb', 'mutation', 'crossover', 'adaptation', 'early_stopping', 'hyperparameter_tuning', 'new')
        image_name (str): Name of the image to test
        color_counts (list): List of color counts to test
        repetitions (int): Number of repetitions for each configuration
        configurations (list): Custom configurations to use for the test, if None, default configurations are used
        output_dir (str): Directory to save results, if None, a timestamped directory will be created
        verbose (bool): Whether to print progress information

    Returns:
        dict: Dictionary containing the output directory
    """        
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    if output_dir is None:
        output_dir = f"analysis/{test_type}_comparison_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"Running {test_type} comparison test with {repetitions} repetitions for each configuration")
        print(f"Testing color counts: {color_counts}")
        print(f"Testing image: {image_name}")
        print(f"Results will be saved to: {output_dir}")
        
    # Generate test configurations based on test_type
    if configurations is None:
        configurations = generate_test_configurations(test_type)

    # Results structure: {color_count: {config_name: results}}
    all_results = {}
    best_images = {}
    
    # Execute the tests
    if test_type == 'rgb':
        run_rgb_comparison(
            configurations, 
            output_dir
        )

    else:
        for num_colors in color_counts:
            if verbose:
                print(f"\n{'='*60}")
                print(f"TESTING WITH {num_colors} COLORS")
                print(f"{'='*60}")
            
            color_dir = os.path.join(output_dir, f"colors_{num_colors}")
            os.makedirs(color_dir, exist_ok=True)
            
            all_results[num_colors] = {}
            best_images[num_colors] = {}
            
            for config in configurations:
                # Create a copy of the configuration and update color count
                test_config = dict(config)
                test_config['num_colors'] = num_colors
                
                config_key = test_config.get('name', '').replace(' ', '_').lower()
                
                if verbose:
                    print(f"\n--- Testing {config_key} with {num_colors} colors ---")
                
                results_store = all_results[num_colors]
                best_store = best_images[num_colors]
                    
                if config_key not in results_store:
                    results_store[config_key] = {
                        'best_fitness': [],          # Best fitness from each run
                        'execution_time': [],        # Total execution time per run 
                        'fitness_history': [],       # Fitness history arrays from each run
                        'avg_population_fitness_history': [], # Average population fitness history from each run
                        'cache_hits': [],            # Cache hits from each run
                        'cache_misses': [],          # Cache misses from each run
                        'cache_hit_ratio': [],       # Cache hit ratio from each run
                        'stagnation_detected': [],   # Whether stagnation was detected in each run
                        'mutation_rate_history': []  # History of mutation rates from each run
                    }
                    
                # Track best result
                best_fitness = -float('inf')
                best_palette = None
                best_image = None
                
                # Run multiple repetitions of this configuration
                for i in range(1, repetitions + 1):
                    if verbose:
                        print(f"Repetition {i}/{repetitions}")
                    
                    start_time = datetime.now()
                    run_output_dir = os.path.join(color_dir, f"{config_key}_rep{i}")
                    os.makedirs(run_output_dir, exist_ok=True)
                    

                    palette, fitness, history, average_history, result_img, _, _, performance_log = run_image_quantization(
                        image_name=image_name,
                        output_dir=run_output_dir,
                        save_results=True,        
                        **{k: v for k, v in test_config.items() if k != 'name'}
                    )
                    
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    # Extract data from performance log
                    last_gen = performance_log[-1]
                    
                    results_store[config_key]['best_fitness'].append(fitness)
                    results_store[config_key]['execution_time'].append(execution_time)
                    results_store[config_key]['fitness_history'].append(history)
                    results_store[config_key]['avg_population_fitness_history'].append(average_history)
                        
                    cache_hits = last_gen['cache_hits']
                    cache_misses = last_gen['cache_misses']
                    cache_hit_ratio = 0
                    if cache_hits + cache_misses > 0:
                        cache_hit_ratio = cache_hits / (cache_hits + cache_misses)
                                
                    results_store[config_key]['cache_hits'].append(cache_hits)
                    results_store[config_key]['cache_misses'].append(cache_misses)
                    results_store[config_key]['cache_hit_ratio'].append(cache_hit_ratio)
                        
                    stagnation_occurred = [log['stagnation_count'] > 0 for log in performance_log]
                    results_store[config_key]['stagnation_detected'].append(stagnation_occurred)
                                            
                    # Build mutation rate history
                    mutation_rate_history = [log['current_mutation_rate'] for log in performance_log]
                    results_store[config_key]['mutation_rate_history'].append(mutation_rate_history)
                    
                    # Check if this is the best result so far
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_palette = palette
                        best_image = result_img
                    
                    if verbose:
                        print(f"Fitness: {fitness:.6f}, Time: {execution_time:.2f}s")
                
                # Save the best image from repetitions
                if best_image is not None:
                    img_path = os.path.join(color_dir, f"{config_key}.png")
                    save_image_with_palette(best_image, best_palette, img_path, best_fitness)
                    
                    # Store best image information for reporting
                    best_store[config_key] = {
                        'fitness': best_fitness,
                        'palette': best_palette,
                        'image_path': img_path
                    }
    
        # Process results to calculate statistics
        stats = process_test_results(all_results)
        
        # Generate visualizations and reports
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        generate_test_results_visualizations(stats, test_type, color_counts, vis_dir)
        generate_test_report(stats, best_images, test_type, color_counts, image_name, repetitions, output_dir)
    
    if verbose:
        print(f"\nComparison complete. Results saved to {output_dir}")
    
    # Return results dict with stats and paths
    return {
        'output_dir': output_dir
    }


def generate_test_configurations(test_type):
    """
    Generate appropriate test configurations based on test type
    
    Args:
        test_type (str): Type of test ('caching', 'restriction_kmeans', 'selection', 'rgb', 'mutation', 'crossover', 'adaptation', 'early_stopping', 'hyperparameter_tuning', 'new')
        
    Returns:
        list: List of configuration dictionaries appropriate for the test type
    """
    base_config = {
        'restricted': True,
        'population_size': 10,
        'generations': 30,
        'mutation_rate': 0.2,
        'crossover_rate': 0.8,
        'selection_method': 'tournament',
        'tournament_size': 3,
        'elitism': 1,
        'adaptation_rate': 1,
        'adaptation_threshold': None,
        'halting_stagnation_threshold': None,
        'kMeans': False,
        'mutate_diverse': False,
        'crossover_method': 'closest_pairs',
        'use_caching': True,
        'display': False
    }
    
    if test_type == 'caching':
        caching_enabled = dict(base_config)
        caching_enabled['name'] = 'caching_enabled'
        caching_enabled['use_caching'] = True
        caching_enabled['generations'] = 50
        
        caching_disabled = dict(base_config)
        caching_disabled['name'] = 'caching_disabled'
        caching_disabled['use_caching'] = False
        caching_disabled['generations'] = 50
        
        return [caching_enabled, caching_disabled]
    
    elif test_type == 'restriction_kmeans':
        configs = []
        
        # Unrestricted without KMeans
        unrestricted = dict(base_config)
        unrestricted['name'] = 'unrestricted'
        unrestricted['restricted'] = False
        unrestricted['kMeans'] = False
        configs.append(unrestricted)
        
        # Unrestricted with KMeans
        unrestricted_kmeans = dict(base_config)
        unrestricted_kmeans['name'] = 'unrestricted_kmeans'
        unrestricted_kmeans['restricted'] = False
        unrestricted_kmeans['kMeans'] = True
        configs.append(unrestricted_kmeans)
        
        # Restricted without KMeans
        restricted = dict(base_config)
        restricted['name'] = 'restricted'
        restricted['restricted'] = True
        restricted['kMeans'] = False
        configs.append(restricted)
        
        # Restricted with KMeans
        restricted_kmeans = dict(base_config)
        restricted_kmeans['name'] = 'restricted_kmeans'
        restricted_kmeans['restricted'] = True
        restricted_kmeans['kMeans'] = True
        configs.append(restricted_kmeans)
        
        return configs
    
    elif test_type == 'selection':
        selection_methods = ['roulette', 'rank']
        configs = []
        
        for method in selection_methods:
            config = dict(base_config)
            config['selection_method'] = method
            config['name'] = f"{method}_selection"
            configs.append(config)
        
        tournament_sizes = [3, 5, 10]
        for size in tournament_sizes:
            config = dict(base_config)
            config['selection_method'] = 'tournament'
            config['tournament_size'] = size
            config['name'] = f"tournament_size_{size}"
            configs.append(config)
        
        return configs
    
    elif test_type == 'mutation':
        configs = []
        
        # Standard mutation with different rates
        standard_low = dict(base_config)
        standard_low['name'] = 'standard_low'
        standard_low['mutation_rate'] = 0.2
        standard_low['mutate_diverse'] = False
        configs.append(standard_low)
        
        standard_high = dict(base_config)
        standard_high['name'] = 'standard_high'
        standard_high['mutation_rate'] = 0.6
        standard_high['mutate_diverse'] = False
        configs.append(standard_high)
        
        # Diverse mutation with different rates
        diverse_low = dict(base_config)
        diverse_low['name'] = 'diverse_low'
        diverse_low['mutation_rate'] = 0.2
        diverse_low['mutate_diverse'] = True
        configs.append(diverse_low)
        
        diverse_high = dict(base_config)
        diverse_high['name'] = 'diverse_high'
        diverse_high['mutation_rate'] = 0.6
        diverse_high['mutate_diverse'] = True
        configs.append(diverse_high)
        
        return configs
    
    elif test_type == 'crossover':
        configs = []
        
        # One-point crossover
        one_point = dict(base_config)
        one_point['name'] = 'one_point_crossover'
        one_point['crossover_method'] = 'one_point'
        configs.append(one_point)
        
        # Uniform crossover
        uniform = dict(base_config)
        uniform['name'] = 'uniform_crossover'
        uniform['crossover_method'] = 'uniform'
        configs.append(uniform)

        # Closest pairs crossover
        closest_pairs = dict(base_config)
        closest_pairs['name'] = 'closest_pairs_crossover'
        closest_pairs['crossover_method'] = 'closest_pairs'
        configs.append(closest_pairs)
        
        return configs
    
    elif test_type == 'adaptation':
        configs = []
        
        # No adaptation
        no_adaptation = dict(base_config)
        no_adaptation['name'] = 'no_adaptation'
        no_adaptation['adaptation_rate'] = 1.0
        no_adaptation['adaptation_threshold'] = None
        no_adaptation['generations'] = 50
        configs.append(no_adaptation)
        
        # Low adaptation rate
        low_adaptation = dict(base_config)
        low_adaptation['name'] = 'low_adaptation'
        low_adaptation['adaptation_rate'] = 1.05
        low_adaptation['adaptation_threshold'] = 5
        low_adaptation['generations'] = 50
        configs.append(low_adaptation)
        
        # High adaptation rate
        high_adaptation = dict(base_config)
        high_adaptation['name'] = 'high_adaptation'
        high_adaptation['adaptation_rate'] = 1.2
        high_adaptation['adaptation_threshold'] = 10
        high_adaptation['generations'] = 50
        configs.append(high_adaptation)
        
        return configs
    
    elif test_type == 'early_stopping':
        configs = []
        
        # No early stopping
        no_early_stopping = dict(base_config)
        no_early_stopping['name'] = 'no_early_stopping'
        no_early_stopping['halting_stagnation_threshold'] = None
        no_early_stopping['generations'] = 50
        configs.append(no_early_stopping)
        
        # Early stopping with low threshold
        early_stopping_10 = dict(base_config)
        early_stopping_10['name'] = 'early_stopping_10'
        no_early_stopping['generations'] = 50
        early_stopping_10['halting_stagnation_threshold'] = 10
        configs.append(early_stopping_10)
        
        # Early stopping with high threshold
        early_stopping_20 = dict(base_config)
        early_stopping_20['name'] = 'early_stopping_20'
        early_stopping_20['generations'] = 50
        early_stopping_20['halting_stagnation_threshold'] = 20
        configs.append(early_stopping_20)
        
        return configs
    
    elif test_type == 'hyperparameter_tuning':
        configs = []
        
        # Base configuration with moderate values
        base_tuning_config = dict(base_config)
        base_tuning_config['population_size'] = 10
        base_tuning_config['generations'] = 30
        base_tuning_config['mutation_rate'] = 0.2
        base_tuning_config['crossover_rate'] = 0.8
        base_tuning_config['elitism'] = 2
        
        # Varying population sizes (keeping other params fixed)
        for pop_size in [5, 10, 20]:
            config = dict(base_tuning_config)
            config['name'] = f'population_size_{pop_size}'
            config['population_size'] = pop_size
            configs.append(config)
        
        # Varying generations (keeping other params fixed)
        for gen in [15, 30, 50]:
            config = dict(base_tuning_config)
            config['name'] = f'gen_{gen}'
            config['generations'] = gen
            configs.append(config)
        
        # Varying crossover rates (keeping other params fixed)
        for cr_rate in [0.7, 0.85]:
            config = dict(base_tuning_config)
            config['name'] = f'crossoverRate_{int(cr_rate*100)}'
            config['crossover_rate'] = cr_rate
            configs.append(config)
        
        # Varying elitism (keeping other params fixed)
        for elite in [1, 4]:
            config = dict(base_tuning_config)
            config['name'] = f'elite_{elite}'
            config['elitism'] = elite
            configs.append(config)
        
        return configs
    
    elif test_type == 'rgb':
        rgb_config = dict(base_config)
        rgb_config['name'] = 'rgb-distance'
        rgb_config['num_colors'] = 4
        rgb_config['rgb_distance'] = True
        rgb_config['kMeans'] = True
        rgb_config['generations'] = 50
        rgb_config['population_size'] = 20
        
        perceptual_config = dict(base_config)
        perceptual_config['name'] = 'perceptual-distance'
        perceptual_config['num_colors'] = 4
        perceptual_config['rgb_distance'] = False
        perceptual_config['kMeans'] = True
        perceptual_config['generations'] = 50
        perceptual_config['population_size'] = 20
        
        return [rgb_config, perceptual_config]
    
    elif test_type == 'new':
        return CONFIGURATIONS
    
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
def process_test_results(all_results):
    """
    Process raw test results to calculate statistics (averages, std dev, etc.)
    
    Args:
        all_results (dict): Raw test results in the structure:
                            {color_count: {config_name: {metric: [values]}}}
        
    Returns:
        dict: Processed statistics in the same structure but with calculated metrics
    """
    processed_stats = {}
    
    for color_count, configs in all_results.items():
        processed_stats[color_count] = {}
        
        for config_name, results in configs.items():
            # Calculate basic statistics
            fitness_values = results['best_fitness']
            time_values = results['execution_time']
            
            processed_stats[color_count][config_name] = {
                'avg_best_fitness': sum(fitness_values) / len(fitness_values),
                'max_best_fitness': max(fitness_values),
                'min_best_fitness': min(fitness_values),
                'std_best_fitness': statistics.stdev(fitness_values) if len(fitness_values) > 1 else 0,
                'avg_time': sum(time_values) / len(time_values),
                'max_time': max(time_values),
                'min_time': min(time_values),
                'std_time': statistics.stdev(time_values) if len(time_values) > 1 else 0
            }

            # Process cache statistics if available
            if 'cache_hits' in results and results['cache_hits']:
                cache_hits = results['cache_hits']
                cache_misses = results['cache_misses']
                hit_ratios = results['cache_hit_ratio']
                
                processed_stats[color_count][config_name]['cache_stats'] = {
                    'avg_cache_hits': sum(cache_hits) / len(cache_hits),
                    'avg_cache_misses': sum(cache_misses) / len(cache_misses),
                    'avg_hit_ratio': sum(hit_ratios) / len(hit_ratios),
                    'total_cache_hits': sum(cache_hits),
                    'total_cache_misses': sum(cache_misses),
                    'overall_hit_ratio': sum(cache_hits) / (sum(cache_hits) + sum(cache_misses)) 
                        if (sum(cache_hits) + sum(cache_misses)) > 0 else 0
                }
            
            # Process stagnation detection
            stagnation_detected = results['stagnation_detected']
            stagnation_count = sum(sum(1 for value in run if value) for run in stagnation_detected if run)
                
            processed_stats[color_count][config_name]['stagnation_stats'] = {
                'stagnation_frequency': stagnation_count / sum(len(run) for run in stagnation_detected),
                'generations_with_stagnation': stagnation_count,
                'total_generations': sum(len(run) for run in stagnation_detected)
            }
            
            # Process mutation rate adaptation if available
            if 'mutation_rate_history' in results and results['mutation_rate_history']:
                mutation_histories = results['mutation_rate_history']
                
                # Find average max and min mutation rates across runs
                max_rates = []
                min_rates = []
                rate_changes = []
                avg_history = []
                
                # Find minimum length of histories for average computation
                min_history_len = min(len(history) for history in mutation_histories if history)
                
                # Calculate statistics for each run
                for history in mutation_histories:
                    if history:
                        max_rates.append(max(history))
                        min_rates.append(min(history))
                        rate_changes.append(max(history) - min(history))
                
                # Calculate average mutation rate at each generation
                if min_history_len > 0:
                    for gen in range(min_history_len):
                        gen_rates = [history[gen] for history in mutation_histories if len(history) > gen]
                        if gen_rates:
                            avg_history.append(sum(gen_rates) / len(gen_rates))
                
                processed_stats[color_count][config_name]['mutation_rate_stats'] = {
                    'avg_max_rate': sum(max_rates) / len(max_rates) if max_rates else 0,
                    'avg_min_rate': sum(min_rates) / len(min_rates) if min_rates else 0,
                    'avg_rate_change': sum(rate_changes) / len(rate_changes) if rate_changes else 0,
                    'avg_mutation_history': avg_history
                }

            # Process fitness history
            best_histories = results['fitness_history']
            average_population_histories = results['avg_population_fitness_history']
            best_fitness_groupped_history = []
            average_population_groupped_history = []
            
            # Find min and max lengths of histories
            min_history_len = min(len(history) for history in best_histories)
            max_history_len = max(len(history) for history in best_histories)
                
            # Calculate early stopping statistics
            early_stopping_count = sum(1 for history in best_histories if len(history) < max_history_len)
            avg_stopping_gen = sum(len(history) for history in best_histories) / len(best_histories)
                
            processed_stats[color_count][config_name]['early_stopping_info'] = {
                'early_stopping_count': early_stopping_count,
                'early_stopping_percentage': (early_stopping_count / len(best_histories)) * 100,
                'min_generations': min_history_len,
                'max_generations': max_history_len,
                'avg_generations': avg_stopping_gen
            }
                
            # Calculate average best fitness at each generation
            for gen in range(max_history_len):
                gen_fitness = [history[gen] for history in best_histories if len(history) > gen]
                best_fitness_groupped_history.append(sum(gen_fitness) / len(gen_fitness))
                gen_avg_population = [history[gen] for history in average_population_histories if len(history) > gen]
                average_population_groupped_history.append(sum(gen_avg_population) / len(gen_avg_population))

            processed_stats[color_count][config_name]['best_fitness_groupped_history'] = best_fitness_groupped_history
            processed_stats[color_count][config_name]['average_population_groupped_history'] = average_population_groupped_history
            
    return processed_stats

def generate_test_results_visualizations(stats, test_type, color_counts, vis_dir):
    """
    Generate visualizations based on the test results.

    Args:
        stats (dict): Processed statistics from the test results.
        test_type (str): Type of test being visualized.
        color_counts (list): List of color counts tested.
        vis_dir (str): Directory to save visualizations.

    Returns:
        None
    """
    # Plot fitness comparison for each color count
    for num_colors in color_counts:
        plt.figure(figsize=(12, 8))
        
        configs = list(stats[num_colors].keys())
        
        # Plot fitness evolution for each configuration
        for config_name in configs:
            # Plot best fitness with solid lines
            best_history = stats[num_colors][config_name]['best_fitness_groupped_history']
            plt.plot(range(1, len(best_history) + 1), best_history, '-', 
                     linewidth=2, label=f"{config_name} (Best)")
            
            # Plot average population fitness with dashed lines
            avg_history = stats[num_colors][config_name]['average_population_groupped_history']
            plt.plot(range(1, len(avg_history) + 1), avg_history, '--', 
                     linewidth=1.5, alpha=0.7, label=f"{config_name} (Avg Pop)")
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title(f'Fitness Evolution ({num_colors} colors)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(vis_dir, f"fitness_evolution_{num_colors}.png"), dpi=300)
        plt.close()
    
    # Plot execution time comparison
    plt.figure(figsize=(12, 8))
    
    x = range(len(color_counts))
    bar_width = 0.8 / len(stats[color_counts[0]])
    
    for i, config_name in enumerate(stats[color_counts[0]].keys()):
        times = [stats[num_colors][config_name]['avg_time'] for num_colors in color_counts]
        plt.bar([p + i * bar_width for p in x], times, bar_width, 
                label=config_name, alpha=0.7)
    
    plt.xlabel('Number of Colors')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('Execution Time by Configuration and Color Count')
    plt.xticks([p + bar_width for p in x], color_counts)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(vis_dir, "execution_time_comparison.png"), dpi=300)
    plt.close()
    
    # Plot best fitness comparison
    plt.figure(figsize=(12, 8))
    
    for i, config_name in enumerate(stats[color_counts[0]].keys()):
        fitness = [stats[num_colors][config_name]['avg_best_fitness'] for num_colors in color_counts]
        plt.bar([p + i * bar_width for p in x], fitness, bar_width, 
                label=config_name, alpha=0.7)
    
    plt.xlabel('Number of Colors')
    plt.ylabel('Average Best Fitness')
    plt.title('Best Fitness by Configuration and Color Count')
    plt.xticks([p + bar_width for p in x], color_counts)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(vis_dir, "best_fitness_comparison.png"), dpi=300)
    plt.close()
    
    # Add cache hit ratio visualization if available (for caching test)
    if test_type == 'caching':
        plt.figure(figsize=(12, 7))
            
        for i, config_name in enumerate(stats[color_counts[0]].keys()):
            if 'cache_stats' in stats[color_counts[0]][config_name]:
                hit_ratios = [stats[num_colors][config_name]['cache_stats']['overall_hit_ratio'] 
                            for num_colors in color_counts]
                plt.bar([p + i * bar_width for p in x], hit_ratios, bar_width, 
                        label=config_name, alpha=0.7)
            
        plt.xlabel('Number of Colors')
        plt.ylabel('Cache Hit Ratio')
        plt.title('Cache Hit Ratio by Configuration and Color Count')
        plt.xticks([p + bar_width for p in x], color_counts)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
            
        plt.savefig(os.path.join(vis_dir, "cache_hit_ratio.png"), dpi=300)
        plt.close()
    
    # Add stagnation frequency visualization
    plt.figure(figsize=(12, 7))
        
    for i, config_name in enumerate(stats[color_counts[0]].keys()):
        if 'stagnation_stats' in stats[color_counts[0]][config_name]:
            stagnation_freq = [stats[num_colors][config_name]['stagnation_stats']['stagnation_frequency'] 
                            for num_colors in color_counts]
            plt.bar([p + i * bar_width for p in x], stagnation_freq, bar_width, 
                    label=config_name, alpha=0.7)
        
    plt.xlabel('Number of Colors')
    plt.ylabel('Stagnation Frequency')
    plt.title('Stagnation Frequency by Configuration and Color Count')
    plt.xticks([p + bar_width for p in x], color_counts)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
        
    plt.savefig(os.path.join(vis_dir, "stagnation_frequency.png"), dpi=300)
    plt.close()
    
    # Add mutation rate adaptation visualization if available
    for num_colors in color_counts:
        mutation_stats_available = any('mutation_rate_stats' in stats[num_colors][config] 
                                      for config in stats[num_colors])
        
        if mutation_stats_available:
            plt.figure(figsize=(12, 7))
            
            for config_name in stats[num_colors].keys():
                if 'mutation_rate_stats' in stats[num_colors][config_name]:
                    mutation_history = stats[num_colors][config_name]['mutation_rate_stats']['avg_mutation_history']
                    if mutation_history:
                        plt.plot(range(1, len(mutation_history) + 1), mutation_history, 
                                '-', linewidth=2, label=config_name)
            
            plt.xlabel('Generation')
            plt.ylabel('Mutation Rate')
            plt.title(f'Mutation Rate Adaptation ({num_colors} colors)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(vis_dir, f"mutation_adaptation_{num_colors}.png"), dpi=300)
            plt.close()

def generate_test_report(stats, best_images, test_type, color_counts, image_name, repetitions, output_dir):
    """
    Generate a comprehensive report of the test results.

    Args:
        stats (dict): Processed statistics from the test results.
        best_images (dict): Best images generated during the tests.
        test_type (str): Type of test performed.
        color_counts (list): List of color counts tested.
        image_name (str): Name of the image used for testing.
        repetitions (int): Number of repetitions per configuration.
        output_dir (str): Directory to save the report.
    
    Returns:
        None
    """
    report_path = os.path.join(output_dir, "analysis_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Genetic Algorithm Analysis Report\n\n")
        f.write(f"**Test Type:** {test_type}\n\n")
        f.write(f"**Test Image:** {image_name}\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Repetitions per configuration:** {repetitions}\n\n")
        
        f.write("## Performance Summary\n\n")
        
        # Summary for each color count
        for num_colors in color_counts:
            f.write(f"\n### {num_colors} Colors\n\n")
            
            f.write("| Configuration | Average Best Fitness | Execution Time (s) |\n")
            f.write("|---------------|----------------------|--------------------|\n")
            
            for config_name, config_stats in stats[num_colors].items():
                f.write(f"| {config_name} | {config_stats['avg_best_fitness']:.6f} ± {config_stats['std_best_fitness']:.6f} | ")
                f.write(f"{config_stats['avg_time']:.2f} ± {config_stats['std_time']:.2f} |\n")
            
            f.write("\n")
        
        if test_type in ['caching', 'new']:
            f.write("\n## Cache Performance Analysis\n\n")
            f.write("| Color Count | Configuration | Cache Hit Ratio | Total Hits | Total Misses |\n")
            f.write("|-------------|---------------|-----------------|------------|-------------|\n")
            
            for num_colors in color_counts:
                for config_name, config_stats in stats[num_colors].items():
                    if 'cache_stats' in config_stats:
                        cache_stats = config_stats['cache_stats']
                        f.write(f"| {num_colors} | {config_name} | {cache_stats['overall_hit_ratio']:.2%} | ")
                        f.write(f"{int(cache_stats['total_cache_hits'])} | {int(cache_stats['total_cache_misses'])} |\n")

        
        if test_type in ['early_stopping', 'adaptation', 'selection', 'new']:
            stagnation_rates = []
            for num_colors in color_counts:
                for config_name, config_stats in stats[num_colors].items():
                    if 'stagnation_stats' in config_stats:
                        stagnation_rates.append(config_stats['stagnation_stats']['stagnation_frequency'])
            
            f.write("\n## Stagnation Analysis\n\n")
            f.write("| Color Count | Configuration | Stagnation Frequency | Generations with Stagnation |\n")
            f.write("|-------------|---------------|----------------------|---------------------|\n")
                
            for num_colors in color_counts:
                for config_name, config_stats in stats[num_colors].items():
                    if 'stagnation_stats' in config_stats:
                        stag_stats = config_stats['stagnation_stats']
                        f.write(f"| {num_colors} | {config_name} | {stag_stats['stagnation_frequency']:.2%} | ")
                        f.write(f"{stag_stats['generations_with_stagnation']}/{stag_stats['total_generations']} |\n")

        if test_type in ['early_stopping', 'adaptation', 'new']:
            f.write("\n## Mutation Rate Adaptation Analysis\n\n")
            f.write("| Color Count | Configuration | Min Rate | Max Rate | Average Change |\n")
            f.write("|-------------|---------------|----------|----------|---------------|\n")
            
            for num_colors in color_counts:
                for config_name, config_stats in stats[num_colors].items():
                    if 'mutation_rate_stats' in config_stats:
                        mut_stats = config_stats['mutation_rate_stats']
                        f.write(f"| {num_colors} | {config_name} | {mut_stats['avg_min_rate']:.3f} | ")
                        f.write(f"{mut_stats['avg_max_rate']:.3f} | {mut_stats['avg_rate_change']:.3f} |\n")
        
        if test_type in ['early_stopping', 'adaptation', 'new']:
            f.write("\n## Early Stopping Analysis\n\n")
            f.write("| Color Count | Configuration | Early Stopping % | Avg Generations |\n")
            f.write("|-------------|---------------|------------------|----------------|\n")
            
            for num_colors in color_counts:
                for config_name, config_stats in stats[num_colors].items():
                    if 'early_stopping_info' in config_stats:
                        es_stats = config_stats['early_stopping_info']
                        f.write(f"| {num_colors} | {config_name} | {es_stats['early_stopping_percentage']:.1f}% | ")
                        f.write(f"{es_stats['avg_generations']:.1f} |\n")
            
            f.write("\nEarly stopping reduces unnecessary generations when no improvement is detected.\n")
        
        elif test_type == 'crossover':
            f.write("\n## Crossover Method Analysis\n\n")
            f.write("| Color Count | Configuration | Method | Crossover Rate | Avg Fitness | Time (s) |\n")
            f.write("|-------------|---------------|--------|---------------|-------------|----------|\n")
            
            # Use all color counts for crossover analysis
            for num_colors in color_counts:
                for config_name, config_stats in stats[num_colors].items():
                    method = config_name.split('_')[0] if '_' in config_name else config_name
                    rate = "0%" if "no_crossover" in config_name else "80%" if "rate" not in config_name else config_name.split('_')[-1] + "%"
                    f.write(f"| {num_colors} | {config_name} | {method} | {rate} | ")
                    f.write(f"{config_stats['avg_best_fitness']:.6f} | {config_stats['avg_time']:.2f} |\n")
        
        f.write("\n## Reference Images\n\n")
        f.write("Best results from each configuration:\n\n")

        # Show all color counts and configurations
        for num_colors in color_counts:
            f.write(f"\n### {num_colors} Colors\n\n")
            
            # Sort configurations by fitness for organized display
            sorted_configs = sorted(
                best_images[num_colors].items(), 
                key=lambda x: x[1]['fitness'], 
                reverse=True
            )
            
            # Use a grid layout for more efficient display
            grid_size = min(3, len(sorted_configs))  # Up to 3 images per row
            if grid_size > 1:
                f.write("<div style='display: grid; grid-template-columns: repeat({}, 1fr); gap: 10px;'>\n".format(grid_size))
            
            # Show all configurations
            for config_name, image_info in sorted_configs:
                # Get relative path for markdown
                rel_path = os.path.relpath(image_info['image_path'], output_dir)
                
                if grid_size > 1:
                    f.write("<div style='text-align: center;'>\n")
                    f.write(f"<p><strong>{config_name}</strong></p>\n")
                    f.write(f"<img src='{rel_path}' style='width: 100%; max-width: 300px;'>\n")
                    f.write(f"<p>Fitness: {image_info['fitness']:.6f}</p>\n")
                    f.write("</div>\n")
                else:
                    f.write(f"\n#### {config_name}\n\n")
                    f.write(f"![{config_name}]({rel_path})\n\n")
                    f.write(f"Fitness: {image_info['fitness']:.6f}\n\n")
            
            if grid_size > 1:
                f.write("</div>\n\n")
    
    print(f"Report generated: {report_path}")
    return report_path

def run_rgb_comparison(configurations, output_dir):
    """
    Generate visual comparison between RGB distance and perceptual metrics.
    Shows original image, kmeans initialization, and final results side by side.

    Args:
        configurations (list): List of configurations for RGB and perceptual distance.
        configurations (dict): Dictionary containing configurations for the test.
        output_dir (str): Directory to save the results. If None, a timestamped directory will be created.

    Returns:
        None. Saves images and analysis results to the specified directory.
    """
    aux_output_dir = os.path.join(output_dir, "auxiliary_images")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(aux_output_dir, exist_ok=True)
    print(f"RGB distance comparison results will be saved to {output_dir}")

    test_image = 'squareFour.jpg'
    
    # Define configurations:
    rgb_config = configurations[0]
    perceptual_config = configurations[1]

    
    original_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", test_image)
    original_image = load_image(original_image_path)
        
    # Run with RGB distance
    print("\nRunning with RGB distance metric...")
    _, rgb_fitness, rgb_history, _, rgb_image_result, _, rgb_kmeans_image, _ = run_image_quantization(
        image_name=test_image,
        **{k: v for k, v in rgb_config.items() if k != 'name'}
    )
        
    # Save RGB results
    rgb_image_path = os.path.join(aux_output_dir, "rgb_result.png")
    save_image(rgb_image_result, rgb_image_path)

    rgb_kmeans_path = os.path.join(aux_output_dir, "rgb_kmeans_init.png")
    save_image(rgb_kmeans_image, rgb_kmeans_path)

        
    # Run with perceptual distance
    print("\nRunning with perceptual distance metric...")
    _, perceptual_fitness, perceptual_history, _, perceptual_img_result, _, perceptual_kmeans_image, _ = run_image_quantization(
        image_name=test_image,
        **{k: v for k, v in perceptual_config.items() if k != 'name'}
    )
        
    # Save perceptual results
    perceptual_img_path = os.path.join(aux_output_dir, "perceptual_result.png")
    save_image(perceptual_img_result, perceptual_img_path)
        
    perceptual_kmeans_path = os.path.join(aux_output_dir, "perceptual_kmeans_init.png")
    save_image(perceptual_kmeans_image, perceptual_kmeans_path)
        
    # Create visualization of the results
    plt.figure(figsize=(15, 10))
        
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')
        
    # RGB distance results
    plt.subplot(2, 3, 2)
    plt.title("RGB Distance - KMeans Initialization")
    plt.imshow(rgb_kmeans_image)
    plt.axis('off')
        
    plt.subplot(2, 3, 3)
    plt.imshow(rgb_image_result)
    plt.title(f"RGB Distance - Final\nFitness: {rgb_fitness:.6f}")
    plt.axis('off')
        
    # Perceptual distance results
    plt.subplot(2, 3, 5)
    plt.title("Perceptual Distance - KMeans Initialization")
    plt.imshow(perceptual_kmeans_image)
    plt.axis('off')
        
    plt.subplot(2, 3, 6)
    plt.imshow(perceptual_img_result)
    plt.title(f"Perceptual Distance - Final\nFitness: {perceptual_fitness:.6f}")
    plt.axis('off')
        

    plt.subplot(2, 3, 4)
    plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distance_metric_comparison.png"), dpi=300)
    plt.close()

    # Side by side comparison of fitness along generations
    plt.figure(figsize=(16, 6))

    # First plot: RGB Distance fitness
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(rgb_history) + 1), rgb_history, 'r-', linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("RGB Distance Fitness")
    plt.grid(True, alpha=0.3)

    # Second plot: Perceptual Distance fitness 
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(perceptual_history) + 1), perceptual_history, 'b-', linewidth=2)
    plt.xlabel("Generation") 
    plt.ylabel("Fitness")
    plt.title("Perceptual Distance Fitness")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distance_metric_fitness_side_by_side.png"), dpi=300)
    plt.close()

    print("\nRGB distance comparison complete. Results saved to:", output_dir)


def run_algorithm(
    image_name=DEFAULT_IMAGE,
    num_colors=16,
    config=CONFIGURATIONS[0],
    output_dir=None,
    verbose=True
):
    """
    Run a single genetic algorithm execution.
    
    Args:
        image_name (str): Name of the image to process
        num_colors (int): Number of colors in the palette
        config (dict): Configuration dictionary. If None, uses the basic configuration
        output_dir (str): Directory to save results. If None, creates a timestamped directory
        verbose (bool): Whether to print progress information
    
    Returns:
        dict: Results containing final color palette, final fitness value, historical of best fitness each generation, average fitness for each generation, the resulting image, the palette for the kMeans initialization, the image obtained by applying the kMeans palette, the execution time, a performance log with data about the performance of the algorithm, and the output directory.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    if output_dir is None:
        output_dir = os.path.join("results", f"single_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Add number of colors to the configuration
    config['num_colors'] = num_colors
    
    if verbose:
        print(f"Running genetic algorithm on {image_name} with {num_colors} colors")
        print(f"Results will be saved to: {output_dir}")
        print(f"Configuration: {config['name']}")
    
    # Run the algorithm
    start_time = datetime.now()
    
    palette, fitness, history, average_history, result_img, kmeans_palette, kmeans_image, performance_log = run_image_quantization(
        image_name=image_name,
        output_dir=output_dir,
        save_results=True,
        **{k: v for k, v in config.items() if k not in ['name', 'num_colors']}
    )
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    if verbose:
        print(f"Algorithm completed in {execution_time:.2f} seconds")
        print(f"Best fitness achieved: {fitness:.6f}")
        print(f"Results saved to: {output_dir}")
    
    return {
        'palette': palette,
        'fitness': fitness,
        'best_fitness_history': history,
        'avg_fitness_history': average_history,
        'result_image': result_img,
        'kmeans_palette': kmeans_palette,
        'kmeans_image': kmeans_image,
        'execution_time': execution_time,
        'performance_log': performance_log,
        'output_dir': output_dir
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze genetic algorithm performance")
    parser.add_argument(
        "--test_type",
        type=str,
        choices=['caching', 'restriction_kmeans', 'selection', 'mutation','crossover', 'adaptation', 'early_stopping', 'hyperparameter_tuning', 'rgb', 'new'],
        help="Type of test to run: 'caching', 'restriction_kmeans', 'selection', 'mutation','crossover', 'adaptation', 'early_stopping', 'hyperparameter_tuning', 'rgb' or 'new'"
    )
    
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run a single algorithm execution."
    )

    parser.add_argument(
        "--images",
        type=str,
        nargs='+',
        default=DEFAULT_IMAGE,
        help=f"Name of the image files to use (must be in the images folder). Default: {DEFAULT_IMAGE}"
    )
    parser.add_argument(
        "--colors",
        type=int,
        nargs='+',
        default=BASIC_COLOR_CHANGES,
        help=f"List of color counts to test. Default: {BASIC_COLOR_CHANGES}"
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=REPETITIONS,
        help=f"Number of times to repeat each test for statistical significance. Default: {REPETITIONS}"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results. Default: auto-generated in 'analysis/' folder"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress detailed progress information"
    )
    
    args = parser.parse_args()
    
    if not args.run and not args.test_type:
        parser.error("Either --run or --test_type must be specified")
    
    if args.run and args.test_type:
        parser.error("Cannot specify both --run and --test_type")
    

    if args.run:
        image_name = args.images[0] if isinstance(args.images, list) else args.images
        num_colors = args.colors[0] if isinstance(args.colors, list) else args.colors
        
        results = run_algorithm(
            image_name=image_name,
            num_colors=num_colors,
            output_dir=args.output_dir,
            verbose=not args.quiet
        )
        
        print(f"Single run complete. Results available in {results['output_dir']}")
    else:
        results = run_multiple_images_test(
            test_type=args.test_type,
            image_names=args.images,
            color_counts=args.colors,
            repetitions=args.repetitions,
            output_dir=args.output_dir,
            verbose=not args.quiet
        )
        
    
    print(f"Analysis complete. Results available in {results['output_dir']}")