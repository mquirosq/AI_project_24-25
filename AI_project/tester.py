import os
import re
import pandas as pd
import seaborn as sns
from datetime import datetime
import argparse
from run_algorithm import run_image_quantization
import matplotlib.pyplot as plt
from helper import (save_image, load_image)
    

"""
Configuration for image quantization experiments
"""

# Parameter configurations to test
CACHING_TEST_CONFIGS = [
        {
        'name': 'caching_enabled',
        'num_colors': 6,
        'restricted': False,
        'population_size': 20,
        'generations': 100,
        'mutation_rate': 0.2,
        'crossover_rate': 0.8,
        'elitism': 2,
        'halting_stagnation_threshold': None,
        'adaptation_rate': 1,
        'adaptation_threshold': 10,
        'kMeans': False,
        'selection_method': 'tournament',
        'tournament_size': 3,
        'mutate_diverse': False,
        'crossover_method': 'one_point',
        'use_caching': True,  # With caching
        'display': False
    },
    {
        'name': 'caching_disabled',
        'num_colors': 6,
        'restricted': False,
        'population_size': 20,
        'generations': 100,
        'mutation_rate': 0.2,
        'crossover_rate': 0.8,
        'elitism': 2,
        'halting_stagnation_threshold': None,
        'adaptation_rate': 1,
        'adaptation_threshold': 10,
        'kMeans': False,
        'selection_method': 'tournament',
        'tournament_size': 3,
        'mutate_diverse': False,
        'crossover_method': 'one_point',
        'use_caching': False,  # Without caching
        'display': False
    }
]

CONFIGURATIONS = [
    # Basic configuration
    {
        'name': 'basic',
        'num_colors': 6,
        
        'restricted': False,
        
        'population_size': 20,
        'generations': 100,
        'mutation_rate': 0.2,
        'crossover_rate': 0.8,
        'elitism': 2,
        'halting_stagnation_threshold': 20,

        'adaptation_rate': 1.1,
        'adaptation_threshold': 10,

        'kMeans': False,

        'selection_method': 'tournament',
        'tournament_size': 3,
        'mutate_diverse': False,
        'crossover_method': 'one_point',
        
        'use_caching': False,
        'display': False
    },
    # Basic caching
    {
        'name': 'basic',
        'num_colors': 6,
        
        'restricted': False,
        
        'population_size': 20,
        'generations': 100,
        'mutation_rate': 0.2,
        'crossover_rate': 0.8,
        'elitism': 2,
        'halting_stagnation_threshold': 20,

        'adaptation_rate': 1.1,
        'adaptation_threshold': 10,

        'kMeans': False,

        'selection_method': 'tournament',
        'tournament_size': 3,
        'mutate_diverse': False,
        'crossover_method': 'one_point',
        
        'use_caching': True,
        'display': False
    },

    # High mutation rate
    {
        'name': 'high_mutation',
        'num_colors': 8,
        'restricted': False,
        'population_size': 30,
        'generations': 30,
        'mutation_rate': 0.4,  # Changed from baseline
        'crossover_rate': 0.8,
        'selection_method': 'tournament',
        'tournament_size': 3,
        'adaptation_rate': 1.2,
        'adaptation_threshold': 10,
        'halting_stagnation_threshold': 20,
        'kMeans': False,
        'mutate_diverse': False,
        'crossover_method': 'one_point',
        'display': False
    },
    # Larger population
    {
        'name': 'large_population',
        'num_colors': 8,
        'restricted': False,
        'population_size': 50,  # Changed from baseline
        'generations': 30,
        'mutation_rate': 0.2,
        'crossover_rate': 0.8,
        'selection_method': 'tournament',
        'tournament_size': 3,
        'adaptation_rate': 1.2,
        'adaptation_threshold': 10,
        'halting_stagnation_threshold': 20,
        'kMeans': False,
        'mutate_diverse': False,
        'crossover_method': 'one_point',
        'display': False
    },
    # With kMeans initialization
    {
        'name': 'kmeans_init',
        'num_colors': 8,
        'restricted': False,
        'population_size': 30,
        'generations': 30,
        'mutation_rate': 0.2,
        'crossover_rate': 0.8,
        'selection_method': 'tournament',
        'tournament_size': 3,
        'adaptation_rate': 1.2,
        'adaptation_threshold': 10,
        'halting_stagnation_threshold': 20,
        'kMeans': True,  # Changed from baseline
        'mutate_diverse': False,
        'crossover_method': 'one_point',
        'display': False
    },
    # Diverse mutation
    {
        'name': 'diverse_mutation',
        'num_colors': 8,
        'restricted': False,
        'population_size': 30,
        'generations': 30,
        'mutation_rate': 0.2,
        'crossover_rate': 0.8,
        'selection_method': 'tournament',
        'tournament_size': 3,
        'adaptation_rate': 1.2,
        'adaptation_threshold': 10,
        'halting_stagnation_threshold': 20,
        'kMeans': False,
        'mutate_diverse': True,  # Changed from baseline
        'crossover_method': 'one_point',
        'display': False
    },
    # Advanced crossover
    {
        'name': 'advanced_crossover',
        'num_colors': 8,
        'restricted': False,
        'population_size': 30,
        'generations': 30,
        'mutation_rate': 0.2,
        'crossover_rate': 0.8,
        'selection_method': 'tournament',
        'tournament_size': 3,
        'adaptation_rate': 1.2,
        'adaptation_threshold': 10,
        'halting_stagnation_threshold': 20,
        'kMeans': False,
        'mutate_diverse': False,
        'crossover_method': 'closest_pairs',  # Changed from baseline
        'display': False
    },
    # Restricted palette
    {
        'name': 'restricted_palette',
        'num_colors': 8,
        'restricted': True,  # Changed from baseline
        'population_size': 30,
        'generations': 30,
        'mutation_rate': 0.2,
        'crossover_rate': 0.8,
        'selection_method': 'tournament',
        'tournament_size': 3,
        'adaptation_rate': 1.2,
        'adaptation_threshold': 10,
        'halting_stagnation_threshold': 20,
        'kMeans': False,
        'mutate_diverse': False,
        'crossover_method': 'one_point',
        'display': False
    },
    # Optimized combination (based on preliminary tests)
    {
        'name': 'optimized',
        'num_colors': 8,
        'restricted': False,
        'population_size': 50,
        'generations': 30,
        'mutation_rate': 0.2,
        'crossover_rate': 0.8,
        'selection_method': 'tournament',
        'tournament_size': 3,
        'adaptation_rate': 1.2,
        'adaptation_threshold': 10,
        'halting_stagnation_threshold': 20,
        'kMeans': True,
        'mutate_diverse': True,
        'crossover_method': 'closest_pairs',
        'display': False
    }
]

# Number of repetitions for each configuration to get statistical significance
REPETITIONS = 3
BASIC_COLOR_CHANGES = [4, 6, 10]

"""
Experiment analyzer for image quantization genetic algorithm
"""

def extract_data_from_performance_report(file_path):
    """
    Extract key metrics from a performance report file
    
    Args:
        file_path (str): Path to the performance report file
    Returns:
        dict: A dictionary containing the extracted metrics
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract basic information
        image_name = re.search(r'Image Name: ([^\n]+)', content)
        image_name = image_name.group(1) if image_name else "unknown"
        
        num_colors = re.search(r'Number of Colors: (\d+)', content)
        num_colors = int(num_colors.group(1)) if num_colors else 0
        
        population_size_matches = re.findall(r'Population Size: (\d+)', content)
        population_size = int(population_size_matches[0]) if population_size_matches else 0
        
        generations = re.search(r'Generations: (\d+)', content)
        generations = int(generations.group(1)) if generations else 0
        
        mutation_rate = re.search(r'Mutation Rate: ([\d\.]+)', content)
        mutation_rate = float(mutation_rate.group(1)) if mutation_rate else 0.0
        
        crossover_rate = re.search(r'Crossover Rate: ([\d\.]+)', content)
        crossover_rate = float(crossover_rate.group(1)) if crossover_rate else 0.0
        
        elitism = re.search(r'Elitism: (\d+)', content)
        elitism = int(elitism.group(1)) if elitism else 0
        
        selection_method = re.search(r'Selection Method: ([^\n]+)', content)
        selection_method = selection_method.group(1).strip() if selection_method else "unknown"
        
        adaptation_rate = re.search(r'Adaptation Rate: ([\d\.]+)', content)
        adaptation_rate = float(adaptation_rate.group(1)) if adaptation_rate else 1.0
        
        kmeans_pattern = re.search(r'Uses KMeans: ([^\n]+)', content)
        kmeans = kmeans_pattern and "Yes" in kmeans_pattern.group(1)
        
        diverse_mutation_pattern = re.search(r'Uses Diverse Mutation: ([^\n]+)', content)
        diverse_mutation = diverse_mutation_pattern and "Yes" in diverse_mutation_pattern.group(1)
        
        crossover_method = re.search(r'Crossover Method: ([^\n]+)', content)
        crossover_method = crossover_method.group(1).strip() if crossover_method else "unknown"
        
        # Extract performance metrics
        best_fitness = re.search(r'Best Fitness: ([\d\.]+)', content)
        best_fitness = float(best_fitness.group(1)) if best_fitness else 0.0
        
        best_generation = re.search(r'Found in Generation: (\d+)', content)
        best_generation = int(best_generation.group(1)) if best_generation else 0
        
        execution_time = re.search(r'Total Execution Time: ([\d\.]+)', content)
        execution_time = float(execution_time.group(1)) if execution_time else 0.0
        
        avg_time_per_gen = re.search(r'Average Time per Generation: ([\d\.]+)', content)
        avg_time_per_gen = float(avg_time_per_gen.group(1)) if avg_time_per_gen else 0.0
        
        total_generations = re.search(r'Total Generations: (\d+)', content)
        total_generations = int(total_generations.group(1)) if total_generations else 0
        
        # Extract cache statistics
        cache_hits_pattern = re.search(r'Cache Hits: (\d+) \(([\d\.]+)%\)', content)
        cache_hits = int(cache_hits_pattern.group(1)) if cache_hits_pattern else 0
        cache_hit_rate = float(cache_hits_pattern.group(2))/100 if cache_hits_pattern else 0.0
        
        # Extract fitness history from the generation statistics table
        fitness_history = []
        avg_fitness_history = []
        gen_stats_match = re.search(r'GENERATION STATISTICS(.*?)CACHE STATISTICS', content, re.DOTALL)
        
        if gen_stats_match:
            gen_stats_text = gen_stats_match.group(1)
            lines = gen_stats_text.strip().split('\n')
            # Skip header lines
            for line in lines[3:]:
                parts = line.split('|')
                if len(parts) >= 3:
                    try:
                        fitness_history.append(float(parts[1].strip()))
                        avg_fitness_history.append(float(parts[2].strip()))
                    except ValueError:
                        pass
        
        # Determine if this is a restricted run based on directory path
        restricted = 'restricted_' in file_path.lower() and 'unrestricted' not in file_path.lower()
        
        # Extract folder name to help identify the run
        folder_name = os.path.basename(os.path.dirname(file_path))
        
        # Create a record with all extracted data
        record = {
            'image_name': image_name,
            'num_colors': num_colors,
            'restricted': restricted,
            'population_size': population_size,
            'generations': generations,
            'mutation_rate': mutation_rate,
            'crossover_rate': crossover_rate,
            'elitism': elitism,
            'selection_method': selection_method,
            'adaptation_rate': adaptation_rate,
            'uses_kmeans': kmeans,
            'uses_diverse_mutation': diverse_mutation,
            'crossover_method': crossover_method,
            'best_fitness': best_fitness,
            'best_generation': best_generation,
            'execution_time': execution_time,
            'avg_time_per_gen': avg_time_per_gen,
            'total_generations': total_generations,
            'cache_hits': cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'folder_name': folder_name,
            'file_path': file_path,
            'fitness_history': fitness_history,
            'avg_fitness_history': avg_fitness_history
        }
        
        return record
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def find_performance_reports(base_dir=None):
    """
    Find all performance report files
    
    Args:
        base_dir (str): Base directory to search for report files
    Returns:
        list: List of paths to performance report files
    """

    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "tests")
    
    report_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.startswith("performance_report_") and file.endswith(".txt"):
                report_files.append(os.path.join(root, file))
    
    print(f"Found {len(report_files)} performance report files")
    return report_files

def collect_and_analyze_reports(output_dir=None):
    """
    Collect and analyze all performance reports
    
    Args:
        output_dir (str): Directory to save the analysis results
    Returns:
        pd.DataFrame: DataFrame containing the analysis results
    """
    # Create output directory for analysis
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join("analysis", f"report_analysis_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Analysis results will be saved to {output_dir}")

    report_files = find_performance_reports()
    
    if not report_files:
        print("No performance reports found. Run the genetic algorithm first.")
        return None
    
    # Process all reports
    records = []
    for file_path in report_files:
        record = extract_data_from_performance_report(file_path)
        if record:
            records.append(record)
    
    if not records:
        print("No valid records extracted from performance reports.")
        return None
    
    df = pd.DataFrame(records)
    
    columns_to_save = [col for col in df.columns if col not in ['fitness_history', 'avg_fitness_history']]
    df[columns_to_save].to_csv(os.path.join(output_dir, "all_performance_data.csv"), index=False)
    
    # Group runs by configuration parameters
    df['config_group'] = df.apply(
        lambda row: (
            f"{row['selection_method']}_" +
            f"m{row['mutation_rate']}_" +
            f"c{row['crossover_rate']}_" +
            f"p{row['population_size']}_" +
            f"{'r' if row['restricted'] else 'u'}_" +
            f"{'k' if row['uses_kmeans'] else 'n'}_" +
            f"{'d' if row['uses_diverse_mutation'] else 's'}_" +
            f"{row['crossover_method']}"
        ),
        axis=1
    )
    
    generate_summary_statistics(df, output_dir)
    
    generate_visualizations(df, output_dir)
    
    print(f"Analysis complete! Results saved to {output_dir}")
    return df

def generate_summary_statistics(df, output_dir):
    """
    Generate summary statistics from the performance data
    
    Args:
        df (pd.DataFrame): DataFrame containing performance data
        output_dir (str): Directory to save the summary statistics
    
    Returns:
        None. Saves summary statistics to CSV files.
    """
    
    # 1. Summary by image
    image_summary = df.groupby('image_name').agg({
        'best_fitness': ['count', 'mean', 'std', 'min', 'max'],
        'execution_time': ['mean', 'min', 'max'],
        'best_generation': ['mean', 'min', 'max']
    })
    
    image_summary.to_csv(os.path.join(output_dir, "summary_by_image.csv"))
    
    # 2. Summary by configuration group
    config_summary = df.groupby('config_group').agg({
        'best_fitness': ['count', 'mean', 'std', 'min', 'max'],
        'execution_time': ['mean', 'min', 'max'],
        'best_generation': ['mean', 'min', 'max']
    })
    
    config_summary.to_csv(os.path.join(output_dir, "summary_by_config.csv"))
    
    # 3. Summary by selection method
    selection_summary = df.groupby('selection_method').agg({
        'best_fitness': ['count', 'mean', 'std', 'min', 'max'],
        'execution_time': ['mean', 'min', 'max'],
        'best_generation': ['mean', 'min', 'max']
    })
    
    selection_summary.to_csv(os.path.join(output_dir, "summary_by_selection.csv"))
    
    # 4. Summary by mutation rate
    mutation_summary = df.groupby('mutation_rate').agg({
        'best_fitness': ['count', 'mean', 'std', 'min', 'max'],
        'execution_time': ['mean', 'min', 'max'],
        'best_generation': ['mean', 'min', 'max']
    })
    
    mutation_summary.to_csv(os.path.join(output_dir, "summary_by_mutation.csv"))
    
    # 5. Summary by restricted/unrestricted
    restricted_summary = df.groupby('restricted').agg({
        'best_fitness': ['count', 'mean', 'std', 'min', 'max'],
        'execution_time': ['mean', 'min', 'max'],
        'best_generation': ['mean', 'min', 'max']
    })
    
    restricted_summary.to_csv(os.path.join(output_dir, "summary_by_restricted.csv"))
    
    # Create a comprehensive report in markdown format
    with open(os.path.join(output_dir, "summary_report.md"), 'w') as f:
        f.write("# Genetic Algorithm Performance Analysis\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total Runs Analyzed: {len(df)}\n\n")
        
        f.write("## Summary by Image\n\n")
        f.write("| Image | Runs | Avg Fitness | Std Dev | Min Fitness | Max Fitness | Avg Time | Avg Gen |\n")
        f.write("|-------|------|------------|---------|-------------|-------------|----------|--------|\n")
        
        for idx, row in image_summary.iterrows():
            f.write(f"| {idx} | {row[('best_fitness', 'count')]} | ")
            f.write(f"{row[('best_fitness', 'mean')]:.6f} | {row[('best_fitness', 'std')]:.6f} | ")
            f.write(f"{row[('best_fitness', 'min')]:.6f} | {row[('best_fitness', 'max')]:.6f} | ")
            f.write(f"{row[('execution_time', 'mean')]:.2f}s | {row[('best_generation', 'mean')]:.1f} |\n")
        
        f.write("\n## Summary by Selection Method\n\n")
        f.write("| Method | Runs | Avg Fitness | Std Dev | Min Fitness | Max Fitness | Avg Time | Avg Gen |\n")
        f.write("|-------|------|------------|---------|-------------|-------------|----------|--------|\n")
        
        for idx, row in selection_summary.iterrows():
            f.write(f"| {idx} | {row[('best_fitness', 'count')]} | ")
            f.write(f"{row[('best_fitness', 'mean')]:.6f} | {row[('best_fitness', 'std')]:.6f} | ")
            f.write(f"{row[('best_fitness', 'min')]:.6f} | {row[('best_fitness', 'max')]:.6f} | ")
            f.write(f"{row[('execution_time', 'mean')]:.2f}s | {row[('best_generation', 'mean')]:.1f} |\n")
        
        f.write("\n## Summary by Mutation Rate\n\n")
        f.write("| Rate | Runs | Avg Fitness | Std Dev | Min Fitness | Max Fitness | Avg Time | Avg Gen |\n")
        f.write("|------|------|------------|---------|-------------|-------------|----------|--------|\n")
        
        for idx, row in mutation_summary.iterrows():
            f.write(f"| {idx:.1f} | {row[('best_fitness', 'count')]} | ")
            f.write(f"{row[('best_fitness', 'mean')]:.6f} | {row[('best_fitness', 'std')]:.6f} | ")
            f.write(f"{row[('best_fitness', 'min')]:.6f} | {row[('best_fitness', 'max')]:.6f} | ")
            f.write(f"{row[('execution_time', 'mean')]:.2f}s | {row[('best_generation', 'mean')]:.1f} |\n")
        
        f.write("\n## Summary by Restricted/Unrestricted\n\n")
        f.write("| Type | Runs | Avg Fitness | Std Dev | Min Fitness | Max Fitness | Avg Time | Avg Gen |\n")
        f.write("|------|------|------------|---------|-------------|-------------|----------|--------|\n")
        
        for idx, row in restricted_summary.iterrows():
            f.write(f"| {'Restricted' if idx else 'Unrestricted'} | {row[('best_fitness', 'count')]} | ")
            f.write(f"{row[('best_fitness', 'mean')]:.6f} | {row[('best_fitness', 'std')]:.6f} | ")
            f.write(f"{row[('best_fitness', 'min')]:.6f} | {row[('best_fitness', 'max')]:.6f} | ")
            f.write(f"{row[('execution_time', 'mean')]:.2f}s | {row[('best_generation', 'mean')]:.1f} |\n")
        
        # Find the best configuration
        best_config_index = None
        best_fitness = -float('inf')
        
        for idx, row in config_summary.iterrows():
            if row[('best_fitness', 'mean')] > best_fitness:
                best_fitness = row[('best_fitness', 'mean')]
                best_config_index = idx
                
        if best_config_index is not None:
            best_config = config_summary.loc[best_config_index]
            
            f.write("\n## Best Configuration\n\n")
            f.write(f"**Configuration Group:** {best_config_index}\n\n")
            f.write(f"- **Average Fitness:** {best_config[('best_fitness', 'mean')]:.6f}\n")
            f.write(f"- **Number of Runs:** {best_config[('best_fitness', 'count')]}\n")
            f.write(f"- **Average Execution Time:** {best_config[('execution_time', 'mean')]:.2f} seconds\n")
            
            # Parse the configuration group
            config_parts = best_config_index.split('_')
            if len(config_parts) >= 8:
                f.write("**Parameters:**\n\n")
                f.write(f"- Selection Method: {config_parts[0]}\n")
                f.write(f"- Mutation Rate: {config_parts[1][1:]}\n")
                f.write(f"- Crossover Rate: {config_parts[2][1:]}\n")
                f.write(f"- Population Size: {config_parts[3][1:]}\n")
                f.write(f"- Restricted: {'Yes' if config_parts[4] == 'r' else 'No'}\n")
                f.write(f"- Uses KMeans: {'Yes' if config_parts[5] == 'k' else 'No'}\n")
                f.write(f"- Diverse Mutation: {'Yes' if config_parts[6] == 'd' else 'No'}\n")
                f.write(f"- Crossover Method: {config_parts[7]}\n")

def generate_visualizations(df, output_dir):
    """
    Generate visualizations from the performance data
    
    Args:
        df (pd.DataFrame): DataFrame containing performance data
        output_dir (str): Directory to save the visualizations
        
    Returns:
        None. Saves visualizations as PNG files.
    """

    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    # Only generate visualizations if we have enough unique values
    unique_selection_methods = df['selection_method'].nunique()
    unique_mutation_rates = df['mutation_rate'].nunique()
    unique_restricted = df['restricted'].nunique()
    unique_kmeans = df['uses_kmeans'].nunique()
    unique_diverse = df['uses_diverse_mutation'].nunique()
    unique_crossover = df['crossover_method'].nunique()
    unique_pop_size = df['population_size'].nunique()
    unique_images = df['image_name'].nunique()
    
    # 1. Selection Method Comparison
    if unique_selection_methods > 1:
        plt.figure(figsize=(12, 6))
        sns.barplot(x='selection_method', y='best_fitness', data=df, errorbar=('ci', 95))
        plt.title('Selection Method vs Best Fitness')
        plt.xlabel('Selection Method')
        plt.ylabel('Best Fitness')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "selection_method_comparison.png"), dpi=300)
        plt.close()
    
    # 2. Mutation Rate Comparison
    if unique_mutation_rates > 1:
        plt.figure(figsize=(12, 6))
        sns.barplot(x='mutation_rate', y='best_fitness', data=df, errorbar=('ci', 95))
        plt.title('Mutation Rate vs Best Fitness')
        plt.xlabel('Mutation Rate')
        plt.ylabel('Best Fitness')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "mutation_rate_comparison.png"), dpi=300)
        plt.close()
    
    # 3. Restricted vs Unrestricted
    if unique_restricted > 1:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='restricted', y='best_fitness', data=df, errorbar=('ci', 95))
        plt.title('Restricted vs Unrestricted Palette')
        plt.xlabel('Restricted Palette')
        plt.ylabel('Best Fitness')
        plt.xticks([0, 1], ['Unrestricted', 'Restricted'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "restricted_comparison.png"), dpi=300)
        plt.close()
    
    # 4. KMeans vs No KMeans
    if unique_kmeans > 1:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='uses_kmeans', y='best_fitness', data=df, errorbar=('ci', 95))
        plt.title('KMeans vs No KMeans Initialization')
        plt.xlabel('KMeans Initialization')
        plt.ylabel('Best Fitness')
        plt.xticks([0, 1], ['No KMeans', 'KMeans'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "kmeans_comparison.png"), dpi=300)
        plt.close()
    
    # 5. Diverse vs Standard Mutation
    if unique_diverse > 1:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='uses_diverse_mutation', y='best_fitness', data=df, errorbar=('ci', 95))
        plt.title('Diverse vs Standard Mutation')
        plt.xlabel('Diverse Mutation')
        plt.ylabel('Best Fitness')
        plt.xticks([0, 1], ['Standard Mutation', 'Diverse Mutation'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "mutation_type_comparison.png"), dpi=300)
        plt.close()
    
    # 6. Crossover Method Comparison
    if unique_crossover > 1:
        plt.figure(figsize=(12, 6))
        sns.barplot(x='crossover_method', y='best_fitness', data=df, errorbar=('ci', 95))
        plt.title('Crossover Method vs Best Fitness')
        plt.xlabel('Crossover Method')
        plt.ylabel('Best Fitness')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "crossover_method_comparison.png"), dpi=300)
        plt.close()
    
    # 7. Population Size Comparison
    if unique_pop_size > 1:
        plt.figure(figsize=(12, 6))
        sns.barplot(x='population_size', y='best_fitness', data=df, errorbar=('ci', 95))
        plt.title('Population Size vs Best Fitness')
        plt.xlabel('Population Size')
        plt.ylabel('Best Fitness')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "population_size_comparison.png"), dpi=300)
        plt.close()
    
    # 8. Image Type Comparison
    if unique_images > 1:
        plt.figure(figsize=(12, 6))
        sns.barplot(x='image_name', y='best_fitness', data=df, errorbar=('ci', 95))
        plt.title('Image Type vs Best Fitness')
        plt.xlabel('Image')
        plt.ylabel('Best Fitness')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "image_comparison.png"), dpi=300)
        plt.close()
    
    # 9. Runtime vs Fitness Scatter
    plt.figure(figsize=(12, 8))

    # Prepare parameters for scatterplot
    scatter_params = {
        'x': 'execution_time',
        'y': 'best_fitness',
        'data': df,
        'alpha': 0.7,
        's': 100
    }

    # Add optional parameters only if we have multiple values
    if unique_selection_methods > 1:
        scatter_params['hue'] = 'selection_method'
    if unique_restricted > 1:
        scatter_params['style'] = 'restricted'
    if unique_pop_size > 1:
        scatter_params['size'] = 'population_size'

    # Create scatter plot with all parameters
    sns.scatterplot(**scatter_params)
            
    plt.title('Runtime vs Fitness')
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Best Fitness')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "runtime_vs_fitness.png"), dpi=300)
    plt.close()
    
    # 10. Convergence Analysis (for selection methods)
    try:
        if 'fitness_history' in df.columns and any(len(history) > 0 for history in df['fitness_history'] if history is not None):
            
            selection_methods = df['selection_method'].unique()
            plt.figure(figsize=(12, 8))
            
            # For each selection method, find a representative run and plot its convergence
            for method in selection_methods:
                method_runs = df[df['selection_method'] == method]
                if not method_runs.empty:
                    # Find run closest to the average fitness for this method
                    avg_fitness = method_runs['best_fitness'].mean()
                    sorted_runs = method_runs.iloc[(method_runs['best_fitness'] - avg_fitness).abs().argsort()]
                    
                    # Take the first valid history
                    for i, run in sorted_runs.iterrows():
                        if 'fitness_history' in run and isinstance(run['fitness_history'], list) and len(run['fitness_history']) > 0:
                            plt.plot(
                                range(1, len(run['fitness_history']) + 1),
                                run['fitness_history'],
                                label=f"{method}"
                            )
                            break
            
            plt.title('Convergence Analysis by Selection Method')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "convergence_by_selection.png"), dpi=300)
            plt.close()
    
    except Exception as e:
        print(f"Error generating convergence plot: {str(e)}")

def generate_latex_tables(df, output_dir):
    """
    Generate LaTeX tables for the paper
    
    Args:
        df (pd.DataFrame): DataFrame containing performance data
        output_dir (str): Directory to save the LaTeX tables
    Returns:
        None. Saves LaTeX tables in the specified directory.
    """
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join("analysis", f"latex_tables_{timestamp}")
    
    latex_dir = os.path.join(output_dir, "latex")
    os.makedirs(latex_dir, exist_ok=True)
    
    print(f"Generating LaTeX tables in: {latex_dir}")
    
    # 1. Summary by selection method
    selection_summary = df.groupby('selection_method').agg({
        'best_fitness': ['count', 'mean', 'std'],
        'execution_time': 'mean',
        'best_generation': 'mean'
    })
    
    with open(os.path.join(latex_dir, "selection_method_table.tex"), 'w') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance by Selection Method}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\hline\n")
        f.write("Selection Method & Runs & Mean Fitness & Std Dev & Avg Time (s) & Avg Gen \\\\\n")
        f.write("\\hline\n")
        
        for idx, row in selection_summary.iterrows():
            f.write(f"{idx} & ")
            f.write(f"{row[('best_fitness', 'count')]} & ")
            f.write(f"{row[('best_fitness', 'mean')]:.6f} & ")
            f.write(f"{row[('best_fitness', 'std')]:.6f} & ")
            f.write(f"{row[('execution_time', 'mean')]:.2f} & ")
            f.write(f"{row[('best_generation', 'mean')]:.1f} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    # 2. Summary by mutation rate
    mutation_summary = df.groupby('mutation_rate').agg({
        'best_fitness': ['count', 'mean', 'std'],
        'execution_time': 'mean',
        'best_generation': 'mean'
    })
    
    with open(os.path.join(latex_dir, "mutation_rate_table.tex"), 'w') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance by Mutation Rate}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\hline\n")
        f.write("Mutation Rate & Runs & Mean Fitness & Std Dev & Avg Time (s) & Avg Gen \\\\\n")
        f.write("\\hline\n")
        
        for idx, row in mutation_summary.iterrows():
            f.write(f"{idx} & ")
            f.write(f"{row[('best_fitness', 'count')]} & ")
            f.write(f"{row[('best_fitness', 'mean')]:.6f} & ")
            f.write(f"{row[('best_fitness', 'std')]:.6f} & ")
            f.write(f"{row[('execution_time', 'mean')]:.2f} & ")
            f.write(f"{row[('best_generation', 'mean')]:.1f} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    # 3. Feature comparison (KMeans, Restricted, Diverse Mutation)
    try:
        if (df['uses_kmeans'].nunique() > 1 and 
            df['restricted'].nunique() > 1 and 
            df['uses_diverse_mutation'].nunique() > 1):
            
            kmeans_summary = df.groupby('uses_kmeans').agg({
                'best_fitness': ['mean', 'std']
            })
            
            restricted_summary = df.groupby('restricted').agg({
                'best_fitness': ['mean', 'std']
            })
            
            diverse_summary = df.groupby('uses_diverse_mutation').agg({
                'best_fitness': ['mean', 'std']
            })
            
            with open(os.path.join(latex_dir, "feature_comparison_table.tex"), 'w') as f:
                f.write("\\begin{table}[ht]\n")
                f.write("\\centering\n")
                f.write("\\caption{Feature Comparison}\n")
                f.write("\\begin{tabular}{l|cc|cc|cc}\n")
                f.write("\\hline\n")
                f.write(" & \\multicolumn{2}{c|}{KMeans Init} & \\multicolumn{2}{c|}{Restricted Palette} & \\multicolumn{2}{c}{Diverse Mutation} \\\\\n")
                f.write("Option & Fitness & Std Dev & Fitness & Std Dev & Fitness & Std Dev \\\\\n")
                f.write("\\hline\n")
                
                for i in [False, True]:  # No/Yes
                    f.write(f"{'Yes' if i else 'No'} & ")
                    
                    if i in kmeans_summary.index:
                        f.write(f"{kmeans_summary.loc[i][('best_fitness', 'mean')]:.6f} & ")
                        f.write(f"{kmeans_summary.loc[i][('best_fitness', 'std')]:.6f} & ")
                    else:
                        f.write("N/A & N/A & ")
                        
                    if i in restricted_summary.index:
                        f.write(f"{restricted_summary.loc[i][('best_fitness', 'mean')]:.6f} & ")
                        f.write(f"{restricted_summary.loc[i][('best_fitness', 'std')]:.6f} & ")
                    else:
                        f.write("N/A & N/A & ")
                        
                    if i in diverse_summary.index:
                        f.write(f"{diverse_summary.loc[i][('best_fitness', 'mean')]:.6f} & ")
                        f.write(f"{diverse_summary.loc[i][('best_fitness', 'std')]:.6f} \\\\\n")
                    else:
                        f.write("N/A & N/A \\\\\n")
                
                f.write("\\hline\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n")
    except Exception as e:
        print(f"Error generating feature comparison table: {str(e)}")

def run_configuration_test(config, image_name='squareSix.jpg'):
    """
    Run a single configuration test
    This function runs the image quantization algorithm with the provided configuration.

    Args:
        config (dict): Configuration parameters for the test
        image_name (str): Name of the image to be processed
    Returns:
        best_fitness (float): Best fitness value obtained
        fitness_history (list): History of fitness values during the run
    """

    print(f"Running test with configuration: {config['name']}")
    print(f"Parameters: {config}")
    
    _, best_fitness, fitness_history, _, _, _ = run_image_quantization(
        image_name=image_name,
        **{k: v for k, v in config.items() if k != 'name'}
    )
    
    print(f"Test completed. Best fitness: {best_fitness}")
    return best_fitness, fitness_history

def run_batch_test(configurations=None, image_name='squareSix.jpg'):
    """Run tests for all configurations and collect results"""
    if configurations is None:
        configurations = CONFIGURATIONS
    
    results = []
    
    for config in configurations:
        print(f"\n=== Testing {config['name']} ===")
        best_fitness, fitness_history = run_configuration_test(config, image_name)
        
        results.append({
            'config_name': config['name'],
            'best_fitness': best_fitness,
            'fitness_history': fitness_history
        })
    
    # Print summary of results
    print("\n=== Test Results Summary ===")
    for result in sorted(results, key=lambda r: r['best_fitness'], reverse=True):
        print(f"{result['config_name']}: {result['best_fitness']:.6f}")
    
    return results

def generate_rgb_distance_comparison(output_dir=None):
    """
    Generate visual comparison between RGB distance and perceptual metrics.
    Shows original image, kmeans initialization, and final results side by side.

    Args:
        output_dir (str): Directory to save the results. If None, a timestamped directory will be created.

    Returns:
        None. Saves images and analysis results to the specified directory.
    """

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join("analysis", f"rgb_distance_comparison_{timestamp}")

    aux_output_dir = os.path.join(output_dir, "auxiliary_images")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(aux_output_dir, exist_ok=True)
    print(f"RGB distance comparison results will be saved to {output_dir}")

    test_image = 'squareFour.jpg'
    
    # Define configurations:
    rgb_config = {
        'name': 'rgb-distance',
        'num_colors': 4,
        'restricted': False,
        'population_size': 20,
        'generations': 30,
        'mutation_rate': 0.2,
        'crossover_rate': 0.8,
        'selection_method': 'tournament',
        'tournament_size': 3,
        'adaptation_rate': 1,
        'adaptation_threshold': None,
        'halting_stagnation_threshold': None,
        'kMeans': True,
        'mutate_diverse': False,
        'crossover_method': 'one_point',
        'display': False,
        'save_results': True,
        'rgb_distance': True
    }
    
    # Lab color space configuration
    perceptual_config = {
        'name': 'perceptual-distance',
        'num_colors': 4,
        'restricted': False,
        'population_size': 20,
        'generations': 30,
        'mutation_rate': 0.2,
        'crossover_rate': 0.8,
        'selection_method': 'tournament',
        'tournament_size': 3,
        'adaptation_rate': 1,
        'adaptation_threshold': None,
        'halting_stagnation_threshold': None,
        'kMeans': True,
        'mutate_diverse': False,
        'crossover_method': 'one_point',
        'save_results': True,
        'display': False,
        'rgb_distance': False
    }
    
    try:
        original_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", test_image)
        original_image = load_image(original_image_path)
        
        # Run with RGB distance
        print("\nRunning with RGB distance metric...")
        rgb_palette, rgb_fitness, rgb_history, rgb_image_result, _, rgb_kmeans_image = run_image_quantization(
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
        perceptual_palette, perceptual_fitness, perceptual_history, perceptual_img_result, _, perceptual_kmeans_image= run_image_quantization(
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
        
        # Empty subplot for layout
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

        # Save the side by side comparison
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "distance_metric_fitness_side_by_side.png"), dpi=300)
        plt.close()

        print("\nRGB distance comparison complete. Results saved to:", output_dir)
        
    except Exception as e:
        import traceback
        print(f"Error generating RGB distance comparison: {str(e)}")
        traceback.print_exc()

def run_caching_comparison_multiple_colors(image_name='squareSix.jpg'):
    """
    Runs a comparison between caching and non-caching implementations
    with multiple color counts while maintaining the rest of the configuration.
    
    Args:
        image_name (str): Name of the image to test
        color_counts (list): List of color counts to test
    """
    repetitions = REPETITIONS
    color_counts = BASIC_COLOR_CHANGES
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"analysis/caching_colors_comparison_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running caching comparison with {repetitions} repetitions for each configuration")
    print(f"Testing color counts: {color_counts}")
    print(f"Image: {image_name}")
    
    # Base configurations from CACHING_TEST_CONFIGS
    base_caching_config = next((cfg for cfg in CACHING_TEST_CONFIGS if cfg['name'] == 'caching_enabled'), None)
    base_no_caching_config = next((cfg for cfg in CACHING_TEST_CONFIGS if cfg['name'] == 'caching_disabled'), None)
    
    if not base_caching_config or not base_no_caching_config:
        print("Error: Caching comparison configurations not found")
        return
    
    all_results = {}
    
    # Run tests for each color count
    for num_colors in color_counts:
        print(f"\n{'='*60}")
        print(f"TESTING WITH {num_colors} COLORS")
        print(f"{'='*60}")
        
        # Create copies of the configurations and update the color count
        caching_config = dict(base_caching_config)
        caching_config['num_colors'] = num_colors
        
        no_caching_config = dict(base_no_caching_config)
        no_caching_config['num_colors'] = num_colors
        
        # Results for this color count
        color_results = {
            'caching': {
                'fitness': [],
                'execution_time': [],
                'fitness_history': []
            },
            'no_caching': {
                'fitness': [],
                'execution_time': [],
                'fitness_history': []
            }
        }
        
        # Run with caching for this color count
        print(f"\n=== With Caching ({num_colors} colors) ===")
        for i in range(1, repetitions + 1):
            print(f"\nRepetition {i}/{repetitions}")
            start_time = datetime.now()
            
            palette, fitness, history, result_img, _, _ = run_image_quantization(
                image_name=image_name,
                **{k: v for k, v in caching_config.items() if k != 'name'}
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            color_results['caching']['fitness'].append(fitness)
            color_results['caching']['execution_time'].append(execution_time)
            color_results['caching']['fitness_history'].append(history)
            
            print(f"Fitness: {fitness:.6f}, Time: {execution_time:.2f}s")
        
        # Run without caching for this color count
        print(f"\n=== Without Caching ({num_colors} colors) ===")
        for i in range(1, repetitions + 1):
            print(f"\nRepetition {i}/{repetitions}")
            start_time = datetime.now()
            
            palette, fitness, history, result_img, _, _ = run_image_quantization(
                image_name=image_name,
                **{k: v for k, v in no_caching_config.items() if k != 'name'}
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            color_results['no_caching']['fitness'].append(fitness)
            color_results['no_caching']['execution_time'].append(execution_time)
            color_results['no_caching']['fitness_history'].append(history)
            
            print(f"Fitness: {fitness:.6f}, Time: {execution_time:.2f}s")
        
        # Calculate and store statistics
        caching_results = color_results['caching']
        no_caching_results = color_results['no_caching']
        
        caching_avg_fitness = sum(caching_results['fitness']) / len(caching_results['fitness'])
        caching_avg_time = sum(caching_results['execution_time']) / len(caching_results['execution_time'])
        
        no_caching_avg_fitness = sum(no_caching_results['fitness']) / len(no_caching_results['fitness'])
        no_caching_avg_time = sum(no_caching_results['execution_time']) / len(no_caching_results['execution_time'])
        
        caching_best_idx = caching_results['fitness'].index(max(caching_results['fitness']))
        no_caching_best_idx = no_caching_results['fitness'].index(max(no_caching_results['fitness']))
        
        # Store summary statistics for this color count
        all_results[num_colors] = {
            'caching': {
                'results': caching_results,
                'avg_fitness': caching_avg_fitness,
                'avg_time': caching_avg_time,
                'best_idx': caching_best_idx
            },
            'no_caching': {
                'results': no_caching_results,
                'avg_fitness': no_caching_avg_fitness,
                'avg_time': no_caching_avg_time,
                'best_idx': no_caching_best_idx
            }
        }
        
        # Print summary for this color count
        print(f"\n=== Summary for {num_colors} Colors ===")
        print("\nWith Caching:")
        print(f"Average Fitness: {caching_avg_fitness:.6f}")
        print(f"Average Time: {caching_avg_time:.2f}s")
        print(f"Best Fitness: {caching_results['fitness'][caching_best_idx]:.6f}")
        
        print("\nWithout Caching:")
        print(f"Average Fitness: {no_caching_avg_fitness:.6f}")
        print(f"Average Time: {no_caching_avg_time:.2f}s")
        print(f"Best Fitness: {no_caching_results['fitness'][no_caching_best_idx]:.6f}")
        
        speedup = no_caching_avg_time / caching_avg_time if caching_avg_time > 0 else 0
        print(f"\nSpeed improvement with caching: {speedup:.2f}x")
        
    # Generate comparison visualizations across all color counts
    generate_color_comparison_visualizations(all_results, color_counts, output_dir)
    
    # Generate the combined report
    generate_color_comparison_report(all_results, color_counts, image_name, repetitions, output_dir)
    
    print(f"\nComparison complete. Results saved to {output_dir}")
    return output_dir

def generate_color_comparison_visualizations(all_results, color_counts, output_dir):
    """Generate visualizations comparing caching performance across different color counts"""
    
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. Execution time comparison across color counts
    plt.figure(figsize=(10, 6))
    x = range(len(color_counts))
    width = 0.35
    
    caching_times = [all_results[c]['caching']['avg_time'] for c in color_counts]
    no_caching_times = [all_results[c]['no_caching']['avg_time'] for c in color_counts]
    
    plt.bar([i - width/2 for i in x], caching_times, width, label='With Caching')
    plt.bar([i + width/2 for i in x], no_caching_times, width, label='Without Caching')
    
    plt.xlabel('Number of Colors')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Average Execution Time by Color Count')
    plt.xticks(x, color_counts)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(vis_dir, 'execution_time_by_colors.png'), dpi=300)
    plt.close()
    
    # 2. Speedup factor across color counts
    plt.figure(figsize=(10, 6))
    speedups = [
        all_results[c]['no_caching']['avg_time'] / all_results[c]['caching']['avg_time'] 
        if all_results[c]['caching']['avg_time'] > 0 else 0
        for c in color_counts
    ]
    
    plt.bar(color_counts, speedups)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Number of Colors')
    plt.ylabel('Speedup Factor')
    plt.title('Caching Speedup by Color Count')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(vis_dir, 'speedup_by_colors.png'), dpi=300)
    plt.close()
    
    # 3. Fitness comparison
    plt.figure(figsize=(10, 6))
    caching_fitness = [all_results[c]['caching']['avg_fitness'] for c in color_counts]
    no_caching_fitness = [all_results[c]['no_caching']['avg_fitness'] for c in color_counts]
    
    plt.bar([i - width/2 for i in x], caching_fitness, width, label='With Caching')
    plt.bar([i + width/2 for i in x], no_caching_fitness, width, label='Without Caching')
    
    plt.xlabel('Number of Colors')
    plt.ylabel('Average Fitness')
    plt.title('Average Fitness by Color Count')
    plt.xticks(x, color_counts)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(vis_dir, 'fitness_by_colors.png'), dpi=300)
    plt.close()

def generate_color_comparison_report(all_results, color_counts, image_name, repetitions, output_dir):
    """Generate a comprehensive report of the caching comparison across color counts"""
    
    with open(os.path.join(output_dir, 'color_caching_comparison_report.md'), 'w') as f:
        f.write("# Caching vs. No Caching Performance Across Color Counts\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Image:** {image_name}\n")
        f.write(f"**Repetitions per test:** {repetitions}\n\n")
        
        # Overall summary table
        f.write("## Summary Table\n\n")
        f.write("| Color Count | Metric | With Caching | Without Caching | Difference | Speedup |\n")
        f.write("|------------|--------|-------------|----------------|------------|--------|\n")
        
        for num_colors in color_counts:
            data = all_results[num_colors]
            speedup = data['no_caching']['avg_time'] / data['caching']['avg_time'] if data['caching']['avg_time'] > 0 else 0
            
            f.write(f"| **{num_colors}** | Avg. Fitness | {data['caching']['avg_fitness']:.6f} | {data['no_caching']['avg_fitness']:.6f} | {data['caching']['avg_fitness'] - data['no_caching']['avg_fitness']:.6f} | - |\n")
            f.write(f"| | Avg. Time (s) | {data['caching']['avg_time']:.2f} | {data['no_caching']['avg_time']:.2f} | {data['no_caching']['avg_time'] - data['caching']['avg_time']:.2f} | **{speedup:.2f}x** |\n")
        
        # Detailed results for each color count
        for num_colors in color_counts:
            data = all_results[num_colors]
            f.write(f"\n## Detailed Results for {num_colors} Colors\n\n")
            
            f.write("### With Caching\n\n")
            f.write("| Run | Fitness | Time (s) |\n")
            f.write("|-----|---------|----------|\n")
            for i in range(repetitions):
                f.write(f"| {i+1} | {data['caching']['results']['fitness'][i]:.6f} | {data['caching']['results']['execution_time'][i]:.2f} |\n")
            
            f.write("\n### Without Caching\n\n")
            f.write("| Run | Fitness | Time (s) |\n")
            f.write("|-----|---------|----------|\n")
            for i in range(repetitions):
                f.write(f"| {i+1} | {data['no_caching']['results']['fitness'][i]:.6f} | {data['no_caching']['results']['execution_time'][i]:.2f} |\n")
        
        # Analysis section
        f.write("\n## Analysis\n\n")
        
        # Calculate average speedup
        avg_speedup = sum([
            all_results[c]['no_caching']['avg_time'] / all_results[c]['caching']['avg_time'] 
            if all_results[c]['caching']['avg_time'] > 0 else 0
            for c in color_counts
        ]) / len(color_counts)
        
        # Find color with max speedup
        max_speedup_color = max(color_counts, key=lambda c: 
            all_results[c]['no_caching']['avg_time'] / all_results[c]['caching']['avg_time'] 
            if all_results[c]['caching']['avg_time'] > 0 else 0
        )
        
        max_speedup = all_results[max_speedup_color]['no_caching']['avg_time'] / all_results[max_speedup_color]['caching']['avg_time']
        
        f.write(f"### Effect of Color Count on Caching Performance\n\n")
        f.write(f"- **Average speedup across all color counts:** {avg_speedup:.2f}x\n")
        f.write(f"- **Maximum speedup:** {max_speedup:.2f}x (with {max_speedup_color} colors)\n")
        
        # Check if speedup increases with color count
        first_speedup = all_results[color_counts[0]]['no_caching']['avg_time'] / all_results[color_counts[0]]['caching']['avg_time']
        last_speedup = all_results[color_counts[-1]]['no_caching']['avg_time'] / all_results[color_counts[-1]]['caching']['avg_time']
        
        if last_speedup > first_speedup:
            f.write("- **Trend:** Caching provides greater benefits as color count increases\n")
        elif first_speedup > last_speedup:
            f.write("- **Trend:** Caching provides greater benefits with fewer colors\n")
        else:
            f.write("- **Trend:** Caching benefit appears to be consistent across color counts\n")
        
        # Conclusion
        f.write("\n## Conclusion\n\n")
        if avg_speedup > 1.5:
            f.write(f"Caching provides a significant performance improvement across all tested color counts, with an average speedup of {avg_speedup:.2f}x. ")
        elif avg_speedup > 1.1:
            f.write(f"Caching provides a moderate performance improvement across all tested color counts, with an average speedup of {avg_speedup:.2f}x. ")
        else:
            f.write(f"Caching provides only a slight performance improvement across all tested color counts, with an average speedup of {avg_speedup:.2f}x. ")
        
        # Impact on solution quality
        avg_fitness_diffs = [all_results[c]['caching']['avg_fitness'] - all_results[c]['no_caching']['avg_fitness'] for c in color_counts]
        avg_fitness_diff = sum(avg_fitness_diffs) / len(avg_fitness_diffs)
        
        if abs(avg_fitness_diff) < 0.001:
            f.write("The use of caching has no significant impact on solution quality.\n")
        elif avg_fitness_diff > 0:
            f.write(f"Additionally, caching slightly improves solution quality by {avg_fitness_diff:.6f} on average.\n")
        else:
            f.write(f"The use of caching slightly reduces solution quality by {-avg_fitness_diff:.6f} on average.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze genetic algorithm performance")
    parser.add_argument('--output', '-o', type=str, default=None,
                        help="Output directory for analysis results")
    parser.add_argument('--latex', '-l', action='store_true',
                        help="Generate LaTeX tables for paper")
    parser.add_argument('--run', '-r', type=str, default=None, 
                        choices=[cfg['name'] for cfg in CONFIGURATIONS] + ['all'],
                        help="Run a specific configuration or 'all' configurations")
    parser.add_argument('--image', '-i', type=str, default='squareSix.jpg',
                        help="Image to use for tests")
    parser.add_argument('--rgb-comparison', action='store_true',
                    help="Generate RGB vs perceptual distance comparison")
    parser.add_argument('--caching-comparison', action='store_true',
                    help="Run caching vs non-caching comparison")
    
    args = parser.parse_args()
    
    # Run tests if requested
    if args.run:
        if args.run == 'all':
            print("Running all configurations...")
            run_batch_test(image_name=args.image)
        else:
            # Find the requested configuration
            config_to_run = next((cfg for cfg in CONFIGURATIONS if cfg['name'] == args.run), None)
            if config_to_run:
                print(f"Running configuration: {args.run}")
                run_configuration_test(config_to_run, image_name=args.image)
            else:
                print(f"Configuration '{args.run}' not found")
    
    # Generate RGB distance comparison if requested
    elif args.rgb_comparison:
        generate_rgb_distance_comparison(args.output)
        print("RGB distance comparison completed")
    
    # Run caching comparison if requested
    elif args.caching_comparison:
        run_caching_comparison_multiple_colors(args.image)
        print("Caching comparison completed")

    else:        
        # Run the analysis
        df = collect_and_analyze_reports(args.output)
        
        if df is not None and args.latex:
            generate_latex_tables(df, args.output)
            print("LaTeX tables generated successfully")