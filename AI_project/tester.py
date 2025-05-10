import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
from run_algorithm import run_image_quantization
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
    

"""
Configuration for image quantization experiments
"""

# Parameter configurations to test
CONFIGURATIONS = [
    # Baseline configuration
    {
        'name': 'baseline-unrestricted',
        'num_colors': 8,
        'restricted': False,
        'population_size': 20,
        'generations': 30,
        'mutation_rate': 0.2,
        'crossover_rate': 0.8,
        'selection_method': 'tournament',
        'adaptation_rate': 1,
        'adaptation_threshold': None,
        'halting_stagnation_threshold': None,
        'kMeans': False,
        'mutate_diverse': False,
        'crossover_method': 'one_point',
        'display': False
    },
    # Enhanced selection method
    {
        'name': 'rank_selection',
        'num_colors': 8,
        'restricted': False,
        'population_size': 30,
        'generations': 30,
        'mutation_rate': 0.2,
        'crossover_rate': 0.8,
        'selection_method': 'rank',  # Changed from baseline
        'adaptation_rate': 1.2,
        'adaptation_threshold': 10,
        'halting_stagnation_threshold': 20,
        'kMeans': False,
        'mutate_diverse': False,
        'crossover_method': 'one_point',
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

# Color palette sizes to test
COLOR_PALETTE_SIZES = [4, 16, 32]

"""
Experiment analyzer for image quantization genetic algorithm
"""

def extract_data_from_performance_report(file_path):
    """Extract key metrics from a performance report file"""
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
    """Find all performance report files"""
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "tests")
    
    # Look for performance report files recursively
    report_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("performance_report_") and file.endswith(".txt"):
                report_files.append(os.path.join(root, file))
    
    print(f"Found {len(report_files)} performance report files")
    return report_files

def collect_and_analyze_reports(output_dir=None):
    """Collect and analyze all performance reports"""
    # Create output directory for analysis
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("analysis", f"report_analysis_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Analysis results will be saved to {output_dir}")
    
    # Find all report files
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
    
    # Generate summary statistics
    generate_summary_statistics(df, output_dir)
    
    # Generate visualizations
    generate_visualizations(df, output_dir)
    
    print(f"Analysis complete! Results saved to {output_dir}")
    return df

def generate_summary_statistics(df, output_dir):
    """Generate summary statistics from the performance data"""
    
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
        
        # Use .iterrows() instead of looking up by index
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
        
        # Find the best configuration - safer method
        best_config_idx = None
        best_fitness = -float('inf')
        
        for idx, row in config_summary.iterrows():
            if row[('best_fitness', 'mean')] > best_fitness:
                best_fitness = row[('best_fitness', 'mean')]
                best_config_idx = idx
                
        if best_config_idx is not None:
            best_config = config_summary.loc[best_config_idx]
            
            f.write("\n## Best Configuration\n\n")
            f.write(f"**Configuration Group:** {best_config_idx}\n\n")
            f.write(f"- **Average Fitness:** {best_config[('best_fitness', 'mean')]:.6f}\n")
            f.write(f"- **Number of Runs:** {best_config[('best_fitness', 'count')]}\n")
            f.write(f"- **Average Execution Time:** {best_config[('execution_time', 'mean')]:.2f} seconds\n")
            f.write(f"- **Average Best Generation:** {best_config[('best_generation', 'mean')]:.1f}\n\n")
            
            # Parse the configuration group
            config_parts = best_config_idx.split('_')
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
    """Generate visualizations from the performance data"""
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set the style for plots
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
        # Only attempt if we have fitness history data
        if 'fitness_history' in df.columns and any(len(history) > 0 for history in df['fitness_history'] if history is not None):
            # Filter for unique selection methods
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
    """Generate LaTeX tables for the paper"""
    latex_dir = os.path.join(output_dir, "latex")
    os.makedirs(latex_dir, exist_ok=True)
    
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
            f.write(f"{row['execution_time']:.2f} & ")
            f.write(f"{row['best_generation']:.1f} \\\\\n")
        
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
            f.write(f"{row['execution_time']:.2f} & ")
            f.write(f"{row['best_generation']:.1f} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    # 3. Feature comparison (KMeans, Restricted, Diverse Mutation)
    try:
        # Only create table if we have data for all features
        if (df['uses_kmeans'].nunique() > 1 and 
            df['restricted'].nunique() > 1 and 
            df['uses_diverse_mutation'].nunique() > 1):
            
            # KMeans comparison
            kmeans_summary = df.groupby('uses_kmeans').agg({
                'best_fitness': ['mean', 'std']
            })
            
            # Restricted comparison
            restricted_summary = df.groupby('restricted').agg({
                'best_fitness': ['mean', 'std']
            })
            
            # Diverse Mutation comparison
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
                    
                    # KMeans values
                    if i in kmeans_summary.index:
                        f.write(f"{kmeans_summary.loc[i][('best_fitness', 'mean')]:.6f} & ")
                        f.write(f"{kmeans_summary.loc[i][('best_fitness', 'std')]:.6f} & ")
                    else:
                        f.write("N/A & N/A & ")
                        
                    # Restricted values
                    if i in restricted_summary.index:
                        f.write(f"{restricted_summary.loc[i][('best_fitness', 'mean')]:.6f} & ")
                        f.write(f"{restricted_summary.loc[i][('best_fitness', 'std')]:.6f} & ")
                    else:
                        f.write("N/A & N/A & ")
                        
                    # Diverse mutation values
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

def run_configuration_test(config, image_name='squareFour.jpg'):
    """Run a single configuration test"""

    print(f"Running test with configuration: {config['name']}")
    print(f"Parameters: {config}")
    
    # Run the algorithm with this configuration
    _, best_fitness, fitness_history, _, _, kMeans_image = run_image_quantization(
        image_name=image_name,
        **{k: v for k, v in config.items() if k != 'name'}
    )
    
    print(f"Test completed. Best fitness: {best_fitness}")
    return best_fitness, fitness_history

def run_batch_test(configurations=None, image_name='squareFour.jpg'):
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
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join("analysis", f"rgb_distance_comparison_{timestamp}")

    aux_output_dir = os.path.join(output_dir, "auxiliary_images")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(aux_output_dir, exist_ok=True)
    print(f"RGB distance comparison results will be saved to {output_dir}")

    test_image = 'squareFour.jpg'  # Image with clear color differences
    
    # Define configurations: one with RGB distance and one with perceptual distance
    rgb_config = {
        'name': 'rgb-distance',
        'num_colors': 4,
        'restricted': False,
        'population_size': 20,
        'generations': 30,
        'mutation_rate': 0.2,
        'crossover_rate': 0.8,
        'selection_method': 'tournament',
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
    
    perceptual_config = {
        'name': 'perceptual-distance',
        'num_colors': 4,
        'restricted': False,
        'population_size': 20,
        'generations': 30,
        'mutation_rate': 0.2,
        'crossover_rate': 0.8,
        'selection_method': 'tournament',
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
        # Load original image
        original_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", test_image)
        original_img = np.array(Image.open(original_img_path))
        
        # Run with RGB distance
        print("\nRunning with RGB distance metric...")
        rgb_palette, rgb_fitness, rgb_history, rgb_img_result, _, rgb_kmeans_image = run_image_quantization(
            image_name=test_image,
            **{k: v for k, v in rgb_config.items() if k != 'name'}
        )
        
        # Save RGB result image
        rgb_img_path = os.path.join(aux_output_dir, "rgb_result.png")
        Image.fromarray(rgb_img_result).save(rgb_img_path)
        
        # Save RGB kmeans initialization image
        rgb_kmeans_path = os.path.join(aux_output_dir, "rgb_kmeans_init.png")
        Image.fromarray(rgb_kmeans_image).save(rgb_kmeans_path)

        # Run with perceptual distance
        print("\nRunning with perceptual distance metric...")
        perceptual_palette, perceptual_fitness, perceptual_history, perceptual_img_result, _, perceptual_kmeans_image= run_image_quantization(
            image_name=test_image,
            **{k: v for k, v in perceptual_config.items() if k != 'name'}
        )
        
        # Save perceptual result image
        perceptual_img_path = os.path.join(aux_output_dir, "perceptual_result.png")
        Image.fromarray(perceptual_img_result).save(perceptual_img_path)
        
        # Save perceptual kmeans initialization image
        perceptual_kmeans_path = os.path.join(aux_output_dir, "perceptual_kmeans_init.png")
        Image.fromarray(perceptual_kmeans_image).save(perceptual_kmeans_path)

        # Create visualization of the results
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(original_img)
        plt.title("Original Image")
        plt.axis('off')
        
        # RGB distance results
        plt.subplot(2, 3, 2)
        plt.title("RGB Distance - KMeans Initialization")
        plt.imshow(rgb_kmeans_image)
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(rgb_img_result)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze genetic algorithm performance")
    parser.add_argument('--output', '-o', type=str, default=None,
                        help="Output directory for analysis results")
    parser.add_argument('--latex', '-l', action='store_true',
                        help="Generate LaTeX tables for paper")
    parser.add_argument('--run', '-r', type=str, default=None, 
                        choices=[cfg['name'] for cfg in CONFIGURATIONS] + ['all'],
                        help="Run a specific configuration or 'all' configurations")
    parser.add_argument('--image', '-i', type=str, default='squareFour.jpg',
                        help="Image to use for tests")
    parser.add_argument('--rgb-comparison', action='store_true',
                    help="Generate RGB vs perceptual distance comparison")
    
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
    elif args.rgb_comparison:
        generate_rgb_distance_comparison(args.output)
        print("RGB distance comparison completed")
    else:
        # Run the analysis
        df = collect_and_analyze_reports(args.output)
        
        if df is not None and args.latex:
            generate_latex_tables(df, args.output)
            print("LaTeX tables generated successfully")