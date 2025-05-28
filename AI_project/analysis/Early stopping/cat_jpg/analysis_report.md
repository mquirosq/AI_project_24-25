# Genetic Algorithm Analysis Report

**Test Type:** early_stopping

**Test Image:** cat.jpg

**Date:** 2025-05-28 13:11:35

**Repetitions per configuration:** 3

## Performance Summary


### 4 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| no_early_stopping | 0.100521 ± 0.003561 | 43.11 ± 0.92 |
| early_stopping_10 | 0.100021 ± 0.000901 | 19.12 ± 5.49 |
| early_stopping_20 | 0.101238 ± 0.002543 | 35.50 ± 8.65 |


### 10 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| no_early_stopping | 0.140687 ± 0.004776 | 79.65 ± 0.09 |
| early_stopping_10 | 0.140243 ± 0.004156 | 41.66 ± 10.15 |
| early_stopping_20 | 0.139502 ± 0.002819 | 80.93 ± 0.76 |


### 20 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| no_early_stopping | 0.170484 ± 0.007315 | 188.17 ± 68.13 |
| early_stopping_10 | 0.163386 ± 0.002371 | 130.49 ± 26.69 |
| early_stopping_20 | 0.174080 ± 0.003766 | 250.33 ± 31.89 |


## Stagnation Analysis

| Color Count | Configuration | Stagnation Frequency | Generations with Stagnation |
|-------------|---------------|----------------------|---------------------|
| 4 | no_early_stopping | 81.33% | 122/150 |
| 4 | early_stopping_10 | 69.23% | 45/65 |
| 4 | early_stopping_20 | 78.51% | 95/121 |
| 10 | no_early_stopping | 77.33% | 116/150 |
| 10 | early_stopping_10 | 58.11% | 43/74 |
| 10 | early_stopping_20 | 70.00% | 105/150 |
| 20 | no_early_stopping | 72.00% | 108/150 |
| 20 | early_stopping_10 | 64.10% | 50/78 |
| 20 | early_stopping_20 | 68.67% | 103/150 |

## Mutation Rate Adaptation Analysis

| Color Count | Configuration | Min Rate | Max Rate | Average Change |
|-------------|---------------|----------|----------|---------------|
| 4 | no_early_stopping | 0.200 | 0.200 | 0.000 |
| 4 | early_stopping_10 | 0.200 | 0.200 | 0.000 |
| 4 | early_stopping_20 | 0.200 | 0.200 | 0.000 |
| 10 | no_early_stopping | 0.200 | 0.200 | 0.000 |
| 10 | early_stopping_10 | 0.200 | 0.200 | 0.000 |
| 10 | early_stopping_20 | 0.200 | 0.200 | 0.000 |
| 20 | no_early_stopping | 0.200 | 0.200 | 0.000 |
| 20 | early_stopping_10 | 0.200 | 0.200 | 0.000 |
| 20 | early_stopping_20 | 0.200 | 0.200 | 0.000 |

Mutation rate adaptation dynamically adjusts mutation during execution based on improvement.

## Early Stopping Analysis

| Color Count | Configuration | Early Stopping % | Avg Generations |
|-------------|---------------|------------------|----------------|
| 4 | no_early_stopping | 0.0% | 50.0 |
| 4 | early_stopping_10 | 66.7% | 21.7 |
| 4 | early_stopping_20 | 66.7% | 40.3 |
| 10 | no_early_stopping | 0.0% | 50.0 |
| 10 | early_stopping_10 | 66.7% | 24.7 |
| 10 | early_stopping_20 | 0.0% | 50.0 |
| 20 | no_early_stopping | 0.0% | 50.0 |
| 20 | early_stopping_10 | 33.3% | 26.0 |
| 20 | early_stopping_20 | 0.0% | 50.0 |

Early stopping occurs when the algorithm halts due to stagnation.

## Reference Images

Best results from each configuration:


### 4 Colors

<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>early_stopping_20</strong></p>
<img src='colors_4\early_stopping_20.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.103712</p>
</div>
<div style='text-align: center;'>
<p><strong>no_early_stopping</strong></p>
<img src='colors_4\no_early_stopping.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.102781</p>
</div>
<div style='text-align: center;'>
<p><strong>early_stopping_10</strong></p>
<img src='colors_4\early_stopping_10.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.100753</p>
</div>
</div>


### 10 Colors

<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>no_early_stopping</strong></p>
<img src='colors_10\no_early_stopping.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.146077</p>
</div>
<div style='text-align: center;'>
<p><strong>early_stopping_10</strong></p>
<img src='colors_10\early_stopping_10.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.144504</p>
</div>
<div style='text-align: center;'>
<p><strong>early_stopping_20</strong></p>
<img src='colors_10\early_stopping_20.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.142735</p>
</div>
</div>


### 20 Colors

<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>no_early_stopping</strong></p>
<img src='colors_20\no_early_stopping.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.178863</p>
</div>
<div style='text-align: center;'>
<p><strong>early_stopping_20</strong></p>
<img src='colors_20\early_stopping_20.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.177515</p>
</div>
<div style='text-align: center;'>
<p><strong>early_stopping_10</strong></p>
<img src='colors_20\early_stopping_10.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.165195</p>
</div>
</div>

