# Genetic Algorithm Analysis Report

**Test Type:** early_stopping

**Test Image:** squareSix.jpg

**Date:** 2025-05-28 12:27:57

**Repetitions per configuration:** 3

## Performance Summary


### 4 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| no_early_stopping | 0.034747 ± 0.004543 | 5.57 ± 0.87 |
| early_stopping_10 | 0.032047 ± 0.000492 | 3.09 ± 1.19 |
| early_stopping_20 | 0.032532 ± 0.000678 | 4.94 ± 1.08 |


### 10 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| no_early_stopping | 0.386284 ± 0.015470 | 10.24 ± 1.72 |
| early_stopping_10 | 0.261686 ± 0.105699 | 7.21 ± 0.32 |
| early_stopping_20 | 0.368858 ± 0.023570 | 9.06 ± 0.73 |


### 20 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| no_early_stopping | 0.505833 ± 0.022816 | 13.63 ± 0.96 |
| early_stopping_10 | 0.480059 ± 0.040730 | 9.44 ± 0.25 |
| early_stopping_20 | 0.504403 ± 0.008142 | 13.78 ± 0.23 |


## Stagnation Analysis

| Color Count | Configuration | Stagnation Frequency | Generations with Stagnation |
|-------------|---------------|----------------------|---------------------|
| 4 | no_early_stopping | 82.00% | 123/150 |
| 4 | early_stopping_10 | 76.06% | 54/71 |
| 4 | early_stopping_20 | 79.84% | 103/129 |
| 10 | no_early_stopping | 63.33% | 95/150 |
| 10 | early_stopping_10 | 51.11% | 46/90 |
| 10 | early_stopping_20 | 67.59% | 98/145 |
| 20 | no_early_stopping | 58.00% | 87/150 |
| 20 | early_stopping_10 | 46.67% | 42/90 |
| 20 | early_stopping_20 | 58.67% | 88/150 |

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
| 4 | early_stopping_10 | 66.7% | 23.7 |
| 4 | early_stopping_20 | 66.7% | 43.0 |
| 10 | no_early_stopping | 0.0% | 50.0 |
| 10 | early_stopping_10 | 0.0% | 30.0 |
| 10 | early_stopping_20 | 33.3% | 48.3 |
| 20 | no_early_stopping | 0.0% | 50.0 |
| 20 | early_stopping_10 | 0.0% | 30.0 |
| 20 | early_stopping_20 | 0.0% | 50.0 |

Early stopping occurs when the algorithm halts due to stagnation.

## Reference Images

Best results from each configuration:


### 4 Colors

<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>no_early_stopping</strong></p>
<img src='colors_4\no_early_stopping.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.039991</p>
</div>
<div style='text-align: center;'>
<p><strong>early_stopping_20</strong></p>
<img src='colors_4\early_stopping_20.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.033307</p>
</div>
<div style='text-align: center;'>
<p><strong>early_stopping_10</strong></p>
<img src='colors_4\early_stopping_10.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.032382</p>
</div>
</div>


### 10 Colors

<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>no_early_stopping</strong></p>
<img src='colors_10\no_early_stopping.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.397911</p>
</div>
<div style='text-align: center;'>
<p><strong>early_stopping_20</strong></p>
<img src='colors_10\early_stopping_20.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.385156</p>
</div>
<div style='text-align: center;'>
<p><strong>early_stopping_10</strong></p>
<img src='colors_10\early_stopping_10.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.383427</p>
</div>
</div>


### 20 Colors

<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>early_stopping_10</strong></p>
<img src='colors_20\early_stopping_10.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.527040</p>
</div>
<div style='text-align: center;'>
<p><strong>no_early_stopping</strong></p>
<img src='colors_20\no_early_stopping.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.526733</p>
</div>
<div style='text-align: center;'>
<p><strong>early_stopping_20</strong></p>
<img src='colors_20\early_stopping_20.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.513278</p>
</div>
</div>

