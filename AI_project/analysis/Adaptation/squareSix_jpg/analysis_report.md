# Genetic Algorithm Analysis Report

**Test Type:** adaptation

**Test Image:** squareSix.jpg

**Date:** 2025-05-28 10:17:00

**Repetitions per configuration:** 3

## Performance Summary


### 4 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| no_adaptation | 0.039396 ± 0.001740 | 5.91 ± 0.55 |
| low_adaptation | 0.039822 ± 0.004111 | 6.03 ± 0.18 |
| high_adaptation | 0.033225 ± 0.001620 | 5.56 ± 0.30 |


### 10 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| no_adaptation | 0.367060 ± 0.010948 | 19.22 ± 5.36 |
| low_adaptation | 0.380217 ± 0.030282 | 22.55 ± 2.10 |
| high_adaptation | 0.326901 ± 0.102721 | 21.14 ± 2.85 |


### 20 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| no_adaptation | 0.519008 ± 0.025086 | 27.81 ± 3.57 |
| low_adaptation | 0.493571 ± 0.033733 | 25.99 ± 1.96 |
| high_adaptation | 0.488696 ± 0.033731 | 27.42 ± 3.20 |


## Stagnation Analysis

| Color Count | Configuration | Stagnation Frequency | Generations with Stagnation |
|-------------|---------------|----------------------|---------------------|
| 4 | no_adaptation | 80.00% | 120/150 |
| 4 | low_adaptation | 79.33% | 119/150 |
| 4 | high_adaptation | 82.67% | 124/150 |
| 10 | no_adaptation | 64.67% | 97/150 |
| 10 | low_adaptation | 58.67% | 88/150 |
| 10 | high_adaptation | 60.00% | 90/150 |
| 20 | no_adaptation | 62.67% | 94/150 |
| 20 | low_adaptation | 66.67% | 100/150 |
| 20 | high_adaptation | 62.00% | 93/150 |

## Mutation Rate Adaptation Analysis

| Color Count | Configuration | Min Rate | Max Rate | Average Change |
|-------------|---------------|----------|----------|---------------|
| 4 | no_adaptation | 0.200 | 0.200 | 0.000 |
| 4 | low_adaptation | 0.200 | 0.308 | 0.108 |
| 4 | high_adaptation | 0.200 | 0.400 | 0.200 |
| 10 | no_adaptation | 0.200 | 0.200 | 0.000 |
| 10 | low_adaptation | 0.200 | 0.242 | 0.042 |
| 10 | high_adaptation | 0.200 | 0.329 | 0.129 |
| 20 | no_adaptation | 0.200 | 0.200 | 0.000 |
| 20 | low_adaptation | 0.200 | 0.267 | 0.067 |
| 20 | high_adaptation | 0.200 | 0.300 | 0.100 |

Mutation rate adaptation dynamically adjusts mutation during execution based on improvement.

## Early Stopping Analysis

| Color Count | Configuration | Early Stopping % | Avg Generations |
|-------------|---------------|------------------|----------------|
| 4 | no_adaptation | 0.0% | 50.0 |
| 4 | low_adaptation | 0.0% | 50.0 |
| 4 | high_adaptation | 0.0% | 50.0 |
| 10 | no_adaptation | 0.0% | 50.0 |
| 10 | low_adaptation | 0.0% | 50.0 |
| 10 | high_adaptation | 0.0% | 50.0 |
| 20 | no_adaptation | 0.0% | 50.0 |
| 20 | low_adaptation | 0.0% | 50.0 |
| 20 | high_adaptation | 0.0% | 50.0 |

Early stopping occurs when the algorithm halts due to stagnation.

## Reference Images

Best results from each configuration:


### 4 Colors

<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>low_adaptation</strong></p>
<img src='colors_4\low_adaptation.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.042237</p>
</div>
<div style='text-align: center;'>
<p><strong>no_adaptation</strong></p>
<img src='colors_4\no_adaptation.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.041277</p>
</div>
<div style='text-align: center;'>
<p><strong>high_adaptation</strong></p>
<img src='colors_4\high_adaptation.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.035096</p>
</div>
</div>


### 10 Colors

<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>low_adaptation</strong></p>
<img src='colors_10\low_adaptation.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.403106</p>
</div>
<div style='text-align: center;'>
<p><strong>high_adaptation</strong></p>
<img src='colors_10\high_adaptation.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.397748</p>
</div>
<div style='text-align: center;'>
<p><strong>no_adaptation</strong></p>
<img src='colors_10\no_adaptation.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.378777</p>
</div>
</div>


### 20 Colors

<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>no_adaptation</strong></p>
<img src='colors_20\no_adaptation.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.544595</p>
</div>
<div style='text-align: center;'>
<p><strong>low_adaptation</strong></p>
<img src='colors_20\low_adaptation.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.532360</p>
</div>
<div style='text-align: center;'>
<p><strong>high_adaptation</strong></p>
<img src='colors_20\high_adaptation.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.525010</p>
</div>
</div>

