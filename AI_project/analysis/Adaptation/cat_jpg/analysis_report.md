# Genetic Algorithm Analysis Report

**Test Type:** adaptation

**Test Image:** cat.jpg

**Date:** 2025-05-28 11:09:05

**Repetitions per configuration:** 3

## Performance Summary


### 4 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| no_adaptation | 0.100403 ± 0.003042 | 91.11 ± 0.81 |
| low_adaptation | 0.101019 ± 0.002310 | 82.98 ± 14.57 |
| high_adaptation | 0.100617 ± 0.003186 | 89.81 ± 2.48 |


### 10 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| no_adaptation | 0.143255 ± 0.002804 | 116.98 ± 20.62 |
| low_adaptation | 0.137914 ± 0.001432 | 87.25 ± 2.10 |
| high_adaptation | 0.142921 ± 0.000760 | 100.56 ± 10.23 |


### 20 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| no_adaptation | 0.174398 ± 0.003713 | 232.13 ± 33.29 |
| low_adaptation | 0.175504 ± 0.005665 | 118.76 ± 1.26 |
| high_adaptation | 0.174836 ± 0.001632 | 119.38 ± 2.13 |


## Stagnation Analysis

| Color Count | Configuration | Stagnation Frequency | Generations with Stagnation |
|-------------|---------------|----------------------|---------------------|
| 4 | no_adaptation | 82.00% | 123/150 |
| 4 | low_adaptation | 80.00% | 120/150 |
| 4 | high_adaptation | 84.67% | 127/150 |
| 10 | no_adaptation | 75.33% | 113/150 |
| 10 | low_adaptation | 76.00% | 114/150 |
| 10 | high_adaptation | 68.00% | 102/150 |
| 20 | no_adaptation | 66.00% | 99/150 |
| 20 | low_adaptation | 72.00% | 108/150 |
| 20 | high_adaptation | 72.00% | 108/150 |

## Mutation Rate Adaptation Analysis

| Color Count | Configuration | Min Rate | Max Rate | Average Change |
|-------------|---------------|----------|----------|---------------|
| 4 | no_adaptation | 0.200 | 0.200 | 0.000 |
| 4 | low_adaptation | 0.200 | 0.339 | 0.139 |
| 4 | high_adaptation | 0.200 | 0.499 | 0.299 |
| 10 | no_adaptation | 0.200 | 0.200 | 0.000 |
| 10 | low_adaptation | 0.200 | 0.419 | 0.219 |
| 10 | high_adaptation | 0.200 | 0.299 | 0.099 |
| 20 | no_adaptation | 0.200 | 0.200 | 0.000 |
| 20 | low_adaptation | 0.200 | 0.346 | 0.146 |
| 20 | high_adaptation | 0.200 | 0.372 | 0.172 |

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
<p><strong>high_adaptation</strong></p>
<img src='colors_4\high_adaptation.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.104136</p>
</div>
<div style='text-align: center;'>
<p><strong>no_adaptation</strong></p>
<img src='colors_4\no_adaptation.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.103671</p>
</div>
<div style='text-align: center;'>
<p><strong>low_adaptation</strong></p>
<img src='colors_4\low_adaptation.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.103590</p>
</div>
</div>


### 10 Colors

<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>no_adaptation</strong></p>
<img src='colors_10\no_adaptation.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.146468</p>
</div>
<div style='text-align: center;'>
<p><strong>high_adaptation</strong></p>
<img src='colors_10\high_adaptation.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.143750</p>
</div>
<div style='text-align: center;'>
<p><strong>low_adaptation</strong></p>
<img src='colors_10\low_adaptation.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.139394</p>
</div>
</div>


### 20 Colors

<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>low_adaptation</strong></p>
<img src='colors_20\low_adaptation.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.180530</p>
</div>
<div style='text-align: center;'>
<p><strong>no_adaptation</strong></p>
<img src='colors_20\no_adaptation.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.176981</p>
</div>
<div style='text-align: center;'>
<p><strong>high_adaptation</strong></p>
<img src='colors_20\high_adaptation.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.176721</p>
</div>
</div>

