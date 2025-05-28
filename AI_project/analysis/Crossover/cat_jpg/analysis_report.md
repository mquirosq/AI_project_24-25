# Genetic Algorithm Analysis Report

**Test Type:** crossover

**Test Image:** cat.jpg

**Date:** 2025-05-27 23:45:12

**Repetitions per configuration:** 3

## Performance Summary


### 4 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| one_point_crossover | 0.100918 ± 0.002239 | 30.73 ± 3.25 |
| uniform_crossover | 0.097134 ± 0.004051 | 36.32 ± 2.48 |
| closest_pairs_crossover | 0.097646 ± 0.001192 | 36.27 ± 3.95 |


### 10 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| one_point_crossover | 0.134212 ± 0.003682 | 70.00 ± 2.82 |
| uniform_crossover | 0.139243 ± 0.003003 | 84.10 ± 12.35 |
| closest_pairs_crossover | 0.142130 ± 0.006711 | 97.21 ± 18.34 |


### 20 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| one_point_crossover | 0.168715 ± 0.001685 | 110.42 ± 1.87 |
| uniform_crossover | 0.171287 ± 0.001939 | 111.30 ± 0.67 |
| closest_pairs_crossover | 0.170267 ± 0.006667 | 105.90 ± 7.34 |


## Crossover Method Analysis

| Color Count | Configuration | Method | Crossover Rate | Avg Fitness | Time (s) |
|-------------|---------------|--------|---------------|-------------|----------|
| 4 | one_point_crossover | one | 80% | 0.100918 | 30.73 |
| 4 | uniform_crossover | uniform | 80% | 0.097134 | 36.32 |
| 4 | closest_pairs_crossover | closest | 80% | 0.097646 | 36.27 |
| 10 | one_point_crossover | one | 80% | 0.134212 | 70.00 |
| 10 | uniform_crossover | uniform | 80% | 0.139243 | 84.10 |
| 10 | closest_pairs_crossover | closest | 80% | 0.142130 | 97.21 |
| 20 | one_point_crossover | one | 80% | 0.168715 | 110.42 |
| 20 | uniform_crossover | uniform | 80% | 0.171287 | 111.30 |
| 20 | closest_pairs_crossover | closest | 80% | 0.170267 | 105.90 |

## Reference Images

Best results from each configuration:


### 4 Colors

<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>one_point_crossover</strong></p>
<img src='colors_4\one_point_crossover.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.102737</p>
</div>
<div style='text-align: center;'>
<p><strong>uniform_crossover</strong></p>
<img src='colors_4\uniform_crossover.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.099842</p>
</div>
<div style='text-align: center;'>
<p><strong>closest_pairs_crossover</strong></p>
<img src='colors_4\closest_pairs_crossover.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.098992</p>
</div>
</div>


### 10 Colors

<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>closest_pairs_crossover</strong></p>
<img src='colors_10\closest_pairs_crossover.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.146308</p>
</div>
<div style='text-align: center;'>
<p><strong>uniform_crossover</strong></p>
<img src='colors_10\uniform_crossover.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.142711</p>
</div>
<div style='text-align: center;'>
<p><strong>one_point_crossover</strong></p>
<img src='colors_10\one_point_crossover.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.137893</p>
</div>
</div>


### 20 Colors

<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>closest_pairs_crossover</strong></p>
<img src='colors_20\closest_pairs_crossover.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.177483</p>
</div>
<div style='text-align: center;'>
<p><strong>uniform_crossover</strong></p>
<img src='colors_20\uniform_crossover.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.173518</p>
</div>
<div style='text-align: center;'>
<p><strong>one_point_crossover</strong></p>
<img src='colors_20\one_point_crossover.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.170241</p>
</div>
</div>

