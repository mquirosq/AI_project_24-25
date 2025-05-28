# Genetic Algorithm Analysis Report

**Test Type:** crossover

**Test Image:** squareSix.jpg

**Date:** 2025-05-27 23:10:56

**Repetitions per configuration:** 3

## Performance Summary


### 4 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| one_point_crossover | 0.033055 ± 0.001808 | 4.00 ± 0.11 |
| uniform_crossover | 0.035987 ± 0.004735 | 4.97 ± 0.47 |
| closest_pairs_crossover | 0.033799 ± 0.003010 | 5.37 ± 1.51 |


### 10 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| one_point_crossover | 0.365206 ± 0.005593 | 9.04 ± 1.39 |
| uniform_crossover | 0.277444 ± 0.148195 | 9.59 ± 1.79 |
| closest_pairs_crossover | 0.352105 ± 0.019492 | 9.79 ± 1.13 |


### 20 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| one_point_crossover | 0.498883 ± 0.020146 | 10.48 ± 3.04 |
| uniform_crossover | 0.470856 ± 0.010013 | 8.57 ± 0.46 |
| closest_pairs_crossover | 0.476987 ± 0.020977 | 9.23 ± 1.28 |


## Crossover Method Analysis

| Color Count | Configuration | Method | Crossover Rate | Avg Fitness | Time (s) |
|-------------|---------------|--------|---------------|-------------|----------|
| 4 | one_point_crossover | one | 80% | 0.033055 | 4.00 |
| 4 | uniform_crossover | uniform | 80% | 0.035987 | 4.97 |
| 4 | closest_pairs_crossover | closest | 80% | 0.033799 | 5.37 |
| 10 | one_point_crossover | one | 80% | 0.365206 | 9.04 |
| 10 | uniform_crossover | uniform | 80% | 0.277444 | 9.59 |
| 10 | closest_pairs_crossover | closest | 80% | 0.352105 | 9.79 |
| 20 | one_point_crossover | one | 80% | 0.498883 | 10.48 |
| 20 | uniform_crossover | uniform | 80% | 0.470856 | 8.57 |
| 20 | closest_pairs_crossover | closest | 80% | 0.476987 | 9.23 |

## Reference Images

Best results from each configuration:


### 4 Colors

<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>uniform_crossover</strong></p>
<img src='colors_4\uniform_crossover.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.041311</p>
</div>
<div style='text-align: center;'>
<p><strong>closest_pairs_crossover</strong></p>
<img src='colors_4\closest_pairs_crossover.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.037273</p>
</div>
<div style='text-align: center;'>
<p><strong>one_point_crossover</strong></p>
<img src='colors_4\one_point_crossover.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.034560</p>
</div>
</div>


### 10 Colors

<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>one_point_crossover</strong></p>
<img src='colors_10\one_point_crossover.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.370189</p>
</div>
<div style='text-align: center;'>
<p><strong>uniform_crossover</strong></p>
<img src='colors_10\uniform_crossover.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.369657</p>
</div>
<div style='text-align: center;'>
<p><strong>closest_pairs_crossover</strong></p>
<img src='colors_10\closest_pairs_crossover.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.365419</p>
</div>
</div>


### 20 Colors

<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>one_point_crossover</strong></p>
<img src='colors_20\one_point_crossover.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.517043</p>
</div>
<div style='text-align: center;'>
<p><strong>closest_pairs_crossover</strong></p>
<img src='colors_20\closest_pairs_crossover.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.499021</p>
</div>
<div style='text-align: center;'>
<p><strong>uniform_crossover</strong></p>
<img src='colors_20\uniform_crossover.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.481541</p>
</div>
</div>

