# Genetic Algorithm Analysis Report

**Test Type:** caching

**Test Image:** squareSix.jpg

**Date:** 2025-05-27 17:10:24

**Repetitions per configuration:** 3

## Performance Summary


### 4 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| caching_enabled | 0.037790 ± 0.004787 | 6.14 ± 0.63 |
| caching_disabled | 0.037726 ± 0.004878 | 6.33 ± 0.58 |


### 10 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| caching_enabled | 0.358901 ± 0.015475 | 8.45 ± 0.76 |
| caching_disabled | 0.351079 ± 0.026894 | 9.37 ± 0.53 |


### 20 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| caching_enabled | 0.493037 ± 0.019515 | 12.39 ± 0.03 |
| caching_disabled | 0.446487 ± 0.011110 | 12.88 ± 0.58 |


## Cache Performance Analysis

| Color Count | Configuration | Cache Hit Ratio | Total Hits | Total Misses |
|-------------|---------------|-----------------|------------|-------------|
| 4 | caching_enabled | 43.60% | 654 | 846 |
| 4 | caching_disabled | 0.00% | 0 | 0 |
| 10 | caching_enabled | 15.87% | 238 | 1262 |
| 10 | caching_disabled | 0.00% | 0 | 0 |
| 20 | caching_enabled | 10.47% | 157 | 1343 |
| 20 | caching_disabled | 0.00% | 0 | 0 |

Higher cache hit ratios indicate more efficient algorithm execution.

## Reference Images

Best results from each configuration:


### 4 Colors

<div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>caching_enabled</strong></p>
<img src='colors_4\caching_enabled.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.043227</p>
</div>
<div style='text-align: center;'>
<p><strong>caching_disabled</strong></p>
<img src='colors_4\caching_disabled.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.041678</p>
</div>
</div>


### 10 Colors

<div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>caching_enabled</strong></p>
<img src='colors_10\caching_enabled.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.376270</p>
</div>
<div style='text-align: center;'>
<p><strong>caching_disabled</strong></p>
<img src='colors_10\caching_disabled.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.367611</p>
</div>
</div>


### 20 Colors

<div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>caching_enabled</strong></p>
<img src='colors_20\caching_enabled.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.515339</p>
</div>
<div style='text-align: center;'>
<p><strong>caching_disabled</strong></p>
<img src='colors_20\caching_disabled.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.458662</p>
</div>
</div>

