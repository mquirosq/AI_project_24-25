# Genetic Algorithm Analysis Report

**Test Type:** caching

**Test Image:** houses.jpg

**Date:** 2025-05-27 18:19:58

**Repetitions per configuration:** 3

## Performance Summary


### 4 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| caching_enabled | 0.054956 ± 0.001070 | 110.13 ± 3.56 |
| caching_disabled | 0.054828 ± 0.001894 | 185.74 ± 4.57 |


### 10 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| caching_enabled | 0.079221 ± 0.001477 | 207.40 ± 7.07 |
| caching_disabled | 0.078283 ± 0.000840 | 239.22 ± 2.06 |


### 20 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| caching_enabled | 0.093664 ± 0.001180 | 304.02 ± 10.90 |
| caching_disabled | 0.091647 ± 0.000428 | 342.67 ± 4.28 |


## Cache Performance Analysis

| Color Count | Configuration | Cache Hit Ratio | Total Hits | Total Misses |
|-------------|---------------|-----------------|------------|-------------|
| 4 | caching_enabled | 42.20% | 633 | 867 |
| 4 | caching_disabled | 0.00% | 0 | 0 |
| 10 | caching_enabled | 16.53% | 248 | 1252 |
| 10 | caching_disabled | 0.00% | 0 | 0 |
| 20 | caching_enabled | 10.07% | 151 | 1349 |
| 20 | caching_disabled | 0.00% | 0 | 0 |

Higher cache hit ratios indicate more efficient algorithm execution.

## Reference Images

Best results from each configuration:


### 4 Colors

<div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>caching_disabled</strong></p>
<img src='colors_4\caching_disabled.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.056021</p>
</div>
<div style='text-align: center;'>
<p><strong>caching_enabled</strong></p>
<img src='colors_4\caching_enabled.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.055681</p>
</div>
</div>


### 10 Colors

<div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>caching_enabled</strong></p>
<img src='colors_10\caching_enabled.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.080834</p>
</div>
<div style='text-align: center;'>
<p><strong>caching_disabled</strong></p>
<img src='colors_10\caching_disabled.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.079000</p>
</div>
</div>


### 20 Colors

<div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>caching_enabled</strong></p>
<img src='colors_20\caching_enabled.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.094990</p>
</div>
<div style='text-align: center;'>
<p><strong>caching_disabled</strong></p>
<img src='colors_20\caching_disabled.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.092089</p>
</div>
</div>

