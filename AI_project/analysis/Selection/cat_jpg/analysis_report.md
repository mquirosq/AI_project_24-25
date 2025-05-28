# Genetic Algorithm Analysis Report

**Test Type:** selection

**Test Image:** cat.jpg

**Date:** 2025-05-27 22:19:17

**Repetitions per configuration:** 3

## Performance Summary


### 4 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| roulette_selection | 0.095794 ± 0.002186 | 35.82 ± 2.64 |
| rank_selection | 0.096312 ± 0.000405 | 32.40 ± 0.73 |
| tournament_size_3 | 0.096680 ± 0.004333 | 39.70 ± 13.01 |
| tournament_size_5 | 0.099838 ± 0.001375 | 56.15 ± 4.50 |
| tournament_size_10 | 0.096354 ± 0.003011 | 52.78 ± 4.29 |


### 10 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| roulette_selection | 0.128099 ± 0.001204 | 103.14 ± 25.68 |
| rank_selection | 0.130551 ± 0.005184 | 113.31 ± 1.57 |
| tournament_size_3 | 0.134687 ± 0.002693 | 96.25 ± 10.21 |
| tournament_size_5 | 0.137583 ± 0.003685 | 112.48 ± 10.74 |
| tournament_size_10 | 0.140297 ± 0.004650 | 107.15 ± 11.18 |


### 20 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| roulette_selection | 0.154862 ± 0.002783 | 155.00 ± 19.32 |
| rank_selection | 0.162454 ± 0.003146 | 169.43 ± 0.36 |
| tournament_size_3 | 0.163294 ± 0.002110 | 172.51 ± 5.66 |
| tournament_size_5 | 0.166556 ± 0.003497 | 179.37 ± 7.20 |
| tournament_size_10 | 0.171505 ± 0.002830 | 151.04 ± 26.04 |


## Reference Images

Best results from each configuration:


### 4 Colors

<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>tournament_size_3</strong></p>
<img src='colors_4\tournament_size_3.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.101549</p>
</div>
<div style='text-align: center;'>
<p><strong>tournament_size_5</strong></p>
<img src='colors_4\tournament_size_5.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.101007</p>
</div>
<div style='text-align: center;'>
<p><strong>tournament_size_10</strong></p>
<img src='colors_4\tournament_size_10.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.098334</p>
</div>
<div style='text-align: center;'>
<p><strong>roulette_selection</strong></p>
<img src='colors_4\roulette_selection.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.097920</p>
</div>
<div style='text-align: center;'>
<p><strong>rank_selection</strong></p>
<img src='colors_4\rank_selection.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.096775</p>
</div>
</div>


### 10 Colors

<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>tournament_size_10</strong></p>
<img src='colors_10\tournament_size_10.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.143990</p>
</div>
<div style='text-align: center;'>
<p><strong>tournament_size_5</strong></p>
<img src='colors_10\tournament_size_5.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.140733</p>
</div>
<div style='text-align: center;'>
<p><strong>tournament_size_3</strong></p>
<img src='colors_10\tournament_size_3.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.137566</p>
</div>
<div style='text-align: center;'>
<p><strong>rank_selection</strong></p>
<img src='colors_10\rank_selection.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.135427</p>
</div>
<div style='text-align: center;'>
<p><strong>roulette_selection</strong></p>
<img src='colors_10\roulette_selection.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.128980</p>
</div>
</div>


### 20 Colors

<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>tournament_size_10</strong></p>
<img src='colors_20\tournament_size_10.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.174733</p>
</div>
<div style='text-align: center;'>
<p><strong>tournament_size_5</strong></p>
<img src='colors_20\tournament_size_5.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.170235</p>
</div>
<div style='text-align: center;'>
<p><strong>tournament_size_3</strong></p>
<img src='colors_20\tournament_size_3.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.164735</p>
</div>
<div style='text-align: center;'>
<p><strong>rank_selection</strong></p>
<img src='colors_20\rank_selection.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.164521</p>
</div>
<div style='text-align: center;'>
<p><strong>roulette_selection</strong></p>
<img src='colors_20\roulette_selection.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.157949</p>
</div>
</div>

