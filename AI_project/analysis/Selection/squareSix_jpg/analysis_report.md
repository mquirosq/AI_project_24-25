# Genetic Algorithm Analysis Report

**Test Type:** selection

**Test Image:** squareSix.jpg

**Date:** 2025-05-27 21:00:15

**Repetitions per configuration:** 3

## Performance Summary


### 4 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| roulette_selection | 0.034154 ± 0.006576 | 4.48 ± 0.84 |
| rank_selection | 0.035145 ± 0.004770 | 6.46 ± 2.77 |
| tournament_size_3 | 0.038164 ± 0.007440 | 9.14 ± 1.16 |
| tournament_size_5 | 0.029919 ± 0.002075 | 8.34 ± 0.64 |
| tournament_size_10 | 0.033783 ± 0.003618 | 6.94 ± 1.42 |


### 10 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| roulette_selection | 0.316216 ± 0.012421 | 11.88 ± 1.45 |
| rank_selection | 0.288531 ± 0.087019 | 10.74 ± 1.09 |
| tournament_size_3 | 0.353820 ± 0.013085 | 14.39 ± 2.39 |
| tournament_size_5 | 0.330839 ± 0.015906 | 15.06 ± 0.96 |
| tournament_size_10 | 0.354964 ± 0.019766 | 15.29 ± 2.30 |


### 20 Colors

| Configuration | Average Best Fitness | Execution Time (s) |
|---------------|----------------------|--------------------|
| roulette_selection | 0.415128 ± 0.011624 | 18.26 ± 1.29 |
| rank_selection | 0.459290 ± 0.019395 | 16.55 ± 1.73 |
| tournament_size_3 | 0.427797 ± 0.014255 | 16.15 ± 2.44 |
| tournament_size_5 | 0.473634 ± 0.012701 | 18.83 ± 0.54 |
| tournament_size_10 | 0.470437 ± 0.017955 | 9.34 ± 1.66 |


## Reference Images

Best results from each configuration:


### 4 Colors

<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>tournament_size_3</strong></p>
<img src='colors_4\tournament_size_3.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.042889</p>
</div>
<div style='text-align: center;'>
<p><strong>roulette_selection</strong></p>
<img src='colors_4\roulette_selection.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.041609</p>
</div>
<div style='text-align: center;'>
<p><strong>rank_selection</strong></p>
<img src='colors_4\rank_selection.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.040622</p>
</div>
<div style='text-align: center;'>
<p><strong>tournament_size_10</strong></p>
<img src='colors_4\tournament_size_10.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.037959</p>
</div>
<div style='text-align: center;'>
<p><strong>tournament_size_5</strong></p>
<img src='colors_4\tournament_size_5.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.032065</p>
</div>
</div>


### 10 Colors

<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>tournament_size_10</strong></p>
<img src='colors_10\tournament_size_10.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.372148</p>
</div>
<div style='text-align: center;'>
<p><strong>tournament_size_3</strong></p>
<img src='colors_10\tournament_size_3.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.364213</p>
</div>
<div style='text-align: center;'>
<p><strong>tournament_size_5</strong></p>
<img src='colors_10\tournament_size_5.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.349095</p>
</div>
<div style='text-align: center;'>
<p><strong>rank_selection</strong></p>
<img src='colors_10\rank_selection.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.340251</p>
</div>
<div style='text-align: center;'>
<p><strong>roulette_selection</strong></p>
<img src='colors_10\roulette_selection.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.330544</p>
</div>
</div>


### 20 Colors

<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
<div style='text-align: center;'>
<p><strong>tournament_size_10</strong></p>
<img src='colors_20\tournament_size_10.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.491149</p>
</div>
<div style='text-align: center;'>
<p><strong>tournament_size_5</strong></p>
<img src='colors_20\tournament_size_5.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.488221</p>
</div>
<div style='text-align: center;'>
<p><strong>rank_selection</strong></p>
<img src='colors_20\rank_selection.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.478052</p>
</div>
<div style='text-align: center;'>
<p><strong>tournament_size_3</strong></p>
<img src='colors_20\tournament_size_3.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.436874</p>
</div>
<div style='text-align: center;'>
<p><strong>roulette_selection</strong></p>
<img src='colors_20\roulette_selection.png' style='width: 100%; max-width: 300px;'>
<p>Fitness: 0.427434</p>
</div>
</div>

