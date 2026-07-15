# Baseline Model Comparison

This project compares the performance of baseline machine learning methods under different simulated high-dimensional data settings.

## Models

The baseline models include:

* Lasso Regression
* Ridge Regression
* Random Forest
* Neural Network

Hyperparameter tuning is performed using cross-validation before model evaluation.

## Project Structure

```text
.
├── README.md
├── research_log.md
├── experiments/
│   ├── 01_linear_independent_x1.ipynb
│   ├── 02_linear_independent_x2.ipynb
│   └── test.ipynb
└── src/
    ├── correlation.py
    ├── experiment.py
    ├── generate_predictors.py
    └── models/
        ├── lasso.py
        ├── ridge.py
        ├── random_forest.py
        └── neural_network.py
```

## Research Plan

The simulation study is organized into three stages. Each stage introduces additional complexity to systematically compare baseline model performance under different data-generating settings.

### Stage 1 — Verify the Pipeline

| Experiment | Signal     | Correlation | Relationship |
| ---------- | ---------- | ----------- | ------------ |
| 1          | (X_1) only | No          | Linear       |
| 2          | (X_2) only | No          | Linear       |
| 3          | Both       | No          | Linear       |

### Stage 2 — Add Correlation

| Experiment | Signal     | Correlation | Relationship |
| ---------- | ---------- | ----------- | ------------ |
| 4          | (X_1) only | Yes         | Linear       |
| 5          | (X_2) only | Yes         | Linear       |
| 6          | Both       | Yes         | Linear       |

### Stage 3 — Add Nonlinear Response

| Experiment | Signal     | Correlation | Relationship |
| ---------- | ---------- | ----------- | ------------ |
| 7          | (X_1) only | No          | Nonlinear    |
| 8          | (X_2) only | No          | Nonlinear    |
| 9          | Both       | No          | Nonlinear    |

## Experiments

### Experiment 01: Sparse Linear Signal

The response variable depends on a small number of predictors.

**Result:** Lasso achieved the lowest MSE, followed by Ridge, while Random Forest and the Neural Network performed worse. This suggests that Lasso is well suited for sparse linear relationships.

### Experiment 02: Dense Linear Signal

The response variable depends on approximately 60% of the high-dimensional predictors.

**Result:** Ridge achieved the lowest average MSE, followed by Lasso and the Neural Network, while Random Forest performed substantially worse. This suggests that Ridge is better suited for dense linear relationships, whereas Lasso performs better under sparse signals.

### Experiment 03: Mixed Linear Signal

The response variable depends on a small number of strong predictors from (X_1) and 150 weaker predictors from (X_2).

**Result:** Lasso achieved the lowest average MSE, closely followed by Ridge, while the Neural Network had a higher MSE and Random Forest performed substantially worse. The similar performance of Lasso and Ridge suggests that both methods are effective under a mixed linear signal with strong sparse effects and weaker distributed effects.

### Experiment 04: Sparse Linear Signal with Correlated Predictors

This experiment uses the same sparse signal setting as Experiment 01, with moderate correlation ((r = 0.3)) introduced between (X_1) and (X_2).

**Result:** Overall model performance was similar to Experiment 01, suggesting that moderate correlation did not substantially affect model performance in the current setting. Different levels of correlation may lead to different results and should be investigated in future experiments.


## Evaluation

Current experiments primarily evaluate model performance using Mean Squared Error (MSE). Each experiment is repeated across multiple randomly generated datasets, and average MSE is used to compare baseline methods.

Future evaluations will include additional metrics such as runtime, variable selection performance, and model-specific characteristics.

## Current Work

* Expanding simulation settings
* Comparing baseline methods under different data-generating processes
* Improving the experiment and evaluation workflow

