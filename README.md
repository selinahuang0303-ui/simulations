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
├── notebooks/
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

## Experiments

### Experiment 01: Sparse Linear Signal

The response variable depends on a small number of predictors.

**Result:** Lasso achieved the lowest MSE, followed by Ridge, while Random Forest and the Neural Network performed worse. This suggests that Lasso is well suited for sparse linear relationships.

### Experiment 02: Dense Linear Signal

The response variable depends on approximately 60% of the high-dimensional predictors.

**Result:** Ridge achieved the lowest average MSE, followed by Lasso and the Neural Network, while Random Forest performed substantially worse. This suggests that Ridge is better suited for dense linear relationships, whereas Lasso performs better under sparse signals.

## Evaluation

Model performance is evaluated using:

* Mean Squared Error (MSE)
* Runtime
* Selected variables for Lasso
* Model-specific outputs such as feature importance and neural network convergence

Experiments are repeated across multiple randomly generated datasets, and average MSE is used to compare model performance.

## Current Work

* Expanding simulation settings
* Comparing baseline methods under different data-generating processes
* Improving the experiment and evaluation workflow

