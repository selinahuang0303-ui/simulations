## 2026-07-09

### Objective

Design a reusable simulation framework for the causal inference project.

### Progress

### Framework Design

Designed the overall simulation pipeline:

→ Generate Predictors
→ Introduce Correlation
→ Generate Response
→ Fit Models
→ Simulation
→ Experiment
→ Evaluation

Clarified the responsibilities of each module.

### Predictor Generation

Started implementing `generate_predictors()`.

Current functionality:
- Generates continuous variables.
- Supports arbitrary dimensions.
- Uses configurable random seed.
- Currently supports Normal distribution.

### Design Decisions

1. Generate X1 and X2 separately.
2. Separate predictor generation from response generation.
3. Separate correlation generation from predictor generation.
4. Use modular functions so new variable types and response mechanisms can be added later.

### Problems Encountered

1. Originally assumed distribution parameters belonged in the global configuration.
2. Realized different predictor blocks (e.g., X1 and X2) may have different distributions.
3. Concluded that predictor-specific settings should not be stored globally.

### Next Steps

- Redesign configuration structure.
- Continue implementing `generate_predictors()`.
- Add binary predictor generation.
- Design correlation module.


## 2026-07-09

### Progress

* Tested and validated `generate_predictors()`.
* Implemented:

  * `introduce_correlation()`
  * `fit_lasso()`
  * `fit_ridge()`
  * `fit_random_forest()`
  * `fit_neural_network()`
* Started **Experiment 01**:

  * (n = 5000), (d = 10), (p = 500)
  * Independent predictors
  * Linear relationship (signal in X1 only)
  * Compared all four baseline models.

### Notes

* Completed the first end-to-end simulation pipeline (data generation → model fitting → evaluation).
* Verified that repeated simulations require generating a new dataset each iteration.
* Random Forest was much slower than the other models; reducing `n_estimators` and `max_depth` significantly improved runtime for development.

### Next Steps

* Optimize the experiment workflow for repeated simulations.
* Tune hyperparameters for each model.
* Complete and summarize Experiment 01.
* Finalize the overall experiment plan.
* Add the project structure and experiment plan to `README.md`.

## 2026-07-12

* Added hyperparameter tuning functions for all baseline models
* Separated model tuning and fitting
* Simplified the experiment workflow for clarity; may optimize later

### Next Steps

* Add the evaluation step to the workflow
* Run experiments using the current workflow

## 2026-07-13

* Found that Random Forest tuning takes significantly longer
* Changed the workflow to tune each model once per data setting, since optimal hyperparameters remained stable across repeated tuning in Experiment 01
* Started Experiment 02

### Next Steps

* Finish Experiment 02 and evaluate the results
