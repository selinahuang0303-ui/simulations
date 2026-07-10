# 2026-07-09

## Objective

Design a reusable simulation framework for the causal inference project.

## Progress

### Framework Design

Designed the overall simulation pipeline:

Generate Predictors
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