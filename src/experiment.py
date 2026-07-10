# experiment.py

from generate_predictors import generate_predictors

GLOBAL_CONFIG = {
    "n": 500,
    "random_state": 42
}

X1_CONFIG = {
    "dimension": 5,
    "variable_type": "continuous",
    "distribution": "normal",
    "distribution_params": {

        "mean": 0,

        "std": 1

    }
}

X2_CONFIG = {
    "dimension": 500,
    "variable_type": "continuous",
    "distribution": "normal",
    "distribution_params": {

        "mean": 2,

        "std": 5

    }
}

X1 = generate_predictors(GLOBAL_CONFIG, X1_CONFIG)
X2 = generate_predictors(GLOBAL_CONFIG, X2_CONFIG)