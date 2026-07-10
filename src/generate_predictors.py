import numpy as np

def generate_predictors(global_config, local_config):
    """
    Generate predictor matrix X.

    Parameters
    ----------
    config : dict
        Dictionary containing simulation settings.
    dimension : int
        Dimension of predictor variables.

    Returns
    -------
    X : numpy.ndarray
        Predictor matrix.
    """

    # Read parameters
    n = global_config["n"]
    random_state = global_config["random_state"]
    variable_type = local_config["variable_type"]
    distribution = local_config["distribution"]
    params = local_config["distribution_params"]
    dimension = local_config["dimension"]

    np.random.seed(random_state)

    # Normal Distribution
    if variable_type == "continuous" and distribution == "normal":
        rng = np.random.default_rng(random_state)
        X = rng.normal(
            loc=params["mean"],
            scale=params["std"],
            size=(n, dimension))
    else:
        raise ValueError("Unsupported variable type")

    return X
