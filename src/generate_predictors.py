import numpy as np

def generate_predictors(x_config):
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
    n = x_config["size"]
    variable_type = x_config["variable_type"]
    distribution = x_config["distribution"]
    params = x_config["distribution_params"]
    dimension = x_config["dimension"]

    # Normal Distribution
    if variable_type == "continuous" and distribution == "normal":
        
        X = np.random.normal(
            loc=params["mean"],
            scale=params["std"],
            size=(n, dimension))
    else:
        raise ValueError("Unsupported variable type")

    return X
