import numpy as np

def introduce_correlation(
        X1, X2, corr_config):
    """
    Introduce correlation between two predictor blocks.

    Parameters
    ----------
    X1 : ndarray
        First predictor block (n × d1)

    X2 : ndarray
        Second predictor block (n × d2)

    corr_config : dict
        Example:
        {
            "method": "linear",
            "strength": 0.6,
            "random_state": 42
        }

    Returns
    -------
    X1_corr, X2_corr
        Correlated predictor blocks.
    """

    method = corr_config["method"]
    strength = corr_config["strength"]


    # Simple linear correlation
    if method == "linear":

        if not (0 <= strength <= 1):
            raise ValueError("strength must be between 0 and 1.")

        X1_corr = X1.copy()
        X2_corr = X2.copy()

        # Correlate matching columns
        n_columns = min(X1.shape[1], X2.shape[1])

        for j in range(n_columns):

            noise = np.random.normal(
                loc=0,
                scale=1,
                size=X1.shape[0]
            )

            X2_corr[:, j] = (
                strength * X1[:, j]
                + np.sqrt(1 - strength**2) * noise
            )

        return X1_corr, X2_corr

    else:
        raise ValueError(f"Unknown correlation method: {method}")