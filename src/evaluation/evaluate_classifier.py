import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

def evaluate_classifier(model, X_test, y_test, runtime=None):
    """
    Evaluate a fitted binary classification model.

    Parameters
    ----------
    model : fitted sklearn classifier
    X_test : array-like
    y_test : array-like
    runtime : float, optional
        Time (in seconds) used to fit the model.

    Returns
    -------
    results : dict
        Dictionary containing evaluation metrics.
    """

    # predicted class labels
    y_pred = model.predict(X_test)

    # predicted probabilities for the positive class
    y_prob = model.predict_proba(X_test)[:, 1]

    # coefficient information (if available)
    if hasattr(model, "coef_"):
        coefficients = model.coef_[0]
        selected_features = np.sum(coefficients != 0)
    else:
        coefficients = None
        selected_features = None

    results = {
    "metrics": {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "runtime": runtime

    },
    "selected_features": selected_features,
    "coefficients": coefficients,
    "prediction": y_pred,
    "probability": y_prob
}

    return results