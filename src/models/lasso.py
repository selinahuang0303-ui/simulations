from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import time


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline



def tune_lasso(X_train, Y_train, alphas, cv=5):

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso())
    ])

    param_grid = {
        "model__alpha": alphas
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_squared_error"
    )

    start = time.perf_counter()
    grid_search.fit(X_train, Y_train)
    tuning_runtime = time.perf_counter() - start

    best_alpha = grid_search.best_params_["model__alpha"]

    return {
        "best_alpha": best_alpha,
        "best_cv_mse": -grid_search.best_score_,
        "tuning_runtime": tuning_runtime
    }


def fit_lasso(
    X_train,
    Y_train,
    X_test,
    Y_test,
    alpha
):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lasso_model = Lasso(alpha=alpha)

    start = time.perf_counter()
    lasso_model.fit(X_train_scaled, Y_train)
    fit_runtime = time.perf_counter() - start

    lasso_pred = lasso_model.predict(X_test_scaled)

    mse = mean_squared_error(Y_test, lasso_pred)

    coefficients = lasso_model.coef_
    intercept = lasso_model.intercept_

    selected_variables = np.where(coefficients != 0)[0]
    n_selected = len(selected_variables)

    return {
        "model": lasso_model,
        "prediction": lasso_pred,
        "mse": mse,
        "fit_runtime": fit_runtime,
        "coefficients": coefficients,
        "intercept": intercept,
        "selected_variables": selected_variables,
        "n_selected": n_selected
    }