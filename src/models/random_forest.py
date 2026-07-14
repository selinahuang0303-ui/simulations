from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import time


def tune_random_forest(
    X_train,
    Y_train,
    param_grid,
    cv=5
):
    rf_model = RandomForestRegressor(
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose = 2
    )

    start = time.perf_counter()

    grid_search.fit(X_train, Y_train)

    tuning_runtime = time.perf_counter() - start

    return {
        "best_params": grid_search.best_params_,
        "best_cv_mse": -grid_search.best_score_,
        "tuning_runtime": tuning_runtime
    }


from sklearn.metrics import mean_squared_error

def fit_random_forest(
    X_train,
    Y_train,
    X_test,
    Y_test,
    params
):
    rf_model = RandomForestRegressor(
        **params,
        random_state=42
    )

    start = time.perf_counter()

    rf_model.fit(X_train, Y_train)

    fit_runtime = time.perf_counter() - start

    rf_pred = rf_model.predict(X_test)

    mse = mean_squared_error(Y_test, rf_pred)

    feature_importances = rf_model.feature_importances_

    return {
        "model": rf_model,
        "prediction": rf_pred,
        "mse": mse,
        "fit_runtime": fit_runtime,
        "feature_importances": feature_importances
    }