from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import time
from sklearn.pipeline import Pipeline

def tune_neural_network(
    X_train,
    Y_train,
    param_grid,
    cv=3
):

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(random_state=42))
    ])

    pipeline_param_grid = {
        f"model__{key}": value
        for key, value in param_grid.items()
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=pipeline_param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=2
    )

    start = time.perf_counter()
    grid_search.fit(X_train, Y_train)
    tuning_runtime = time.perf_counter() - start

    best_params = {
        key.replace("model__", ""): value
        for key, value in grid_search.best_params_.items()
    }

    return {
        "best_params": best_params,
        "best_cv_mse": -grid_search.best_score_,
        "tuning_runtime": tuning_runtime
    }


from sklearn.metrics import mean_squared_error


def fit_neural_network(
    X_train,
    Y_train,
    X_test,
    Y_test,
    params
):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    nn_model = MLPRegressor(
        **params,
        random_state=42
    )

    start = time.perf_counter()

    nn_model.fit(X_train_scaled, Y_train)

    fit_runtime = time.perf_counter() - start

    nn_pred = nn_model.predict(X_test_scaled)

    mse = mean_squared_error(Y_test, nn_pred)

    return {
        "model": nn_model,
        "prediction": nn_pred,
        "mse": mse,
        "fit_runtime": fit_runtime,
        "n_iter": nn_model.n_iter_,
        "loss": nn_model.loss_,
        "loss_curve": nn_model.loss_curve_,
        "converged": nn_model.n_iter_ < params["max_iter"]
    }