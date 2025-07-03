import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def fit(X, y, lam):
    """
    Fit ridge regression model with regularization parameter lambda.

    Parameters:
    X: array-like, shape (n_samples, n_features)
        Training data.
    y: array-like, shape (n_samples,)
        Target values.
    lam: float
        Regularization parameter lambda.

    Returns:
    w: array-like, shape (n_features,)
        Optimal parameters of ridge regression.
    """
    n_features = X.shape[1]
    gram_matrix = X.T @ X
    reg_term = np.eye(n_features) * lam
    w = np.linalg.solve(gram_matrix + reg_term, X.T @ y)
    return w

def calculate_RMSE(w, X, y):
    """
    Calculate Root Mean Squared Error (RMSE).

    Parameters:
    w: array-like, shape (n_features,)
        Model parameters.
    X: array-like, shape (n_samples, n_features)
        Test data.
    y: array-like, shape (n_samples,)
        True labels.

    Returns:
    RMSE: float
        Root Mean Squared Error.
    """
    y_pred = X @ w
    squared_errors = (y_pred - y) ** 2
    RMSE = np.sqrt(np.mean(squared_errors))
    return RMSE

def average_LR_RMSE(X, y, lambdas, n_folds):
    """
    Perform K-fold cross-validation for ridge regression.

    Parameters:
    X: array-like, shape (n_samples, n_features)
        Input data.
    y: array-like, shape (n_samples,)
        Target values.
    lambdas: list
        List of regularization parameters.
    n_folds: int
        Number of folds.

    Returns:
    avg_RMSE: array-like, shape (n_lambdas,)
        Average RMSE for each lambda.
    """
    kf = KFold(n_splits=n_folds)
    RMSE_mat = np.zeros((n_folds, len(lambdas)))

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for j, lam in enumerate(lambdas):
            w = fit(X_train, y_train, lam)
            RMSE = calculate_RMSE(w, X_test, y_test)
            RMSE_mat[i, j] = RMSE

    avg_RMSE = np.mean(RMSE_mat, axis=0)
    return avg_RMSE

if __name__ == "__main__":
    # Load data
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    X = data.drop(columns="y").to_numpy()

    # Define lambda values and perform cross-validation
    lambdas = [0.1, 1, 10, 100, 200]
    n_folds = 10
    avg_RMSE = average_LR_RMSE(X, y, lambdas, n_folds)

    # Compute score
    v_star = np.array([9.303763455113494] * len(lambdas))
    score = 100 * np.sum(np.abs(avg_RMSE - v_star) / v_star)

    # Save results
    np.savetxt("./results.csv", avg_RMSE, fmt="%.12f")
