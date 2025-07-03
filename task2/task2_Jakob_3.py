# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.impute import KNNImputer, SimpleImputer

from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

verbose = False


def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")

    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')

    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # Dummy initialization of the X_train, X_test and y_train
    # TODO: Depending on how you deal with the non-numeric data, you may want to
    # modify/ignore the initialization of these variables

    # X_train = np.zeros_like(train_df.drop(['price_CHF'], axis=1))
    # y_train = np.zeros_like(train_df['price_CHF'])
    # X_test = np.zeros_like(test_df)

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test

    # Replace non-numeric values
    train_df = train_df.replace(to_replace='spring', value=0)
    train_df = train_df.replace(to_replace='summer', value=1)
    train_df = train_df.replace(to_replace='autumn', value=2)
    train_df = train_df.replace(to_replace='winter', value=3)

    test_df = test_df.replace(to_replace='spring', value=0)
    test_df = test_df.replace(to_replace='summer', value=1)
    test_df = test_df.replace(to_replace='autumn', value=2)
    test_df = test_df.replace(to_replace='winter', value=3)

    # Modify the NaN values in the provided data depending on given mod value
    mod = 3

    if mod == 0:
        train_df = train_df.dropna(how="any")
        test_df = test_df.dropna(how="any")
    elif mod == 1:
        train_df = train_df.fillna(train_df.mean())
        test_df = test_df.fillna(test_df.mean())
    elif mod == 2:
        train_df = train_df.fillna(train_df.median())
        test_df = test_df.fillna(test_df.median())
    elif mod == 3:
        train_df = train_df.interpolate(method="pchip", limit_direction="both")
        test_df = test_df.interpolate(method="pchip", limit_direction="both")

    # Transform the data from panda to numeric array representation
    X_train = train_df.drop(['price_CHF'], axis=1).to_numpy(dtype=float, na_value=np.nan)
    y_train = train_df['price_CHF'].to_numpy(dtype=float, na_value=np.nan)
    X_test = test_df.to_numpy(dtype=float, na_value=np.nan)

    if mod == 4:
        # Define imputer
        imputer = KNNImputer()

        # impute train data
        X_train = imputer.fit_transform(X_train)

        # impute test data
        X_test = imputer.fit_transform(X_test)

    if verbose:
        print("Training data:")
        print("Shape:", train_df.shape)
        print(train_df.head(2))
        print('\n')
        print("Test data:")
        print(test_df.shape)
        print(test_df.head(2))

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test


def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    y_pred = np.zeros(X_test.shape[0])

    # TODO: Define the model and fit it using training data. Then, use test data to make predictions

    gpr = GaussianProcessRegressor(kernel=Matern())
#   gpr.fit(X_train, y_train)
#   y_pred = gpr.predict(X_test)

    if verbose:
        # Cross-Validation: Do Cross-Validation on model, score with rs_score for each test; compute mean; choose model with highest mean score

        n_folds = 3
        mean_score = 0

        # create the splits
        kf = KFold(n_splits=n_folds)

        # iterate over all splits and lambdas
        # for all, fit the ridge regression and then calculate the RMSE for each
        for i, (train, test) in enumerate(kf.split(X_train)):
            X_pred, X_score = X_train[train], X_train[test]
            y_pred, y_score = y_train[train], y_train[test]
            gpr.fit(X_pred, y_pred)
            y_predicted = gpr.predict(X_score)
            mean_score += r2_score(y_score, y_predicted)

        mean_score /= n_folds

        print(mean_score)

        ######

    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_test)

    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred = modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred)
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")
