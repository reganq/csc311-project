"""
Train the best model and extract the parameters.

Best model (on all features):
    
"""
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# default value from neural_network.py
TRAINING_DATA = "cleaned_data.csv"
RANDOM_STATE = 1
MAX_ITER = 250

# maximum number of iterations to train for
MAX_ITER_LOG = 1000
TOLERANCE = 0.0001
RANDOM_STATE_ADA = 20260312

PARAMS_DIR = "regression_params"

METADATA_HEADERS = [
    "unique_id",
    "painting",
    "is_train"
]

LAMBDA = 9.332471

"""
Normalizes the columns of the given numpy array to have mean 0 and standard deviation 1.
"""
def normalize(X: np.array) -> np.array:
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    print('mean:')
    print(mean)
    print('std:')
    print(std)

    return (X - mean) / std

if __name__ == '__main__':
    # copy in the training data
    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df['is_train'] == True]
    val_df = df[df['is_train'] == False]
    
    t_train = np.argmax(np.stack([
        (train_df['painting'] == 'The Persistence of Memory').astype(np.int8),
        (train_df['painting'] == 'The Starry Night').astype(np.int8),
        (train_df['painting'] == 'The Water Lily Pond').astype(np.int8)
    ]), axis=0)

    t_val = np.argmax(np.stack([
        (val_df['painting'] == 'The Persistence of Memory').astype(np.int8),
        (val_df['painting'] == 'The Starry Night').astype(np.int8),
        (val_df['painting'] == 'The Water Lily Pond').astype(np.int8)
    ]), axis=0)

    # remove headers
    X_train = np.array(train_df.drop(METADATA_HEADERS, axis=1))
    X_val = np.array(val_df.drop(METADATA_HEADERS, axis=1))

    # similarity vectors - normalize everything
    X_train = normalize(X_train)
    X_val = normalize(X_val)
    
    # train the logistic regression model
    log = LogisticRegression(fit_intercept=True, max_iter=MAX_ITER_LOG, tol=TOLERANCE, C=(1/LAMBDA), l1_ratio=0)

    log.fit(X_train, t_train)

    # extract params - saved as numpy format
    np.save(os.path.join(PARAMS_DIR, f'weights'), log.coef_)
    np.save(os.path.join(PARAMS_DIR, f'bias'), log.intercept_)
