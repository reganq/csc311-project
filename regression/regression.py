"""
Logistic Regression Model Family
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# maximum number of iterations to train for
MAX_ITER = 1000
TOLERANCE = 0.0001

# data headers for likeness vector - 29 features
DATA_HEADERS = [
    "emotional_intensity",
    "evokes_sombre",
    "evokes_content",
    "evokes_calm",
    "evokes_uneasy",
    "prominent_colour_count",
    "prominent_object_count",
    "place_bedroom",
    "place_office",
    "place_living_room",
    "place_dining_room",
    "view_friends",
    "view_family",
    "view_coworkers",
    "view_by_yourself",
    "like_fall",
    "like_winter",
    "like_spring",
    "like_summer",
    "monetary_value",
    "text_clock_feels",
    "text_clock_food",
    "text_clock_music",
    "text_starry_feels",
    "text_starry_food",
    "text_starry_music",
    "text_lilies_feels",
    "text_lilies_food",
    "text_lilies_music"
]

# data headers for bag of words
DATA_HEADERS_2 = []

"""
Train and return a logistic regression model with the given hyperparamters. 
Arguments:
    - X: 2D numpy array containing the training set.
    - t: 1D numpy array containing the labels.
    - reg: Which regularizer to use. Can be 'l1', 'l2', or None, in which case no regularizer used.
    - lam: Regularization parameter. Must be set if str is not None
    - feats: List of integers. Interpreted as the column indices of the features to train with.
"""
def train_logistic(X: np.array, t: np.array, reg: str=None, lam: float=0, feats: list[int]=[]) -> LogisticRegression:
    if reg is not None and lam == 0:
        print("Set lam if using regularizer")
        return None
    
    if reg == 'l1':
        log = LogisticRegression(fit_intercept=True, max_iter=MAX_ITER, tol=TOLERANCE, C=(1/lam),
                                     l1_ratio=1, solver='saga')
    elif reg == 'l2':
        log = LogisticRegression(fit_intercept=True, max_iter=MAX_ITER, tol=TOLERANCE, C=(1/lam),
                                     l1_ratio=0)
    else:
        log = LogisticRegression(fit_intercept=True, max_iter=MAX_ITER, tol=TOLERANCE, C=np.inf, l1_ratio=0)

    if len(feats) != 0:
        X_use = X[:, feats]
    else:
        X_use = X

    log.fit(X_use, t)

    return log

"""
Normalizes the columns of the given numpy array to have mean 0 and standard deviation 1.
"""
def normalize(X: np.array) -> np.array:
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return (X - mean) / std

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f'usage: {sys.argv[0]} <data-path> <text-type OPTIONAL>')
        exit(1)

    dpath = sys.argv[1]
    df = pd.read_csv(dpath)

    # split the data
    train_df = df[df['is_train'] == True]
    val_df = df[df['is_train'] == False]

    # split off the labels - they cannot be one-hot encoded for logreg. Instead, we have 3 targets (0, 1, 2)
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

    # split off the features
    if len(sys.argv) < 3 or int(sys.argv[2]) == 1:
        # use the similarity vectors
        headers = DATA_HEADERS
    else:
        # use bag of words
        headers = DATA_HEADERS_2

    X_train = np.array(train_df.get(headers))
    X_val = np.array(val_df.get(headers))

    # normalize the data
    X_train = normalize(X_train)
    X_val = normalize(X_val)

    model = train_logistic(X_train, t_train, reg='l1', lam=0.5)
    print(model.score(X_val, t_val))
    print(model.coef_.shape)
    print(model.intercept_.shape)
    print(model.coef_)
    print(model.intercept_)
    print(model.get_params()['solver'])
    

    
