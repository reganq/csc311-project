"""
Logistic Regression Model Family
"""
import sys
import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score

# maximum number of iterations to train for
MAX_ITER = 2000
ADA_ITER = 1000
TOLERANCE = 0.0001

# random state for adaboost reproducability
RANDOM_STATE = 20260312

# bounds for random search
LAMBDA_BOUND = (1, 100)
LAMBDA_STEP = 2
LAMBDA_DIV = 100

REGULARIZER_ARR = [
    'l1',
    'l2',
    None # note that it doesn't matter if we pass a lambda without a regularizer, it will be ignored
]

NEST_BOUND = (1, 5)
NEST_STEP = 1

LEARN_BOUND = (1, 100)
LEARN_STEP = 2
LEARN_DIV = 100

# Hyperparameters values for linear regression
LAMBDA = [0.1, 0.25, 0.5, 0.75, 0.8, 1]
REGULARIZER = ['l1', 'l2', None]

# hyperparams for adaboost (linreg also used)
NUM_ESTIMATORS = [25, 50, 75, 100, 150]
LEARNING_RATE = [0.01, 0.1, 0.25, 0.5, 1]

# metadata headers
METADATA_HEADERS = [
    "unique_id",
    "painting",
    "is_train"
]

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
Train and return a logistic regression model with the given hyperparamters, using adaboost. 
Arguments:
    - X: 2D numpy array containing the training set.
    - t: 1D numpy array containing the labels.
    - reg: Which regularizer to use. Can be 'l1', 'l2', or None, in which case no regularizer used.
    - lam: Regularization parameter. Must be set if str is not None
    - rate: Learning rate to use in Adaboost training
    - Estimators: Number of estimators to compute
"""
def train_adaboost(X: np.array, t: np.array, reg: str=None, lam: float=0, rate: float=1, estimators: int=50) -> AdaBoostClassifier:
    if reg is not None and lam == 0:
        print("Set lam if using regularizer")
        return None
    
    if reg == 'l1':
        log = LogisticRegression(fit_intercept=True, max_iter=ADA_ITER, tol=TOLERANCE, C=(1/lam),
                                     l1_ratio=1, solver='saga')
    elif reg == 'l2':
        log = LogisticRegression(fit_intercept=True, max_iter=ADA_ITER, tol=TOLERANCE, C=(1/lam),
                                     l1_ratio=0)
    else:
        log = LogisticRegression(fit_intercept=True, max_iter=ADA_ITER, tol=TOLERANCE, C=np.inf, l1_ratio=0)

    ada = AdaBoostClassifier(estimator=log, n_estimators=estimators, learning_rate=rate, random_state=RANDOM_STATE)

    ada.fit(X, t)

    return ada


def build_all_models(lam,
                     regularizer,
                     estimators,
                     learning,
                     X_train,
                     t_train,
                     X_valid,
                     t_valid,
                     random_search=False) -> dict:
    """
    Returns a dictionary, `out`, whose keys are the the hyperparameter choices, and whose values are
    the training and validation accuracies (via the `score()` method).

    If estimators are learning are None, then trains a logistic regression model. Otherwise, trains 
    an adaboost model with logistic regression as the weak learner.
    Arguments:
        - lam: Regularization strength
        - regularizer: Which regularizer is used
        - estimators: Number of estimators in adaboost. Stops if perfect accuracy reached
        - learning: Learning rate for adaboost
        - random_search: if this is set, all parameter arrays are the same size. Train one model for each parameter
    """
    out = {}

    train_log = learning is None or estimators is None

    if random_search:
        for i in range(len(lam)):
            la = lam[i]
            r = regularizer[i]
            e = estimators[i] if not train_log else 0
            le = learning[i] if not train_log else 0

            t = (la, r) if train_log else (la, r, e, le)

            out[t] = {}
            # Train a model based on the given hyperparameters
            if train_log:
                model = train_logistic(X_train, t_train, reg=r, lam=la)
            else:
                model = train_adaboost(X_train, t_train, reg=r, lam=la, estimators=e, rate=le)

            # store the validation and training scores in the `out` dictionary
            t_pred = model.predict(X_val)
            t_pred_train = model.predict(X_train)
            out[t]['val_acc'] = accuracy_score(t_val, t_pred)
            out[t]['val_recall'] = recall_score(t_val, t_pred, average='macro')
            out[t]['val_pre'] = precision_score(t_val, t_pred, average='macro')
            out[t]['val_f1'] = f1_score(t_val, t_pred, average='macro')
            out[t]['train_acc'] = accuracy_score(t_train, t_pred_train)

    else:
        estimators_tmp = [0] if train_log else estimators
        learning_tmp = [0] if train_log else learning

        for la in lam:
            for r in regularizer:
                for e in estimators_tmp:
                    for le in learning_tmp:

                        t = (la, r) if train_log else (la, r, e, le)

                        out[t] = {}
                        # Train a model based on the given hyperparameters
                        if train_log:
                            model = train_logistic(X_train, t_train, reg=r, lam=la)
                        else:
                            model = train_adaboost(X_train, t_train, reg=r, lam=la, estimators=e, rate=le)

                        # store the validation and training scores in the `out` dictionary
                        t_pred = model.predict(X_val)
                        t_pred_train = model.predict(X_train)
                        out[t]['val_acc'] = accuracy_score(t_val, t_pred)
                        out[t]['val_recall'] = recall_score(t_val, t_pred, average='macro')
                        out[t]['val_pre'] = precision_score(t_val, t_pred, average='macro')
                        out[t]['val_f1'] = f1_score(t_val, t_pred, average='macro')
                        out[t]['train_acc'] = accuracy_score(t_train, t_pred_train)

    return out

"""
Return a random value within the given bounds, aligned to the given step and divided by the given divisor.
"""
def get_rand_val(bound: tuple[int, int], step: int) -> int:
    return random.randrange(bound[0], bound[1] + 1, step)

"""
Create param dictionary with N elements, randomly generated
"""
def generate_params_rand(N: int) -> dict[str, list]:
    out_d = {'lambda': [], 'regularizer': [], 'estimators': [], 'learning': []}

    for _ in range(N):
        out_d['lambda'].append(get_rand_val(LAMBDA_BOUND, LAMBDA_STEP) / LAMBDA_DIV)
        out_d['learning'].append(get_rand_val(LEARN_BOUND, LEARN_STEP) / LEARN_STEP)
        out_d['estimators'].append(get_rand_val(NEST_BOUND, NEST_STEP))
        out_d['regularizer'].append(REGULARIZER_ARR[random.randint(0, 2)])

    return out_d

"""
Train the adaboost model with varying hyperparameters. Save the results in the 
provided file and print out the best configuration and validation/test score.
Arguments:
    - X_train: 2D numpy array of training features
    - t_train: 1D numpy array of training labels
    - X_val: 2D numpy array of validation features
    - t_val: 1D numpy array of validation labels
    - N: number of random parameters to test.
"""
def parameter_search_adaboost(X_train: np.array, t_train: np.array, X_val: np.array, t_val: np.array, N: int=0) -> dict:
    if N == 0:
        out = build_all_models(lam=LAMBDA, regularizer=REGULARIZER, estimators=NUM_ESTIMATORS, learning=LEARNING_RATE, random_search=False, X_train=X_train, X_valid=X_val, t_train=t_train, t_valid=t_val)
    else:
        params = generate_params_rand(N)

        out = build_all_models(lam=params['lambda'], regularizer=params['regularizer'], estimators=params['estimators'], learning=params['learning'], random_search=True, X_train=X_train, X_valid=X_val, t_train=t_train, t_valid=t_val)

    # search for the optimal combination of parameters
    max_acc = 0
    acc_entry = {}
    params_acc = None
    max_recall = 0
    recall_entry = {}
    params_recall = None
    max_pre = 0
    pre_entry = {}
    params_pre = None
    max_f1 = 0
    f1_entry = {}
    params_f1 = None
    for t in out:
        if out[t]['val_acc'] > max_acc:
            max_acc = out[t]['val_acc']
            params_acc = t
            acc_entry = out[t]
        if out[t]['val_recall'] > max_recall:
            max_recall = out[t]['val_acc']
            params_recall = t
            recall_entry = out[t]
        if out[t]['val_pre'] > max_pre:
            max_pre = out[t]['val_pre']
            params_pre = t
            pre_entry = out[t]
        if out[t]['val_f1'] > max_f1:
            max_f1 = out[t]['val_f1']
            params_f1 = t
            f1_entry = out[t]

    print('ACCURACY')
    print(f"\tBest parameters: {params_acc}")
    print(f"\tBest score: {acc_entry}")

    print('RECALL')
    print(f"\tBest parameters: {params_recall}")
    print(f"\tBest score: {recall_entry}")

    print('PRECISION')
    print(f"\tBest parameters: {params_pre}")
    print(f"\tBest score: {pre_entry}")

    print('F1')
    print(f"\tBest parameters: {params_f1}")
    print(f"\tBest score: {f1_entry}")

    return out


"""
Train the adaboost model with varying hyperparameters. Save the results in the 
provided file and print out the best configuration and validation/test score.
Arguments:
    - X_train: 2D numpy array of training features
    - t_train: 1D numpy array of training labels
    - X_val: 2D numpy array of validation features
    - t_val: 1D numpy array of validation labels
    - output_file: file to write to
"""
def parameter_search_logistic(X_train: np.array, t_train: np.array, X_val: np.array, t_val: np.array, N: int=0) -> dict:
    if N == 0:
        out = build_all_models(lam=LAMBDA, regularizer=REGULARIZER, estimators=None, learning=None, random_search=False, X_train=X_train, X_valid=X_val, t_train=t_train, t_valid=t_val)
    else:
        params = generate_params_rand(N)

        out = build_all_models(lam=params['lambda'], regularizer=params['regularizer'], estimators=None, learning=None, random_search=True, X_train=X_train, X_valid=X_val, t_train=t_train, t_valid=t_val)

    # search for the optimal combination of parameters
    max_acc = 0
    acc_entry = {}
    params_acc = None
    max_recall = 0
    recall_entry = {}
    params_recall = None
    max_pre = 0
    pre_entry = {}
    params_pre = None
    max_f1 = 0
    f1_entry = {}
    params_f1 = None
    for t in out:
        if out[t]['val_acc'] > max_acc:
            max_acc = out[t]['val_acc']
            params_acc = t
            acc_entry = out[t]
        if out[t]['val_recall'] > max_recall:
            max_recall = out[t]['val_acc']
            params_recall = t
            recall_entry = out[t]
        if out[t]['val_pre'] > max_pre:
            max_pre = out[t]['val_pre']
            params_pre = t
            pre_entry = out[t]
        if out[t]['val_f1'] > max_f1:
            max_f1 = out[t]['val_f1']
            params_f1 = t
            f1_entry = out[t]

    print('ACCURACY')
    print(f"\tBest parameters: {params_acc}")
    print(f"\tBest score: {acc_entry}")

    print('RECALL')
    print(f"\tBest parameters: {params_recall}")
    print(f"\tBest score: {recall_entry}")

    print('PRECISION')
    print(f"\tBest parameters: {params_pre}")
    print(f"\tBest score: {pre_entry}")

    print('F1')
    print(f"\tBest parameters: {params_f1}")
    print(f"\tBest score: {f1_entry}")

    return out


"""
Normalizes the columns of the given numpy array to have mean 0 and standard deviation 1.
"""
def normalize(X: np.array) -> np.array:
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return (X - mean) / std

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f'usage: {sys.argv[0]} <data-path> <outfile> <type> <num_hyper OPTIONAL> <text-type OPTIONAL>')
        exit(1)

    # seed the RNG
    random.seed(RANDOM_STATE)

    # basic args
    dpath = sys.argv[1]
    ofile = sys.argv[2]
    train_ada = sys.argv[3] == 'ada'

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

    # number of random parameters to sample
    if len(sys.argv) > 4:
        N = int(sys.argv[4])
    else:
        N = 0

    # split off the features
    X_train = np.array(train_df.drop(METADATA_HEADERS, axis=1))
    X_val = np.array(val_df.drop(METADATA_HEADERS, axis=1))

    # normalize the data
    if len(sys.argv) < 6 or int(sys.argv[5]) == 1:
        # similarity vector
        X_train = normalize(X_train)
        X_val = normalize(X_val)
    else:
        # bag of words
        X_train[:, :20] = normalize(X_train[:, :20])
        X_val[:, :20] = normalize(X_val[:, :20])

    # do the training
    if train_ada:
        out_dict = parameter_search_adaboost(X_train, t_train, X_val, t_val, N)
        names = ['lambda', 'regularizer', 'num_estimators', 'learning_rate']
    else:
        out_dict = parameter_search_logistic(X_train, t_train, X_val, t_val, N)
        names = ['lambda', 'regularizer']

    res_df = pd.DataFrame.from_dict(out_dict, orient='index')
    res_df.index.names = names
    res_df.to_csv(ofile)
