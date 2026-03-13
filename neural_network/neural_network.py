"""
Neural Network Model Family
"""
import sys
import numpy as np
import pandas as pd
import random
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

RANDOM_STATE = 20260312

# maximum number of iterations to train for
MAX_ITER = 1000
TOLERANCE = 0.0001

# bounds for random search
ALPHA_BOUND = (1, 1000)
ALPHA_STEP = 2
ALPHA_DIV = 1000000

BATCH_BOUND = (10, 100)
BATCH_STEP = 2

HL_BOUND = (1, 5)
HL_STEP = 1

HL_COUNT_BOUND = (10, 500)
HL_COUNT_STEP = 5

# Hyperparameters values for neural network
ALPHA = [0.00001, 0.0001, 0.001]
ACTIVATION = ['relu', 'tanh']
BATCH_SIZE = [16, 32, 48, 64]
HIDDEN_LAYER_SIZES = [(25,), (25, 25), (50,), (50, 50), (100,), (100, 100)]

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
Train and return a neural network model with the given hyperparameters.
Arguments:
    - X: 2D numpy array containing the training set.
    - t: 1D numpy array containing the labels.
    - feats: List of integers. Interpreted as the column indices of the features to train with.
    ---Hyperparameters---
    - alpha: Regularization strength.
    - activation: Which activation function to use.
    - batch_size: Size of the mini-batches for stochastic optimizers.
    - hidden_layer_sizes: Tuple of integers. The i-th element represents the number of neurons in the i-th hidden layer.
    - random_state: Seed for the random number generator.
"""
def train_neural_network(X: np.array, t: np.array, alpha: float, activation: str, batch_size: int, hidden_layer_sizes: tuple[int], feats: list[int]=[], random_state: int=1) -> MLPClassifier:
    mlp = MLPClassifier(alpha=alpha, activation=activation, batch_size=batch_size, hidden_layer_sizes=hidden_layer_sizes, random_state=random_state, max_iter=MAX_ITER)

    if len(feats) != 0:
        X_use = X[:, feats]
    else:
        X_use = X

    mlp.fit(X_use, t)

    return mlp

"""
Normalizes the columns of the given numpy array to have mean 0 and standard deviation 1.
"""
def normalize(X: np.array) -> np.array:
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return (X - mean) / std

def build_all_models(alpha,
                     activation,
                     batch_size,
                     hidden_layer_sizes,
                     X_train,
                     t_train,
                     X_valid,
                     t_valid,
                     random_search=False):
    """
    Returns a dictionary, `out`, whose keys are the the hyperparameter choices, and whose values are
    the training and validation accuracies (via the `score()` method).
    Arguments:
        - alpha: Regularization strength.
        - activation: Which activation function to use.
        - batch_size: Size of the mini-batches for stochastic optimizers.
        - hidden_layer_sizes: Tuple of integers. The i-th element represents the number of neurons in the i-th hidden layer.
        - random_search: if this is set, all parameter arrays are the same size. Train one model for each parameter
    """
    out = {}

    if random_search:
        for i in range(len(alpha)):
            a = alpha[i]
            act = activation[i]
            b = batch_size[i]
            h = hidden_layer_sizes[i]

            out[(a, act, b, h)] = {}
            # Create a neural network model based on the given hyperparameters and fit it to the data
            model = train_neural_network(X_train, t_train, alpha=a, activation=act, batch_size=b, hidden_layer_sizes=h)
                        
            # store the validation and training scores in the `out` dictionary
            out[(a, act, b, h)]['val'] = model.score(X_valid, t_valid)
            out[(a, act, b, h)]['train'] = model.score(X_train, t_train)

    else:
        for a in alpha:
            for act in activation:
                for b in batch_size:
                    for h in hidden_layer_sizes:
                        out[(a, act, b, h)] = {}
                        # Create a neural network model based on the given hyperparameters and fit it to the data
                        model = train_neural_network(X_train, t_train, alpha=a, activation=act, batch_size=b, hidden_layer_sizes=h)
                        
                        # store the validation and training scores in the `out` dictionary
                        out[(a, act, b, h)]['val'] = model.score(X_valid, t_valid)
                        out[(a, act, b, h)]['train'] = model.score(X_train, t_train)
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
    out_d = {'alpha': [], 'activation': [], 'batch': [], 'hidden': []}

    for _ in range(N):
        out_d['alpha'].append(get_rand_val(ALPHA_BOUND, ALPHA_STEP) / ALPHA_DIV)
        out_d['batch'].append(get_rand_val(BATCH_BOUND, BATCH_STEP))
        out_d['activation'].append('relu' if (random.randint(0, 1) == 0) else 'tanh')

        # create hidden layer
        hl_size = get_rand_val(HL_BOUND, HL_STEP)
        hl = []
        for _ in range(hl_size):
            hl.append(get_rand_val(HL_COUNT_BOUND, HL_COUNT_STEP))
        out_d['hidden'].append(tuple(hl))


    return out_d

def tune_hyperparams(X_train, X_val, t_train, t_val, N=0):
    # tune hyperparameters - N should only be set if we are randomly generating hyperparameters
    if N == 0:
        res = build_all_models(alpha=ALPHA, activation=ACTIVATION, batch_size=BATCH_SIZE, hidden_layer_sizes=HIDDEN_LAYER_SIZES, random_search=False, X_train=X_train, X_valid=X_val, t_train=t_train, t_valid=t_val)
    else:
        params = generate_params_rand(N)

        res = build_all_models(alpha=params['alpha'], activation=params['activation'], batch_size=params['batch'], hidden_layer_sizes=params['hidden'], random_search=True, X_train=X_train, X_valid=X_val, t_train=t_train, t_valid=t_val)

    # search for the optimal combination of parameters
    max_score = 0
    best_params = None
    for a, act, b, h in res:
        if res[(a, act, b, h)]['val'] > max_score:
            max_score = res[(a, act, b, h)]['val']
            best_params = (a, act, b, h)

    print(f"Best parameters: {best_params}")
    print(f"Best score: {max_score}")

    return res

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f'usage: {sys.argv[0]} <data-path> <outfile> <num_hyper OPTIONAL> <text-type OPTIONAL>')
        exit(1)

    # seed the RNG
    random.seed(RANDOM_STATE)

    dpath = sys.argv[1]
    opath = sys.argv[2]
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

    # remove metadata columns
    X_train = np.array(train_df.drop(METADATA_HEADERS, axis=1))
    X_val = np.array(val_df.drop(METADATA_HEADERS, axis=1))

    # normalize the (non-BOW) data
    if len(sys.argv) < 5 or int(sys.argv[4]) == 1:
        # similarity vectors - normalize everything
        X_train = normalize(X_train)
        X_val = normalize(X_val)
    else:
        # bag of words - only normalize other features
        X_train[:20] = normalize(X_train[:20])
        X_val[:20] = normalize(X_val[:20])

    if len(sys.argv) > 3:
        N = int(sys.argv[3])
    else:
        N = 0

    out_dict = tune_hyperparams(X_train, X_val, t_train, t_val, N)

    res_df = pd.DataFrame.from_dict(out_dict, orient='index')
    res_df.index.names = ['alpha', 'activation', 'batch_size', 'hidden_layer_sizes']
    res_df.to_csv(outfile)
