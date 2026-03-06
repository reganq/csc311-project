"""
Neural Network Model Family
"""
import sys
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

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

"""
def train_neural_network(X: np.array, t: np.array, alpha: float, activation: str, batch_size: int, hidden_layer_sizes: tuple[int], feats: list[int]=[]) -> MLPClassifier:
    mlp = MLPClassifier(alpha=alpha, activation=activation, batch_size=batch_size, hidden_layer_sizes=hidden_layer_sizes)

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

    model = train_neural_network(X_train, t_train, alpha=0.001, activation='relu', batch_size=min(200, len(X_train)), hidden_layer_sizes=(100,), feats=[])
    print(model.score(X_val, t_val))