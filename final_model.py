"""
Train the best model and extract the parameters.

Best model (on two missing features): 
    - learning rate (alpha) = 0.000755
    - activation = tanh
    - batch size = 32
    - hidden layer sizes = (410, 415, 445, 455)
    
Stats:
    - 'val_acc': 0.9363636363636364, 
    - 'val_recall': 0.9363636363636365, 
    - 'val_pre': 0.9376354090950677, 
    - 'val_f1': 0.9364288705667536, 
    - 'train_acc': 1.0
"""
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier

# default value from neural_network.py
TRAINING_DATA = "cleaned_data.csv"
RANDOM_STATE = 1
MAX_ITER = 250

METADATA_HEADERS = [
    "unique_id",
    "painting",
    "is_train"
]

"""
Normalizes the columns of the given numpy array to have mean 0 and standard deviation 1.
"""
def normalize(X: np.array) -> np.array:
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return (X - mean) / std

if __name__ == '__main__':
    # setup the model
    alpha = 0.000755
    activation = 'tanh'
    batch_size = 32
    hidden_layer_sizes = (410, 415, 445, 455)

    mlp = MLPClassifier(
        alpha=alpha, 
        activation=activation, 
        batch_size=batch_size, 
        hidden_layer_sizes=hidden_layer_sizes, 
        random_state=RANDOM_STATE, 
        max_iter=MAX_ITER
    )

    # copy in the training data
    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df['is_train'] == True]
    
    t_train = np.argmax(np.stack([
        (train_df['painting'] == 'The Persistence of Memory').astype(np.int8),
        (train_df['painting'] == 'The Starry Night').astype(np.int8),
        (train_df['painting'] == 'The Water Lily Pond').astype(np.int8)
    ]), axis=0)

    # remove headers
    X_train = np.array(train_df.drop(METADATA_HEADERS, axis=1))

    # similarity vectors - normalize everything
    X_train = normalize(X_train)

    # train the model
    mlp.fit(X_train, t_train)

    # extract params
    print(mlp.coefs_)

