"""
Train the best model and extract the parameters.

Best model (on all features):
    - learning rate: 0.000657
    - activation = tanh
    - batch size = 46
    hidden layer sizes = (310, 405, 50)

Stats:
    - 'val_acc': 0.9393939393939394
    - 'val_recall': 0.9393939393939393
    - 'val_pre': 0.9395711500974658
    - 'val_f1': 0.9393294014853648
    - 'train_acc': 1.0

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
import os
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# default value from neural_network.py
TRAINING_DATA = "cleaned_data.csv"
RANDOM_STATE = 1
MAX_ITER = 250

# maximum number of iterations to train for
MAX_ITER_LOG = 1000
TOLERANCE = 0.0001
RANDOM_STATE_ADA = 20260312

PARAMS_DIR = "neural_params"

METADATA_HEADERS = [
    "unique_id",
    "painting",
    "is_train"
]

ALPHA = 0.000657
ACTIVATION = 'tanh'
BATCH = 46
HIDDEN = (310, 405, 50)

LAM = 9.489221
REG = None
NITERS = 260
LEARN = 2.2

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
    
    # train the neural network model
    mlp = MLPClassifier(
        alpha=ALPHA, 
        activation=ACTIVATION, 
        batch_size=BATCH, 
        hidden_layer_sizes=HIDDEN, 
        random_state=RANDOM_STATE, 
        max_iter=MAX_ITER
    )

    mlp.fit(X_train, t_train)

    # create and save a confusion matrix
    mat = ConfusionMatrixDisplay.from_estimator(
        mlp,
        X_train,
        t_train,
        display_labels=["Persistence", "Starry", "Water Lily"], #["The Persistence of Memory","The Starry Night","The Water Lily Pond"],
        cmap=plt.cm.Blues,
    )

    plt.title("Confusion Matrix - Neural Network")
    plt.savefig("neural-confusion-test.png")
    plt.close()
    
    # train the logistic regression
    log = LogisticRegression(fit_intercept=True, max_iter=MAX_ITER_LOG, tol=TOLERANCE, C=np.inf, l1_ratio=0)
    ada = AdaBoostClassifier(estimator=log, n_estimators=NITERS, learning_rate=LEARN, random_state=RANDOM_STATE_ADA)
    ada.fit(X_train, t_train)

    # create and save a confusion matrix
    mat = ConfusionMatrixDisplay.from_estimator(
        ada,
        X_train,
        t_train,
        display_labels=["Persistence", "Starry", "Water Lily"], #["The Persistence of Memory","The Starry Night","The Water Lily Pond"],
        cmap=plt.cm.Blues,
    )

    plt.title("Confusion Matrix - Logistic Regression trained with Adaboost")
    plt.savefig("ada-confusion-test.png")
    plt.close()

    # extract params - saved as numpy format
    #for i in range(len(mlp.coefs_)):
    #    np.save(os.path.join(PARAMS_DIR, f'weights-{i}'), mlp.coefs_[i])

    
