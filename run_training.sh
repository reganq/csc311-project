#!/bin/bash

# runs the training scripts

TIMESTAMP=$(date +"%m-%d-%H-%M")
FILENAME="out_${TIMESTAMP}.txt"
touch "$FILENAME"

echo "NEURAL NETWORK - GRIDSEARCH - SIM VECTOR" >> "$FILENAME"
python3 neural_network/neural_network.py cleaned_data.csv neural_network/nn_tuning_results_non_bag_grid.csv 0 1 >> "$FILENAME"

echo "NEURAL NETWORK - RANDOM SEARCH N=500 - SIM VECTOR" >> "$FILENAME"
python3 neural_network/neural_network.py cleaned_data.csv neural_network/nn_tuning_results_non_bag_rand.csv 500 1 >> "$FILENAME"

echo "NEURAL NETWORK - GRIDSEARCH - BAG OF WORDS" >> "$FILENAME"
python3 neural_network/neural_network.py cleaned_data_bag.csv neural_network/nn_tuning_results_bag_grid.csv 0 0 >> "$FILENAME"

echo "NEURAL NETWORK - RANDOM SEARCH N=500 - BAG OF WORDS" >> "$FILENAME"
python3 neural_network/neural_network.py cleaned_data_bag.csv neural_network/nn_tuning_results_bag_rand.csv 500 0 >> "$FILENAME"

echo "LOGISTIC REGRESSION - GRIDSEARCH - SIM VECTOR" >> "$FILENAME"
python3 regression/regression.py cleaned_data.csv regression/log-non_bag-grid.csv log 0 1 >> "$FILENAME"

echo "LOGISTIC REGRESSION - RANDOM SEARCH N=500 - SIM VECTOR" >> "$FILENAME"
python3 regression/regression.py cleaned_data.csv regression/log-non_bag-rand.csv log 500 1 >> "$FILENAME"

echo "LOGISTIC REGRESSION - GRIDSEARCH - BAG OF WORDS" >> "$FILENAME"
python3 regression/regression.py cleaned_data_bag.csv regression/log-bag-grid.csv log 0 0 >> "$FILENAME"

echo "LOGISTIC REGRESSION - RANDOM SEARCH N=500 - BAG OF WORDS" >> "$FILENAME"
python3 regression/regression.py cleaned_data_bag.csv regression/log-bag-rand.csv log 500 0 >> "$FILENAME"

echo "ADABOOST - GRIDSEARCH - SIM VECTOR" >> "$FILENAME"
python3 regression/regression.py cleaned_data.csv regression/ada-non_bag-grid.csv ada 0 1 >> "$FILENAME"

echo "ADABOOST - RANDOM SEARCH N=500 - SIM VECTOR" >> "$FILENAME"
python3 regression/regression.py cleaned_data.csv regression/ada-non_bag-rand.csv ada 500 1 >> "$FILENAME"

echo "ADABOOST - GRIDSEARCH - BAG OF WORDS" >> "$FILENAME"
python3 regression/regression.py cleaned_data_bag.csv regression/ada-bag-grid.csv ada 0 0 >> "$FILENAME"

echo "ADABOOST - RANDOM SEARCH N=500 - BAG OF WORDS" >> "$FILENAME"
python3 regression/regression.py cleaned_data_bag.csv regression/ada-bag-rand.csv ada 500 0 >> "$FILENAME"
