"""
Generate a list of ids randomly, and print to stdout. Takes the overall
 number of ids, and outputs 70% of the ids, sampled randomly without replacement.
"""
import sys
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} num")
    
    num = int(sys.argv[1])

    arr = np.arange(1, num + 1, 1)

    trainval_size = (num * 7) // 10

    rng = np.random.default_rng()
    samples = rng.choice(arr, trainval_size, replace=False)

    for i in range(samples.size):
        print(samples[i])