import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 3 # Reduced number of repetitions for efficiency

# Function to create fake data
def generate_data(N, P, data_type='binary', output_type='discrete'):
    if data_type == 'binary':
        X = pd.DataFrame(np.random.randint(2, size=(N, P)), columns=[f'feature_{i}' for i in range(P)])
    else:
        X = pd.DataFrame(np.random.randn(N, P), columns=[f'feature_{i}' for i in range(P)])

    if output_type == 'discrete':
        y = pd.Series(np.random.randint(2, size=N), dtype='category')
    else:
        y = pd.Series(np.random.randn(N))

    return X, y

def measure_runtime(N_values, P_values, data_type, output_type):
    avg_train_times = []
    avg_predict_times = []

    for N in N_values:
        for P in P_values:
            train_times = []
            predict_times = []

            for _ in range(num_average_time):
                # Generate data
                X, y = generate_data(N, P, data_type=data_type, output_type=output_type)
                tree = DecisionTree(criterion="information_gain", max_depth=5)

                # Measure training time
                start_time = time.time()
                tree.fit(X, y)
                train_times.append(time.time() - start_time)

                # Measure prediction time
                start_time = time.time()
                tree.predict(X)
                predict_times.append(time.time() - start_time)

            avg_train_times.append(np.mean(train_times))
            avg_predict_times.append(np.mean(predict_times))

    return avg_train_times, avg_predict_times

# Function to plot the results
def plot_results(N_values, P_values, train_times, predict_times, case_description):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Runtime Complexity for {case_description}", fontsize=16)

    # Plot training time
    ax1.set_title("Training Time vs N")
    for P in P_values:
        ax1.plot(N_values, train_times[:len(N_values)], label=f'P={P}')
        train_times = train_times[len(N_values):]  
    ax1.set_xlabel("Number of Samples (N)")
    ax1.set_ylabel("Training Time (seconds)")
    ax1.legend()

    # Plot prediction time
    ax2.set_title("Prediction Time vs N")
    for P in P_values:
        ax2.plot(N_values, predict_times[:len(N_values)], label=f'P={P}')
        predict_times = predict_times[len(N_values):]  
    ax2.set_xlabel("Number of Samples (N)")
    ax2.set_ylabel("Prediction Time (seconds)")
    ax2.legend()

    plt.tight_layout()
    plt.show()

N_values = [100, 500, 1000] 
P_values = [5, 10]  

# Run experiments for all four cases
cases = [
    ('binary', 'discrete', "Discrete Input, Discrete Output"),
    ('binary', 'real', "Discrete Input, Real Output"),
    ('real', 'discrete', "Real Input, Discrete Output"),
    ('real', 'real', "Real Input, Real Output")
]

for data_type, output_type, description in cases:
    print(f"Running for case: {description}")
    avg_train_times, avg_predict_times = measure_runtime(N_values, P_values, data_type, output_type)
    plot_results(N_values, P_values, avg_train_times, avg_predict_times, description)
