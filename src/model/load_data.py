import pandas as pd
import numpy as np

# Use pandas to load data
def load_data ():

    return pd.read_csv("../../data/housing-data.csv") 

# Refer to README for parameter selection
def get_training_sets ():
    
    data = load_data()

    # Set number of training examples, m, and features, n
    m, n = data.shape[0], 8

    X_train = np.empty((m, n), dtype=np.float64)
    
    # Refer to README.md for features information
    X_train[:, 0] = data.longitude
    X_train[:, 1] = data.latitude
    X_train[:, 2] = X_train[:, 0] * X_train[:, 1]
    X_train[:, 3] = data.housing_median_age
    X_train[:, 4] = data.total_rooms
    X_train[:, 5] = data.median_income
    X_train[:, 6] = data.total_bedrooms
    X_train[:, 7] = data.population

    print(f"DATA: \n{X_train}")

    # Output will be house value
    Y_train = np.array(data.median_house_value, dtype=np.float32)

    return X_train, Y_train
