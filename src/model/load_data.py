import pandas as pd
import numpy as np

def load_data ():
    """
    Retrieve data from CSV using Pandas

    Returns:
        data (DataFrame): Return pandas dataframe from CSV
    
    """
    
    # Read CSV
    data = pd.read_csv("../../data/housing-data.csv") 

    return data 

# Refer to README for parameter selection
def get_training_sets ():
    """
    Organize training sets based off of model parameters

    Returns:
        X_train (ndarray): Training independent varaibles set, shape (m_limit, n), where m_limit is number of examples and n is number of features
        Y_train (ndarray): Training dependent variables set, shape (,m_limit), where m_limit is number of examples

    """

    # Get data
    data = load_data()

    # Set number of training examples, m, and features, n
    m, n = data.shape[0], 8

    # Initialize placeholder array
    X_train = np.empty((m, n), dtype=np.float64)
    
    # Organize features
    X_train[:, 0] = data.longitude
    X_train[:, 1] = data.latitude
    X_train[:, 2] = X_train[:, 0] * X_train[:, 1]
    X_train[:, 3] = data.housing_median_age
    X_train[:, 4] = data.total_rooms
    X_train[:, 5] = data.median_income
    X_train[:, 6] = data.total_bedrooms
    X_train[:, 7] = data.population

    # Print diagnostics
    print(f"DATA: \n{X_train}")

    # Save some data for evaluation
    m_limit = 15000

    # Output will be house value
    Y_train = np.array(data.median_house_value, dtype=np.float32)

    return X_train[m_limit:,:], Y_train[m_limit:]

def get_evaluation_data (m_limit):
    """
    Organize data that model will not be trained for and will be used for evaluation

    Returns:
        X_train (ndarray): Evaluation data set of shape (m_limit, n), where m_limit is number of examples and n is number of features
        Y_train (ndarray): Evaluation dependent variables of shape (m_limit), where m_limit is number of examples

    """

    # Get data
    data = load_data()

    # Set number of training examples, m, and features, n
    m, n = data.shape[0], 8

    # Initialize placeholder
    X_train = np.empty((m, n), dtype=np.float64)
    
    # Organize features
    X_train[:, 0] = data.longitude
    X_train[:, 1] = data.latitude
    X_train[:, 2] = X_train[:, 0] * X_train[:, 1]
    X_train[:, 3] = data.housing_median_age
    X_train[:, 4] = data.total_rooms
    X_train[:, 5] = data.median_income
    X_train[:, 6] = data.total_bedrooms
    X_train[:, 7] = data.population

    # Print diagnostics
    print(f"DATA: \n{X_train}")

    # Save some data for evaluation
    m_limit = m - m_limit

    # Output will be house value
    Y_train = np.array(data.median_house_value, dtype=np.float32)

    return X_train[:m_limit,:], Y_train[:m_limit]
