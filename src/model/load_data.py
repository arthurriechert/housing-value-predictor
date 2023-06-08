import pandas as pd
import numpy as np

def load_data ():
    """
    Retrieve data from CSV using Pandas

    Returns:
        data (DataFrame): Return pandas dataframe from CSV
    
    """
    
    # Read CSV
    data = pd.read_csv("../data/housing-data.csv") 

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
    
    # Organize features
    data["longitude x latitude"] = data["longitude"] * data["latitude"]
    X_train = data[["longitude", "latitude", "longitude x latitude", "housing_median_age", "total_rooms", "median_income", "total_bedrooms", "population"]].to_numpy()

    # Save some data for evaluation
    m_limit = 15000

    # Output will be house value
    Y_train = data["median_house_value"].to_numpy()

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

     # Organize features
    data["longitude x latitude"] = data["longitude"] * data["latitude"]
    X_train = data[["longitude", "latitude", "longitude x latitude", "housing_median_age", "total_rooms", "median_income", "total_bedrooms", "population"]].to_numpy()

    # Output will be house value
    Y_train = data["median_house_value"].to_numpy()

    print(f"DATA: \n{X_train}")

    # Save some data for evaluation
    m_limit = m - m_limit

    return X_train[:m_limit,:], Y_train[:m_limit]
