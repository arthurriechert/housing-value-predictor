import numpy as np
import load_data as ld

# Basic dot product to get model results
def compute_model(W, X, b):
    return np.dot(X, W) + b

# Compute mean squared error
def compute_cost(W, X, b, Y):
    
    # Store prediction to prevent redundant calculations
    predictions = compute_model(W, X, b)
    m = X.shape[0]

    # Compute MSE vectorized
    cost = 1/(2*m) * np.sum(np.square(predictions - Y))

    return cost

# Revise this version it is wrong
def compute_gradient(W, X, b, Y):

    # Old method
    """
    m = X.shape[0]
    predictions = compute_model(W, X, b)
    error = predictions - Y

    # compute gradients
    grad_W = 1/m * np.dot(X.T, error)
    grad_b = 1/m * np.sum(error)
    """
    
    m, n = X.shape
    grad_W = np.zeros(n)
    grad_b = 0

    for i in range(m):
        predictions = np.dot(X[i], W)
        
        predictions += b

        loss = predictions - Y[i]

        grad_b += loss

        for j in range(n):

            grad_W[j] += loss * X[i, j]

    grad_W /= m
    grad_b /= n 


    return grad_W, grad_b
    

def gradient_descent(W, X, b, Y, iters, alpha):

    dW = 0
    db = 0

    for i in range(iters):

        dW, db = compute_gradient(W, X, b, Y)
        cost = compute_cost(W, X, b, Y)

        W = W - alpha * dW
        b = b - alpha * db

        print(f"{i} -> Cost: {cost} | dW: {dW} | db: {db}")

    return W, b

def z_score_normalization(np_array):

    # Compute the mean
    mean = np.mean(np_array)

    # Compute the standard deviation
    std_deviation = np.std(np_array)

    # Compute normalized values
    normalized_array = (np_array - mean) / std_deviation

    return normalized_array, mean, std_deviation

def reverse_normalization(np_array, mean, std_deviation):
    return np_array * std_deviation + mean

# Basic test
def test_model():

    X_unorm, Y_unorm = ld.get_training_sets()

    X_train, X_mean, X_std = z_score_normalization(X_unorm)
    Y_train, Y_mean, Y_std = z_score_normalization(Y_unorm)

    # Save computing time by limiting amount of examples
    m_test = int(X_train.shape[0] / 100)

    # Initialize a W and b

    W_test = np.array([-51.83733287,   1.01983882,   2.96451753,   0.79474719])
    b_test = -24.45082487324418

    # Compute for test
    test_predictions = compute_model(W_test, X_train[:m_test, :], b_test)

    dW, db = compute_gradient(W_test, X_train, b_test, Y_train)


    print(f"""
          Computing for {m_test} training sets. 
          
          ######### TEST PREDICTIONS #########
          
          {test_predictions}

          
          ######### INPUT #########
          
          {X_train[:m_test, :]}

          
          ######### GRADIENTS #########

          W: {dW} b: {db}

          
          ######### GRADIENT DESCENT #########
          """)
    
    W_final, b_final = gradient_descent(W_test, X_train, b_test, Y_train, 1000, 1e-4)

    print(f'\033[32mOptimized at {W_final, b_final}\033[0m')

    while(True):

        print(f"""
              
              Type in 0 to quit or enter values to get a prediction.

              """)
        
        lat = float(input("What is the latitude?" ))

        if lat == 0:
            break

        long = float(input("What is the longitude? "))
        median_age = float(input("What is the median age of the housing block? "))
        total_rooms = float(input("What is the total number of rooms for this block? "))
        #  median_income = float(input("What is the median income for this area? "))


        inputs = np.array([long, lat, median_age, total_rooms])

        normalized_inputs, std, mean = z_score_normalization(inputs)

        print(f"INPUTS: {normalized_inputs}")

        prediction = compute_model(W_final, inputs, b_final)

        final_prediction = reverse_normalization(prediction, mean, std)

        print(f"The median home value should be: ${final_prediction}")

        """
        What is the latitude?-122.23
        What is the longitude? 37.88
        What is the median age of the housing block? 41.0
        What is the total number of rooms for this block? 880.0
        What is the median income for this area? 129.0
        """


test_model()
