import numpy as np
import load_data as ld
import os

# Initialize Random Weights
def parameter_init(n):
    W = 1e-1  * np.random.rand(n)
    b = 0
    return W, b

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
    grad_b /= m 


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
    
    np.save('weights.npy', W)
    np.save('bias.npy', b)

    return W, b

def z_score_normalization(np_array):
   # Check if np_array is 1-D or 2-D
    if np_array.ndim == 1:
        # For 1-D array, calculate the mean and standard deviation of the entire array
        mean = np.nanmean(np_array)
        std_dev = np.nanstd(np_array)
        # If np_array contains any NaN values, replace them with the mean
        np_array = np.where(np.isnan(np_array), mean, np_array)
    else:
        # For 2-D array, calculate the mean and standard deviation of each column
        mean = np.nanmean(np_array, axis=0)
        std_dev = np.nanstd(np_array, axis=0)
        # If np_array contains any NaN values, replace them with the mean of their column
        inds = np.where(np.isnan(np_array))
        np_array[inds] = np.take(mean, inds[1]) 
    print(f"MEAN: {mean}\nSTANDARD DEVIATION: {std_dev}")

    # Compute normalized values
    normalized_array = (np_array - mean) / std_dev

    return normalized_array, mean, std_dev

def reverse_normalization(np_array, mean, std_deviation):
    return np_array * std_deviation + mean

# Basic test
def test_model():

    X_unorm, Y_unorm = ld.get_training_sets()

    X_train, X_mean, X_std = z_score_normalization(X_unorm)
    Y_train, Y_mean, Y_std = z_score_normalization(Y_unorm)


    print(f"NORMALIZED DATA: {X_train}")

    # Save computing time by limiting amount of examples
    m_test = int(X_train.shape[0] / 100)
    m, n = X_train.shape

    # Initialize a W and b

    W_test, b_test = parameter_init(n)

    if os.path.exists('weights.npy') and os.path.exists('bias.npy'):
        W_test = np.load('weights.npy')
        b_test = np.load('bias.npy')
    # else:
      #  W_test, b_test = gradient_descent(W_test, X_train, b_test, Y_train, 1000, 1e-4)
      #  print(f'Optimized at {W_test, b_test}')

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
    running = True

    while(running):
        try:
            epochs = int(input("How many epochs? "))
            alpha = float(input("What's alpha? "))
            should_norm = input("Use normalized or unnormalized data(yes or no)? ")

            if should_norm == "yes":
                W_final, b_final = gradient_descent(W_test, X_train, b_test, Y_train, epochs, alpha)    
            elif  should_norm == "no":
                W_final, b_final = gradient_descent(W_test, X_unorm, b_test, Y_unorm, epochs, alpha)    
 

            running = False

        except:
            quit_ = input("Would you like to quit? ")
            if quit_ == "yes":
                break

            continue

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
        median_income = float(input("What is the median income for this area? "))
        total_bedrooms = float(input("What is the total number of bedrooms? "))
        population = float(input("What is the total population? "))

        inputs = np.array([long, lat, lat*long,  median_age, total_rooms, median_income, total_bedrooms, population])

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
