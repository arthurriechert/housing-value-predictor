import numpy as np
import model.load_data as ld
import os

def parameter_init(j):
    """
    Initialize random parameters

    Args:
        j (int): Number of features. 
    
    Returns:
        W (ndarray): Array of weights, shape of (,j)
        b (int): Scalar bias

    """

    # Add random noise
    W = 1e-1  * np.random.rand(j)
    b = 0

    return W, b


def compute_model(W, X, b):
    """
    Get predictions from the model

    Args:
        W (ndarray): Array of weights, shape (,n), where n is number of features
        X (ndarray): Array of training examples, shape (m, n), where m is number of training examples & n is number of features
        b (float): Scalar bias

    Returns:
        predictions (ndarray): Array of computations, shape (m, n), computed using dot product

    """
    
    predictions = np.dot(X, W) + b

    return predictions


def compute_cost(W, X, b, Y, lambda_):
    """
    Compute cost using Mean Squared Error

    Args:
        W (ndarray): Array of weights, shape (,n), where n is number of features
        X (ndarray): Array of training examples, shape(m, n), where m is number of training examples & n is number of features
        b (float): Scalar bias
        Y (ndarray): Array of results for training examples, shape(,m), where m is number of training examples
        lambda_ (float): Regularization parameter

    Returns:
        cost (float): Scalar MSE

    """

    # Get y_hat
    predictions = compute_model(W, X, b)
    
    # Get training examples
    m = X.shape[0]

    # Compute MSE vectorized and regularized
    cost = 1/(2*m) * np.sum(np.square(predictions - Y)) + (lambda_ / (2 * m)) * np.sum(np.square(W)) 

    return cost


# Find a way to vectorize gradient computation
def compute_gradient(W, X, b, Y, lambda_):
    """
    Use pre-calculated partial derivatives to determine the derivatives of the MSE

    Args:
        W (ndarray): Array of weights, shape(,n), where n is number of features
        X (ndarray): Array of training examples, shape(m, n), where m is number of training examples & n is number of features
        b (float): Scalar bias
        Y (ndarray): Array of results for training examples, shape(,m), where m is number of training examples
        lambda_ (float): Regularization parameter

    Returns:
        grad_W (ndarray): Array partial derivatives, shape (,n), where is number of features
        grad_b (float): Scalar partial derivative for bias
           
        """

    # m: training examples, n: features
    m, n = X.shape

    # Initialize placeholders for gradients
    grad_W = np.zeros(n)
    grad_b = 0

    # Iterate over each training example
    for i in range(m):

        # Obtain a batch of predictions 
        predictions = compute_model(W, X[i], b)

        # Compute loss here to avoid redundant computation
        loss = predictions - Y[i]

        # Compute partial derivatives
        grad_b += loss
        for j in range(n):
            grad_W[j] += loss * X[i, j]

    # Apply regularization
    grad_W += lambda_ / m * W

    # Get the averages at once
    grad_W /= m
    grad_b /= m 


    return grad_W, grad_b
    

def gradient_descent(W, X, b, Y, iters, alpha, lambda_):
    """
    Perform standard gradient descent

    Args:
        W (ndarray): Array of weights, shape(,n), where n is number of features
        X (ndarray): Array of training examples, shape(m, n), where m is number of training examples & n is number of features
        b (float): Scalar bias
        Y (ndarray): Array of results for training examples, shape(,m), where m is number of training examples
        iters: (int): Number of times to perform single round of gradient descent
        alpha: (float): Learning rate (usually less than 1)
        lambda_ (float): Regularization parameter    

    Returns:
        W (ndarray): Updated array of weights after gradient descent, shape(,n), where n is number of features
        b (float): Updated scalar b after gradient descent

    """

    # Placeholder values
    dW = 0
    db = 0

    # Perform number (iters) rounds of gradient descent
    for i in range(iters):

        # Get gradients
        dW, db = compute_gradient(W, X, b, Y, lambda_)

        # Get costs to track progress
        cost = compute_cost(W, X, b, Y, lambda_)

        # Apply gradients
        W = W - alpha * dW
        b = b - alpha * db

        # Print diagnostic information
        print(f"{i} -> Cost: {cost} | dW: {dW} | db: {db}")
    
    # After gradient descent save weights and bias for later
    np.save('weights.npy', W)
    np.save('bias.npy', b)

    return W, b

# Should experiment with other types of normalization
def z_score_normalization(np_array):
    """
    Use the Z-score to decrease range and avoid NaN values

    Args:
        np_array (ndarray): Any numpy array

    Returns:
        normalized_array (ndarray): Array normalized using Z-score
        mean (float): Mean of np_array
        std_dev (float): Standard deviation of np_array

    """


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

    # Print diagnostics
    print(f"MEAN: {mean}\nSTANDARD DEVIATION: {std_dev}")

    # Compute normalized values
    normalized_array = (np_array - mean) / std_dev

    return normalized_array, mean, std_dev

def reverse_normalization(np_array, mean, std_deviation):
    """
    Turn predictions back into realistic housing prices

    Args:
        np_array (ndarray): Any numpy array
        mean: (float): Scalar mean of training sets
        std_deviation: (float): Scalar of standard deviation of training sets

    Returns:
        unorm_np_array (ndarray): Numpy array, shape of np_array

    """

    # Do opposite of normalization
    unorm_np_array = np_array * std_deviation + mean
    
    return unorm_np_array

# Consider adding new tests
def test_model():
    """
    A basic test for performing gradient descent and getting predictions

    Steps:
        1) Initialize training and evaluation data 
            - Use 3/4 of training data for training & 1/4 for evaluation
            - Perform z-score normalization
        2) Print some basic predictions, etc
        3) Perform gradient descent
            - Basic CLI for selecting alpha, epochs, and norm vs. unorm data
        4) CLI for getting predictions on novel data
    """

    # Get training and evaluation data from csv
    X_unorm, Y_unorm = ld.get_training_sets()
    X_eval_unorm, Y_eval_unorm = ld.get_evaluation_data(X_unorm.shape[0])

    # Normalize training data
    X_train, X_mean, X_std = z_score_normalization(X_unorm)
    Y_train, Y_mean, Y_std = z_score_normalization(Y_unorm)

    # Normalize evaluation data
    X_eval,_,_ = z_score_normalization(X_eval_unorm)
    Y_eval,_,_ = z_score_normalization(Y_eval_unorm)

    # Print diagnostic information
    print(f"NORMALIZED DATA: {X_train}\nX_EVAL_DATA: {X_eval}\nY_EVAL_DATA: {Y_eval}")

    # Save computing time by limiting amount of examples
    m_test = int(X_train.shape[0] / 100)
    m, n = X_train.shape

    # Initialize random W (ndarray) and b (float)
    W_test, b_test = parameter_init(n)

    # Check if there are already stored weights and load them if so
    if os.path.exists('weights.npy') and os.path.exists('bias.npy'):
        W_test = np.load('weights.npy')
        b_test = np.load('bias.npy')

    # Compute test predictions
    test_predictions = compute_model(W_test, X_train[:m_test, :], b_test)

    # Compute test gradients
    dW, db = compute_gradient(W_test, X_train, b_test, Y_train, 1.0)

    # Print diagnostics
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
    
    # Start Gradient Descent
    running = True
    while(running):

        try:

            # Get settings for gradient descent
            epochs = int(input("How many epochs? "))
            alpha = float(input("What's alpha? "))
            should_norm = input("Use normalized or unnormalized data(yes or no)? ")
            lambda_ = float(input("What's lambda? "))

            # Determine use of normalized or unnormalized data
            if should_norm == "yes":
                W_final, b_final = gradient_descent(W_test, X_train, b_test, Y_train, epochs, alpha, lambda_)    
            elif should_norm == "no":
                W_final, b_final = gradient_descent(W_test, X_unorm, b_test, Y_unorm, epochs, alpha, lambda_)    
 
            # Stop loop
            running = False
        
        # If user uses ctrl + C prompt to see if they want to exit in middle of gradient descent
        except Exception as e:
            print(f"ERROR: {e}")
            quit_ = input("Would you like to quit? ")

            # End loop
            if quit_ == "yes":
                break

            continue

    # Compute cost against evaluation data
    final_cost = compute_cost(W_final, X_eval, b_final, Y_eval, 1.0)

    # Display final W (ndarray) and b (float)
    print(f'\033[32mOptimized at {W_final, b_final}\033[0m\nCOST AGAINST EVALUATION DATA:{final_cost}')

    # Start CLI to get novel predictions
    while(True):

        print(f"""
              
              Type in 0 to quit or enter values to get a prediction.

              """)

        # Get parameters
        lat = float(input("What is the latitude?" ))

        # Exit program
        if lat == 0:
            break

        long = float(input("What is the longitude? "))
        median_age = float(input("What is the median age of the housing block? "))
        total_rooms = float(input("What is the total number of rooms for this block? "))
        median_income = float(input("What is the median income for this area? "))
        total_bedrooms = float(input("What is the total number of bedrooms? "))
        population = float(input("What is the total population? "))

        # Format the inputs
        inputs = np.array([long, lat, lat*long,  median_age, total_rooms, median_income, total_bedrooms, population])

        # Normalize inputs
        normalized_inputs, std, mean = z_score_normalization(inputs)

        # Get novel predictions
        prediction = compute_model(W_final, inputs, b_final, 1.0)

        # Convert it to a realistic house price
        final_prediction = reverse_normalization(prediction, mean, std)

        # Display the prediction
        print(f"The median home value should be: ${final_prediction}")

