import numpy as np
import load_data as ld

# Basic dot product to get model results
def compute_model(W, X, b):
    return np.dot(X, W) + b

def test_model():

    X_train, Y_train = ld.get_training_sets()

    # Save computing time by limiting amount of examples
    m_test = int(X_train.shape[0] / 100)

    # Initialize a W and b
    W_test = np.array([-50, 5, 3, 4, 5, 6])
    b_test = 23

    # Compute for test
    test_predictions = compute_model(W_test, X_train[m_test, :], b_test)

    print(f"Computing for {m_test} training sets.\n\n######### TEST PREDICTIONS #########\n{test_predictions}\n\n######### INPUT #########\n{X_train[[i for i in range(m_test)], :]}")

test_model()
