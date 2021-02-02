import numpy as np
from numpy.linalg import pinv

def perceptron(X, Y, X_test):
    """
    Perceptron algorithm
    """
    M, N = X.shape
    
    w = np.zeros(N)
    error = 1
    
    while error:
        mistakes = 0
        
        # train weights until convergence
        for m in range(M):
            x = X[m,:]
            y = Y[m]

            # if misclassified, update weights
            if 2*(np.dot(x, w) > 0) - 1 != y:            
                w += y*x
                mistakes += 1
        
        # calculate error
        error = mistakes/M
    
    # return prediction
    pred = 2*(np.matmul(X_test, w) > 0) - 1
    
    return pred

def winnow(X, Y, X_test):
    """
    Winnow algorithm
    """
    X = (X + 1)/2
    Y = (Y + 1)/2
    X_test = (X_test + 1)/2
    
    M, N = X.shape
    
    w = np.ones(N)
    error = 1
    
    while error:
        mistakes = 0
        
        # train weights until convergence
        for m in range(M):
            x = X[m,:]
            y = Y[m]

            # predict
            pred = (np.dot(x, w) - N >= 0)
            # if misclassified, update weights
            if pred != y:            
                w *= np.power(2, (y - pred) * x)
                mistakes += 1
        
        # calculate error
        error = mistakes/M
    
    # return prediction
    pred = 2*(np.matmul(X_test, w) - N >= 0) - 1

    return pred
    
        

def least_squares(X, Y, X_test):
    """
    Least squares algorithm
    """
    
    # train weight vector
    w = np.matmul(pinv(X), Y)
    
    # classify test points
    pred = 2 * (np.matmul(X_test, w) > 0) - 1
    
    return pred


def one_nearest_neighbors(X, Y, X_test):
    """
    1 nearest neighbour algorithm
    """
    M, N = X_test.shape
    
    # calculate Eucledian distance between a(m,n) and b(m,n)
    eucl_dist = lambda a, b: np.sqrt(np.sum((a-b)**2, axis=1))
    
    # calculate all distances between test and training points
    dist = np.array([eucl_dist(x_test, X) for x_test in X_test])
    
    # get indexi of smallest distances
    nn_idx = np.argmin(dist, axis=1)

    # assign to class of nearest neighbor
    pred = Y[nn_idx]
    
    return pred
