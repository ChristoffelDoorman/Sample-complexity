import matplotlib.pyplot as plt
import numpy as np

# linear, logarithmic and exponential functions
lin_func = lambda x, a, b: a*x + b
log_func = lambda x, a, b, c: a * np.log(b*x) + c
exp_func = lambda x, a, b: a*np.exp(b*x)


def generate_data(m, n):
    """
    Generate a random dataset of size m,n
    
    m: int -- number of samples
    n: int -- dimension of samples
    
    Returns:
    X: (m,n)-array -- dataset
    Y: (m,)-array -- labels
    
    """

    X = np.ones((m,n), dtype=int)*np.random.choice([-1,1],(m,n))
        
    # labels are first column of patterns
    Y = X[:,0]
    
    return X, Y


def plot_sample_complexity(N, M, algorithm_name):
    """
    Plot the sample complexity m vs. dimension n for given algorithm.
    
    N: int -- maximum dimension
    M: np.array(N,) -- sample complexity for every dimension
    algorithm_name: str -- name of the algorithm to be displayed in the title
    """
        
    plt.figure()
    plt.plot(range(1, N+1), M)
    plt.xlabel('Dimension n')
    plt.ylabel('Sample complexity m')
    plt.title(algorithm_name)
    
def plot_sample_complexity_ax(ax, N, M, algorithm_name):
    """
    Plot the sample complexity m vs. dimension n for given algorithm.
    Can be used in a for loop.
    
    ax: -- axis to plot on
    N: int -- maximum dimension
    M: np.array(N,) -- sample complexity for every dimension
    algorithm_name: str -- name of the algorithm to be displayed in the title
    
    Returns the axis with plot and info
    """
        
    ax.plot(range(1, N+1), M)
    ax.set_title(algorithm_name, fontsize=20)
    ax.tick_params(axis='both', labelsize=16)
    
    return ax


def plot_sample_complexity_ax2(ax, N, M, algorithm_name):
    """
    Plot the sample complexity m vs. dimension n for given algorithm.
    Can be used in a for loop.
    
    N: int -- maximum dimension
    M: np.array(N,) -- sample complexity for every dimension
    algorithm_name: str -- name of the algorithm to be displayed in the title
    
    Returns the axis with plot and info
    """
        
    ax.plot(range(1, N+1), M, label=algorithm_name)
    ax.tick_params(axis='both', labelsize=16)
    
    return ax