import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import pinv
from scipy.optimize import curve_fit as fit
from tqdm import tqdm

from algorithms import perceptron, winnow, least_squares, one_nearest_neighbors
from helper import *


def sample_complexity(algorithm, N, max_error=0.1):
    """
    Calculate the sample complexity on average at given generalisation error.
    
    algorithm: least squares, perceptron, winnow or 1-NN
    N: max dimension
    max_error: generalisation error
    
    Returns the sample complexity for every dimension n.
    """
    
    # for every dimension n, find m
    M = np.zeros(N)
    
    # loop over all dimensions n
    for n in tqdm(range(1, N+1)):
        
        m = 1

        while True:
            
            error = 0
            for i in range(n):

                # generate training data
                X, Y = generate_data(m, n)

                # generate test example
                X_test, Y_test = generate_data(m, n)

                pred = algorithm(X, Y, X_test)
                error += np.sum(pred != Y_test)
        
            if error/(m*n) <= max_error:
                M[n-1] = m
                break
        
            m += 1
                
    return M


def __main__():
    # least squares algorithm
    N_ls = 100
    M_ls = sample_complexity(least_squares, N_ls, 0.1)

    # perceptron algorithm
    N_p = 100
    M_p = sample_complexity(perceptron, N_p, 0.1)

    # Winnow algorithm
    N_w = 150
    M_w = sample_complexity(winnow, N_w, 0.1)

    # 1-NN algorithm
    N_nn = 15
    M_nn = sample_complexity(one_nearest_neighbors, N_nn, 0.1)

    
    ########## plot all algorithms in 2,2 matrix plot #######
    M = [M_ls, M_p, M_w, M_nn]
    N = [N_ls, N_p, N_w, N_nn]
    name = ['Least squares', 'Perceptron', 'Winnow', '1-NN']

    fig, ax = plt.subplots(2,2, figsize=(14,14))

    i=0
    for row in range(ax.shape[0]):
        for col in range(ax.shape[1]):
            ax[row,col] = plot_sample_complexity_ax(ax[row,col], N[i], M[i], name[i])
            ax[1,col].set_xlabel('Dimension n', fontsize=18)
            i += 1

        ax[row,0].set_ylabel('Sample complexity m', fontsize=18)
        
    ######### Fit linear, log and exp functions ##########
    M = [M_ls, M_p, M_w, M_nn]
    N = [N_ls, N_p, N_w, N_nn]

    lin_fits = []
    log_fits = []
    exp_fits = []

    lin_popts = []
    log_popts = []
    exp_popts = []

    log_start = 1

    for i, m in enumerate(M):
        n = np.arange(1,N[i]+1)
        popt_lin, pcov_lin = fit(lin_func, n, m, bounds=([0, -5],[2, 5]))
        popt_log, pcov_log = fit(log_func, n, m, bounds=([-np.inf, 0, -np.inf],[np.inf, np.inf, np.inf]))
        popt_exp, pcov_exp = fit(exp_func, n, m)

        lin_popts.append(popt_lin)
        log_popts.append(popt_log)
        exp_popts.append(popt_exp)

        lin_fits.append(lin_func(n, *popt_lin))
        log_fits.append(log_func(n, *popt_log))
        exp_fits.append(exp_func(n, *popt_exp))


    ############# plot all fits in 2,2 matrix plot #########
    fig, ax = plt.subplots(2,2, figsize=(14,14))

    i=0
    for row in range(ax.shape[0]):
        for col in range(ax.shape[1]):
            n = np.arange(1,N[i]+1)
            ax[row,col] = plot_sample_complexity_ax(ax[row,col], N[i], M[i], name[i])
            ax[row,col].plot(n, lin_fits[i], linestyle='--', label=(r'$f(n) = %.1f n + %.1f$'%(lin_popts[i][0], lin_popts[i][1])))
            ax[row,col].plot(n, log_fits[i], linestyle='dotted', label=r'$f(n) = %.1f\cdot log(%.1f n) + %.1f$'%(log_popts[i][0], log_popts[i][1], log_popts[i][2]))
            ax[row,col].plot(n, exp_fits[i], linestyle='dashdot', label=r'$f(n) = %.1f\cdot exp(%.1f n)$'%(exp_popts[i][0], exp_popts[i][1]))
            ax[row,col].set_ylim([0,max(M[i])+3])
            ax[row,col].legend()
            ax[1,col].set_xlabel('Dimension n', fontsize=18)
            i += 1

        ax[row,0].set_ylabel('Sample complexity m', fontsize=18)
    
if __name__ == "__main__":
    __main__()
        