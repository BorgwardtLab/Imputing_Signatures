import argparse 
import numpy as np
import os
from IPython import embed
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from dataset import get_ucr_dataset
from preprocessing import subsample
from interpolation import gp_interpolation, interpolate_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', 
                    default='../data/UCRArchive_2018',
                    help='Path to input data directory')
parser.add_argument('--output_path', 
                    default='../data/processed',
                    help='Path to processed data directory')
parser.add_argument('--dataset', 
                    default=10,
                    help='Dataset index pointing to one of 128 UCR datasets')
parser.add_argument('--thres', 
                    default=0.5, type=float, 
                    help='Threshold for subsampling. Probability of dropping an observation')
parser.add_argument('--interpol', 
                    default='GP',
                    help='Interpolation Scheme after subsampling: [GP, linear] ')

def main():
    # Parse Arguments and unpack the args:
    args = parser.parse_args()
    input_path = args.input_path # path to UCR datasets 
    dataset_index = args.dataset # index, to currently used UCR dataset
    thres = args.thres # threshold for subsampling
    interpol = args.interpol # interpolation scheme

    datasets = os.listdir(input_path) #list of all available UCR datasets

    #Load current UCR dataset
    X_train, y_train, X_test, y_test = get_ucr_dataset(input_path, datasets[dataset_index]) #'ItalyPowerDemand')

    np.random.seed(42)
    
    X_int = interpolate_dataset(X_train, thres, interpolation_type=interpol, plot_sample=4)
    embed()


#    #select sample
#    i = 4
#    T_max = len(X_train[i,:])
#    T_grid = np.arange(T_max)
#    T = T_grid.reshape(-1,1) #np.atleast_2d(T_grid).T
#    embed()
#    X = X_train[i,:]
#    T_old, X_old = T, X
#    T,X = subsample(T,X,thres)
#
#
#    ## Apply GP interpolation to current sample:
#    X_pred, T_pred, sigma = gp_interpolation(X, T, T_max)
#    embed()
#
#    # Plot the function, the prediction and the 95% confidence interval based on
#    # the MSE
#    plt.figure()
#    plt.plot(T,X, 'o', label='Subsampled and observed')
#    plt.plot(T_old,X_old, '.', label='Original training points')
#    plt.plot(T_pred, X_pred, 'b-', label='Prediction')
#    plt.fill(np.concatenate([T_pred, T_pred[::-1]]),
#             np.concatenate([X_pred - 1.9600 * sigma,
#                            (X_pred + 1.9600 * sigma)[::-1]]),
#             alpha=.5, fc='b', ec='None', label='95% confidence interval')
#    plt.xlabel('$x$')
#    plt.ylabel('$f(x)$')
#    #plt.ylim(-10, 20)
#    plt.legend(loc='upper left')
#
#    plt.show()
#

if __name__ in "__main__":
    main()

