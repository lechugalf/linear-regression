#!/usr/bin/python

import numpy as np

from SimpleLinearRegression import *
import matplotlib.pyplot as plt

def main():
    #load dataset
    data = np.genfromtxt("../datasets/dataset5.csv", delimiter=",")

    #create model
    linreg = LinearRegression(data, 'test')
    linreg.param = [1.0, 1.0]
    linreg.describeModel()

    #training model
    epochs = 50
    linreg.training(epochs, 0.0000003)

    #plot data with result line
    axis = [min(data[:,0])-1, 
            max(data[:,0])+1, 
            min(data[:,1])-1, 
            max(data[:,1])+1]
    setx = np.linspace(axis[0], axis[1])
    plt.scatter(data[:,0], data[:,1])
    plt.plot(setx, linreg.modelFunction(setx))
    plt.axis(axis)
    plt.show()

    #plot training cost function
    plt.plot(np.arange(0, epochs), linreg.training_cost)

    #plt.axis([0, epochs/2, 0, linreg.training_cost[0] / 2])
    plt.show()

if __name__ == '__main__':
    main()
