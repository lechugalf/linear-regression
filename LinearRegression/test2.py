#!/usr/bin/python

import numpy as np

from LinearRegression import *
import matplotlib.pyplot as plt

def main():
    #load dataset
    data = np.genfromtxt("../datasets/mdataset.csv", delimiter=",")

    #create model
    linreg = LinearRegression(data, 2, 'test')
    linreg.describeModel()

    #training model
    epochs = 60
    linreg.training(epochs, 0.001)


    ##plot data with result lines
    plt.figure(1)


    axis1 = [min(data[:,0]), max(data[:,0]), min(data[:,2]), max(data[:,2])]
    axis2 = [min(data[:,1]), max(data[:,1]), min(data[:,2]), max(data[:,2])]
    axis = [min([axis1[0], axis2[0]]), max([axis1[1], axis2[1]]), 
            min([axis1[2], axis2[2]]), max([axis1[3], axis2[3]])]

    setx = np.asmatrix(np.linspace(axis[0], axis[1])).T
    x0 = np.ones((setx.size, 1)) 
    x = np.concatenate((x0, setx, setx), axis=1)

    plt.scatter(data[:,0], data[:,2])
    plt.scatter(data[:,1], data[:,2])
    plt.plot(setx, linreg.modelFunction(x))
    plt.axis(axis)

    #plt.subplot(212)
    
    #setx = np.linspace(axis[0], axis[1])
    
    #plt.plot(setx, linreg.modelFunction(setx))
    #plt.axis(axis)

    plt.show()

if __name__ == '__main__':
    main()
