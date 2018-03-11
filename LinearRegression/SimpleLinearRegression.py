import numpy as np
import matplotlib as plt

class LinearRegression:
    """Linear Regression by gradient descent"""

    def __init__(self, tdata, name='unknown'):
        self.name = name
        self.tdata = tdata
        self.param = [0, 0]
        self.lrate = 0.001
        self.N = len(tdata)
        self.training_cost = []

    def describeModel(self):
        print "Model '{}'".format(self.name)
        print "label x: " + ", ".join(map(str, self.tdata[:, 0]))
        print "label y: " + ", ".join(map(str, self.tdata[:, 1]))
        print "Parameters: {}, {}".format(self.param[0], self.param[1])
        print "Cost: {}".format(self.costFunction())
        print "Learning Rate: {}".format(self.lrate)

    def modelFunction(self, xval):
        return self.param[0] + (self.param[1] * xval)

    def costFunction(self, x=None, y=None):
        if (x != None) and (y != None):
            self.param[0] = x
            self.param[1] = y
        cost = np.sum((self.modelFunction(self.tdata[:, 0]) - self.tdata[:, 1])**2)
        return (cost / (2 * self.N))

    def gradientDescent(self):
        t1, t2 = 0, 0
        t1 = np.sum(self.modelFunction(self.tdata[:, 0]) - self.tdata[:, 1])
        t1 = self.lrate * (t1 / self.N)
        t2 = np.sum((self.modelFunction(self.tdata[:, 0]) - self.tdata[:, 1]) * self.tdata[:, 0])
        t2 = self.lrate * (t2 / self.N)
        self.param[0] -= t1
        self.param[1] -= t2

    def training(self, times, lrate=None):
        self.training_cost = []
        if (lrate != None):
            self.lrate = lrate
        for step in range(0, times):
            #result training cost
            self.training_cost.append(self.costFunction())
            #descent step
            self.gradientDescent()
            if (step % (times / 4) == 0):
                print self.costFunction()
        print 'training done...'

    def scatterPlot(self):
        axis = [min(self.tdata[:,0])-1, 
                max(self.tdata[:,0])+1, 
                min(self.tdata[:,1])-1, 
                max(self.tdata[:,1])+1]

        setx = np.linspace(axis[0], axis[1])
        plt.scatter(self.tdata[:,0], self.tdata[:,1])
        plt.plot(setx, self.modelFunction(setx))
        plt.axis(axis)
        plt.show()
