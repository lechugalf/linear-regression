import numpy as np

class LinearRegression:
    """Multivaribale Linear Regression by gradient descent"""

    def __init__(self, tdata, nvar, name='unknown'):
        self.name = name
        self.N = len(tdata)
        self.nvar = nvar+1

        #formating data
        x0 = np.ones((self.N, 1)) 
        self.X = np.asmatrix(np.concatenate((x0, tdata[:,:-1]), axis=1))
        self.y = np.asmatrix(tdata[:,-1]).transpose()
        self.param = np.matrix([5.0] * self.nvar).T
        self.lrate = 0.001

    def describeModel(self):
        print "Model '{}'".format(self.name)
        #missing print x elements
        print 'label y: ' + ', '.join(map(str, self.y))
        print 'Parameters: ' + str(self.param)
        print 'Cost: {}'.format(self.costFunction())
        print 'Learning Rate: {}'.format(self.lrate)

    def modelFunction(self, xval):
        return np.multiply(self.param.T, xval)
        #return self.param.transpose() * self.X

    def costFunction(self):
        cf = np.sum(np.square(np.subtract(self.modelFunction(self.X), self.y)))
        #cf = np.sum(((self.modelFunction(self.X) - self.y))**2)
        return (cf / (2*self.N))

        #cost function for normal equation
        #self.param = np.linalg.inv(self.X.transpose() * self.X) * self.X.transpose() * self.y

    def gradientDescent(self):
        t = np.matrix([1.0] * self.nvar).T
        for p in range(self.nvar):
            t[p] = np.sum(np.multiply((np.subtract(self.modelFunction(self.X), self.y)), self.X[:, p]))
            t[p] = self.lrate * (t[p] / self.N)
        self.param -= t
        #self.param = np.subtract(self.param, t)

    def training(self, times, lrate=None):
        if (lrate != None):
            self.lrate = lrate

        for step in range(0, times):
            #descent step
            self.gradientDescent()
            if (step % (times / 4) == 0):
                print 'Parameters: ' + str(self.param)
                print self.costFunction()

        print 'training done...'

    def graphResults():
        axis = [min(X), max(X), min(y), max(y)]
        setx = np.linspace(axis[0], axis[1])

        for v in range(1, nvar):
            plt.scatter(X[v], y)

        plt.plot(setx, linreg.modelFunction(setx))
        plt.axis(axis)
        plt.show()