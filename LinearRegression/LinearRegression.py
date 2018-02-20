import numpy as np

class LinearRegression:
    """Multivaribale Linear Regression by gradient descent"""

    def __init__(self, tdata, nvar, name='unknown'):
        self.name  = name
        self.N = len(tdata)
        self.nvar  = nvar+1
        #formating data
        x0 = np.ones((self.N, 1)) 
        self.X = np.concatenate((x0, tdata[:,:-1]), axis=1)
        self.y = tdata[:,-1]
        self.param = [0.0] * self.nvar
        self.lrate = 0.001
        #training data !-- check --!
        self.Tp = np.zeros((self.nvar, 1))
        self.Tcf = []

    def describeModel(self):
        print "Model '{}'".format(self.name)
        for x in range(0, self.nvar):
            print 'label x' + x + ':' + ', '.join(map(str, self.X[:, x]))
        print 'label y: ' + ', '.join(map(str, self.y))
        print 'Parameters: ' + self.param
        print 'Cost: {}'.format(self.costFunction())
        print 'Learning Rate: {}'.format(self.lrate)

    def modelFunction(self, xval):
        return self.param.transpose() * self.X

    def costFunction(self):
        self.param = np.linalg.inv(self.X.transpose() * self.X) * self.X.transpose() * self.y

    def gradientDescent(self):
        #multiple variable
        t = [0] * self.nvar
        for v in range(self.nvar):
            t[v] = np.sum((self.modelFunction(self.X) - self.y) * self.X[:, v])
            t[v] = self.lrate * (t[v] / self.N)
        self.param = t

    def training(self, times, lrate=None):
        self.Tx = np.zeros((self.nvar, 1))
        self.Tcf = []
        if (lrate != None):
            self.lrate = lrate

        for step in range(0, times):
            #result training data
            self.Tx = np.concatenate((self.Tx, self.param), axis=1)
            self.Tcf.append(self.costFunction())
            #descent step
            self.gradientDescent()
            if (step % (times / 4) == 0):
                print self.costFunction()
        print 'training done...'