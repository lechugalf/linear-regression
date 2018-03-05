import matplotlib.pyplot as plt

class ScatterPlot:
    """Scatter plot for graph linear regression"""

    def __init__(self, data, linreg):
        self.data = data

    def setup():
        axis = [min(data[:,0])-1, 
            max(data[:,0])+1, 
            min(data[:,1])-1, 
            max(data[:,1])+1]
        setx = np.linspace(axis[0], axis[1])
        plt.scatter(data[:,0], data[:,1])
        plt.plot(setx, linreg.modelFunction(setx))
        plt.axis(axis)
        

    def show():
        pass
