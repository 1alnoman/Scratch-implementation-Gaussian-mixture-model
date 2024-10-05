import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from gausian_mixture_model import GaussianMixtureModel

def plot_gmm(gmm,data,mu,sigma,pi,iterations):
    if data.shape[1] != 2:
        print('Error: can only plot 2D data')
        exit()

    min_x,max_x = np.min(data[:,0]),np.max(data[:,0])
    min_y,max_y = np.min(data[:,1]),np.max(data[:,1])

    x = np.linspace(min_x-2,max_x+2,100)
    y = np.linspace(min_y-2,max_y+2,100)

    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    non,Z = gmm.log_likelihood(XX, mu, sigma, pi)
    Z = -Z

    Z = Z.reshape(X.shape)

    CS = plt.contour(
        X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10)
    )
    CB = plt.colorbar(CS, shrink=0.8, extend="both")

    plt.scatter(data[:, 0], data[:, 1], s=0.8, color='red')
    plt.scatter(mu[:, 0], mu[:, 1], s=1000, color='green',marker='+')

    plt.title("Iteration: %d" % iterations)
    plt.axis("tight")

    plt.draw()



# main
if __name__ == "__main__":
    # load data
    X = np.loadtxt("./data2D.txt")

    params = {'num_components': 3, 'max_iter': 300, 'tolerance': 1e-5, 'init_param': 'random_points'}
    gmm = GaussianMixtureModel(params)

    gmm.fit(X)

    plt.ion()
    for i in range(0,gmm.convrged_at_iter):
        plt.cla()
        plt.clf()
        plot_gmm(gmm,X,gmm.means_history[i],gmm.covariances_history[i],gmm.weights_history[i],i)
        plt.pause(0.1)

    plt.ioff()
    plt.show()

