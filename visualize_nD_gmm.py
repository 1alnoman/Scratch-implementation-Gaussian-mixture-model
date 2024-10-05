import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from gausian_mixture_model import GaussianMixtureModel

from sklearn.decomposition import PCA




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

    Z = (-Z).reshape(X.shape)
    Z[np.where(Z<=0)] = 1e-10

    CS = plt.contour(
        X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10)
    )
    CB = plt.colorbar(CS, shrink=0.8, extend="both")

    plt.scatter(data[:, 0], data[:, 1], s=0.8)
    plt.scatter(mu[:, 0], mu[:, 1], s=1000, color='red',marker='+')

    plt.title("Iteration: %d" % iterations)
    plt.axis("tight")

    plt.draw()

def predict_cluster(data,gmm,means,covariances,weights):
    gamma_i = gmm.expectaion_step(data,means,covariances,weights)

    return np.argmax(gamma_i,axis=1)

def params_from_predicted_cluster(data,cluster):
    means = np.zeros((np.unique(cluster).shape[0],data.shape[1]))
    covariances = np.zeros((np.unique(cluster).shape[0],data.shape[1],data.shape[1]))
    weights = np.zeros((np.unique(cluster).shape[0]))

    for i in range(0,np.unique(cluster).shape[0]):
        # print('cluster: %d has elements %d' % (i,data[np.where(cluster==i)[0]].shape[0]))

        if data[np.where(cluster==i)[0]].shape[0] == 0:
            means[i] = np.zeros((data.shape[1])) + 1e-10
            covariances[i] = np.eye(data.shape[1])
            weights[i] = 1e-10
        elif data[np.where(cluster==i)[0]].shape[0] == 1:
            means[i] = np.mean(data[np.where(cluster==i)[0]],axis=0)
            covariances[i] = np.eye(data.shape[1])
            weights[i] = np.where(cluster==i)[0].shape[0]/data.shape[0]
        else:
            means[i] = np.mean(data[np.where(cluster==i)[0]],axis=0)
            covariances[i] = np.cov(data[np.where(cluster==i)[0]].T)
            weights[i] = np.where(cluster==i)[0].shape[0]/data.shape[0]


    return means,covariances,weights

    


# main
if __name__ == "__main__":
    # load data
    X = np.loadtxt("./data6D.txt")

    params = {'num_components': 5, 'max_iter': 100, 'tolerance': 1e-5, 'init_param': 'random_points'}
    gmm = GaussianMixtureModel(params)

    gmm.fit(X)

    X_2D = PCA(n_components=2).fit_transform(X)

    plt.ion()
    
    for i in range(0,gmm.convrged_at_iter):
        plt.cla()
        plt.clf()

        predicted_cluster = predict_cluster(X,gmm,gmm.means_history[i],gmm.covariances_history[i],gmm.weights_history[i])

        means,covariances,weights = params_from_predicted_cluster(X_2D,predicted_cluster)
        plot_gmm(gmm,X_2D,means,covariances,weights,i)
        plt.pause(0.2)

    plt.ioff()
    plt.show()

