import numpy as np
import matplotlib.pyplot as plt
from gausian_mixture_model import GaussianMixtureModel

# main
if __name__ == "__main__":
    # load data
    X = np.loadtxt("./data3D.txt")

    max_clusters = 10
    losses = np.zeros(max_clusters)

    for i in range(0,max_clusters):
        params = {'num_components': i+1, 'max_iter': 300, 'tolerance': 1e-8, 'init_param': 'random_points'}

        gmm = GaussianMixtureModel(params)
        gmm.fit(X)
        losses[i] = gmm.score(X)
        print('k: %d loss: %f' % (i+1,losses[i]))

    plt.plot(range(1,max_clusters+1),losses,marker='x')
    plt.xlabel('Number of clusters')
    plt.ylabel('Loss')
    plt.title('Loss vs Number of clusters')
    plt.savefig('k_vs_loss.png')