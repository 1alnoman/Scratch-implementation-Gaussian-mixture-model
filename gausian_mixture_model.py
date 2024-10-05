import numpy as np
import scipy.stats as stats


class GaussianMixtureModel:
    def __init__(self, params):
        self.num_components = params.get('num_components', 3)
        self.max_iter = params.get('max_iter', 100)
        self.tolerance = params.get('tolerance', 0)
        self.init_param = params.get('init_param', 'random_points')
        self.means = None
        self.covariances = None
        self.weights = None

        self.means_history = None
        self.covariances_history = None
        self.weights_history = None

        self.convrged_at_iter = self.max_iter

    def initialize_params(self, X, k, init_params="random_points"):
        """
        Initialize the parameters of the GMM.
        
        Args:
            X: N x D numpy array of data points
            k: number of clusters
            init_params: string specifying how to initialize the parameters
            
        Returns:
            mu: k x D numpy array of cluster means
            sigma: k x D x D numpy array of covariance matrices
            pi: k x 1 numpy array of cluster weights
        """
        N, D = X.shape
        mu = np.zeros((k, D))
        sigma = np.zeros((k, D, D))
        pi = np.zeros(k)


        if init_params == "random_points":
            # Initialize mu to random points from X
            # Initialize sigma to identity matrices
            # Initialize pi to uniform
            mu = X[np.random.choice(N, k, replace=False)]
            sigma = np.array([np.cov(X.T) for _ in range(k)]) #np.array([np.eye(D) for i in range(k)])
            pi = np.ones(k) / k
        elif init_params == "random":
            # Initialize mu to random points from a Gaussian
            # Initialize sigma to identity matrices
            # Initialize pi to uniform
            # mu = np.random.randn(k, D)
            mu = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), size=(k, D))
            sigma = np.array([np.cov(X.T) for _ in range(k)]) #np.array([np.eye(D) for i in range(k)])
            pi = np.ones(k) / k


        return mu, sigma, pi

    def gaussian_multivariate(self, X, mu, sigma):
        """
        Compute the probability of a set of points under a multivariate Gaussian distribution.
        
        Args:
            X: N x D numpy array of data points
            mu: D x 1 numpy array of mean
            sigma: D x D numpy array of covariance matrix
            
        Returns:
            p: N x 1 numpy array of probabilities
        """
        return stats.multivariate_normal.pdf(X, mean=mu, cov=sigma,allow_singular=True)


    def expectaion_step(self, X, mu, sigma, pi):
        """
        Perform the expectation step of the EM algorithm.
        
        Args:
            X: N x D numpy array of data points
            mu: k x D numpy array of cluster means
            sigma: k x D x D numpy array of covariance matrices
            pi: k x 1 numpy array of cluster weights
            
        Returns:
            gamma: N x k numpy array of posterior probabilities
        """
        N, D = X.shape
        k, D = mu.shape
        gamma = np.zeros((N, k))

        for j in range(k):
            gamma[:, j] = pi[j] * self.gaussian_multivariate(X, mu[j], sigma[j]).reshape(-1) + np.finfo(float).eps
        gamma /= np.sum(gamma, axis=1).reshape(-1, 1)
        return gamma

    def maximization_step(self ,X, gamma):
        """
        Perform the maximization step of the EM algorithm.
        
        Args:
            X: N x D numpy array of data points
            gamma: N x k numpy array of posterior probabilities
            
        Returns:
            mu: k x D numpy array of cluster means
            sigma: k x D x D numpy array of covariance matrices
            pi: k x 1 numpy array of cluster weights
        """
        N, D = X.shape
        N, k = gamma.shape
        mu = np.zeros((k, D))
        sigma = np.zeros((k, D, D))
        pi = np.zeros(k)

        for j in range(k):
            mu[j] = np.sum(gamma[:, j].reshape(-1, 1) * X, axis=0) / np.sum(gamma[:, j])
            sigma[j] = np.dot((gamma[:, j].reshape(-1, 1) * (X - mu[j])).T, X - mu[j]) / np.sum(gamma[:, j])
            pi[j] = np.sum(gamma[:, j]) / N

        return mu, sigma, pi

    def log_likelihood(self, X, mu, sigma, pi):
        N, D = X.shape
        k, D = mu.shape
        l_in = np.zeros((N, k))
        for j in range(k):
            l_in[:, j] = pi[j] * self.gaussian_multivariate(X, mu[j], sigma[j]).reshape(-1)
        l_i = np.log(np.sum(l_in, axis=1))
        return np.sum(l_i),l_i


    def fit(self, X):
        """
        Fit a GMM to the data.
        
        Args:
            X: N x D numpy array of data points
            
        Returns:
            mu: k x D numpy array of cluster means
            sigma: k x D x D numpy array of covariance matrices
            pi: k x 1 numpy array of cluster weights
        """
        N, D = X.shape
        k=self.num_components
        max_epochs=self.max_iter
        init_params=self.init_param
        tolerance=self.tolerance


        self.means_history = np.zeros((max_epochs, k, D))
        self.covariances_history = np.zeros((max_epochs, k, D, D))
        self.weights_history = np.zeros((max_epochs, k))


        mu, sigma, pi = self.initialize_params(X, k, init_params)
        loss,loss_i = self.log_likelihood(X, mu, sigma, pi)
        
        for i in range(max_epochs):
            gamma = self.expectaion_step(X, mu, sigma, pi)
            mu, sigma, pi = self.maximization_step(X, gamma)
            l,l_i = self.log_likelihood(X, mu, sigma, pi)

            self.means_history[i] = mu
            self.covariances_history[i] = sigma
            self.weights_history[i] = pi

            if np.abs(loss - l) < tolerance:
                self.convrged_at_iter = i+1
                break
            loss,loss_i = l,l_i

            if i % 10 == 0:
                # print('i: %d loss: %f' % (i,log_likelihood(X, mu, sigma, pi)))
                None


        self.means = mu
        self.covariances = sigma
        self.weights = pi

    def score(self, X):
        """
        Compute the log likelihood of the data.
        
        Args:
            X: N x D numpy array of data points
            
        Returns:
            l: scalar log likelihood
        """
        l,l_i = self.log_likelihood(X, self.means, self.covariances, self.weights)
        return l

    def score_samples(self, X):
        """
        Compute the log likelihood of the data.
        
        Args:
            X: N x D numpy array of data points
            
        Returns:
            l: N x 1 numpy array of log likelihoods
        """
        l,l_i = self.log_likelihood(X, self.means, self.covariances, self.weights)
        return l_i



