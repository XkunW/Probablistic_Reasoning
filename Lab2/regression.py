import numpy as np
import matplotlib.pyplot as plt
import util


def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the prior distribution
    
    Outputs: None
    -----
    """
    mean = np.array([0, 0])
    covariance = np.array([[beta, 0], [0, beta]])

    # prior distribution contour plot
    xx = np.arange(-1, 1, 0.01)
    yy = np.arange(-1, 1, 0.01)
    X, Y = np.meshgrid(xx, yy)
    plot = []
    for i in range(200):
        x_set = np.concatenate((X[0].reshape(200, 1), Y[i].reshape(200, 1)), 1)
        plot.append(util.density_Gaussian(mean, covariance, x_set))
    plt.title('Prior Distribution')
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.contour(X, Y, plot)
    plt.plot([-0.1], [-0.5], marker='*', markersize=10)
    plt.savefig('./prior.pdf')
    # plt.show()
    plt.close()


def posteriorDistribution(x, z, beta, sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from training set
    beta: hyperparameter in the prior distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    n = len(x)
    inv_covariance_a = np.array([[1 / beta, 0], [0, 1 / beta]])
    A = np.append(np.ones(shape=(n, 1)), x, axis=1)
    Cov = np.linalg.inv(inv_covariance_a + np.matmul(A.T, A) / sigma2)
    mu = np.matmul(Cov, np.matmul(A.T, z) / sigma2).reshape(1, 2).squeeze()
    # posterior distribution contour plot
    xx = np.arange(-1, 1, 0.01)
    yy = np.arange(-1, 1, 0.01)
    X, Y = np.meshgrid(xx, yy)
    plot = []
    for i in range(200):
        data = np.concatenate((X[0].reshape(200, 1), Y[i].reshape(200, 1)), 1)
        plot.append(util.density_Gaussian(mu, Cov, data))
    plt.title('Posterior Distribution: n = {}'.format(n))
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.contour(X, Y, plot)
    plt.plot([-0.1], [-0.5], marker='*', markersize=10)
    plt.savefig('./posterior{}.pdf'.format(n))
    # plt.show()
    plt.close()

    return mu, Cov


def predictionDistribution(x, beta, sigma2, mu, Cov, x_train, z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    n = len(x_train)
    A = np.append(np.ones([len(x), 1]), np.expand_dims(x, 1), axis=1)

    mu_z = np.matmul(A, mu)

    cov_z = np.matmul(np.matmul(A, Cov), A.T) + sigma2
    std_z = np.sqrt(np.diag(cov_z))

    plt.title('Prediction Distribution: n = {}'.format(n))
    plt.xlabel('input: x')
    plt.ylabel('target: z')
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.scatter(x_train, z_train, marker='*', s=40, label='Training Samples')
    plt.errorbar(x, mu_z, yerr=std_z, fmt='rx', label='Testing Samples')
    plt.legend()
    plt.savefig('./predict{}.pdf'.format(n))
    # plt.show()
    plt.close()


if __name__ == '__main__':
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4, 4.01, 0.2)]

    # known parameters 
    sigma2 = 0.1
    beta = 1

    # prior distribution p(a)
    priorDistribution(beta)

    # number of training samples used to compute posterior
    for ns in [1, 5, 100]:
        # used samples
        x = x_train[0:ns]
        z = z_train[0:ns]

        # posterior distribution p(a|x,z)
        mu, Cov = posteriorDistribution(x, z, beta, sigma2)

        # distribution of the prediction
        predictionDistribution(x_test, beta, sigma2, mu, Cov, x, z)
