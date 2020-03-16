import numpy as np
import matplotlib.pyplot as plt
import util


def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elements: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    # TODO: Write your code here
    # Find number of males and number of females
    num_males = np.sum(y == 1)
    num_females = np.sum(y == 2)
    num_people = len(y)
    # Find the indices for males and females
    male_indices = np.argwhere(y == 1).reshape(num_males)
    female_indices = np.argwhere(y == 2).reshape(num_females)
    # Separate male info and female info
    x_male = np.take(x, male_indices, axis=0)
    x_female = np.take(x, female_indices, axis=0)
    # Mu's
    mu_male = (1 / num_males) * x_male.sum(axis=0)
    mu_female = (1 / num_females) * x_female.sum(axis=0)
    # Covariances
    cov_male = np.dot((x_male - mu_male).T, (x_male - mu_male)) / num_males
    cov_female = np.dot((x_female - mu_female).T, (x_female - mu_female)) / num_females

    cov = (num_males * cov_male + num_females * cov_female) / num_people
    
    # LDA plot
    plt.figure(0)

    plt.scatter(x_male.T[0], x_male.T[1], color='b')
    plt.scatter(x_female.T[0], x_female.T[1], color='r')

    x = np.arange(50, 81, 0.5)
    y = np.arange(80, 281)
    X, Y = np.meshgrid(x, y)

    LDA_male = []
    LDA_female = []

    x_val = X[0].reshape(62, 1)
    for i in range(201):
        data = np.concatenate((x_val, Y[i].reshape(62, 1)), 1)
        LDA_male.append(util.density_Gaussian(mu_male, cov, data))
        LDA_female.append(util.density_Gaussian(mu_female, cov, data))

    plt.contour(X, Y, LDA_male)
    plt.contour(X, Y, LDA_female)

    LDA_boundary = np.array(LDA_male) - np.array(LDA_female)
    plt.contour(X, Y, LDA_boundary, [0])
    plt.title('Linear Discriminant Analysis')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.legend(['Male', 'Female'])
    plt.savefig('lda.pdf')

    plt.show()

    # QDA plot
    plt.figure(1)

    plt.scatter(x_male.T[0], x_male.T[1], color='b')
    plt.scatter(x_female.T[0], x_female.T[1], color='r')

    QDA_male = []
    QDA_female = []

    for i in range(201):
        data = np.concatenate((x_val, Y[i].reshape(62, 1)), 1)
        QDA_male.append(util.density_Gaussian(mu_male, cov_male, data))
        QDA_female.append(util.density_Gaussian(mu_female, cov_female, data))

    plt.contour(X, Y, QDA_male)
    plt.contour(X, Y, QDA_female)

    QDA_boundary = np.array(QDA_male) - np.array(QDA_female)
    plt.contour(X, Y, QDA_boundary, [0])
    plt.title('Quadratic Discriminant Analysis')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.legend(['Male', 'Female'])
    plt.savefig('qda.pdf')

    plt.show()

    return mu_male, mu_female, cov, cov_male, cov_female


def misRate(mu_male, mu_female, cov, cov_male, cov_female, x, y):
    """
    Use LDA/QDA on the testing set and compute the mis-classification rate
    
    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    # TODO: Write your code here

    # x here is already x_transpose in equation
    cov_inv = np.linalg.inv(cov)
    cov_male_inv = np.linalg.inv(cov_male)
    cov_female_inv = np.linalg.inv(cov_female)
    mu_male_t = mu_male.reshape(1, 2)
    mu_female_t = mu_female.reshape(1, 2)

    # LDA, constant term does not effect comparison, eliminated
    LDA_male = -0.5 * np.dot(np.dot(mu_male_t, cov_inv), mu_male) + np.dot(np.dot(x, cov_inv), mu_male)
    LDA_female = -0.5 * np.dot(np.dot(mu_female_t, cov_inv), mu_female) + np.dot(np.dot(x, cov_inv), mu_female)

    LDA_y_hat = (LDA_female >= LDA_male) + 1
    miss_LDA = np.sum(y != LDA_y_hat) / len(y)

    # QDA, constant term does not effect comparison, eliminated
    QDA_y_male = -0.5 * np.dot(np.dot(x - mu_male, cov_male_inv), (x - mu_male).T) - np.log(np.sqrt(np.linalg.det(cov_male)))
    QDA_y_female = -0.5 * np.dot(np.dot(x - mu_female, cov_female_inv), (x - mu_female).T) - \
                                                                                   np.log(np.sqrt(np.linalg.det(cov_female)))

    QDA_y_male = np.diagonal(QDA_y_male)
    QDA_y_female = np.diagonal(QDA_y_female)

    QDA_y_hat = (QDA_y_female >= QDA_y_male) + 1
    miss_QDA = np.sum(y != QDA_y_hat) / len(y)

    return miss_LDA, miss_QDA


if __name__ == '__main__':
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')

    # parameter estimation and visualization in LDA/QDA
    mu_male, mu_female, cov, cov_male, cov_female = discrimAnalysis(x_train, y_train)

    # mis-classification rate computation
    miss_LDA, miss_QDA = misRate(mu_male, mu_female, cov, cov_male, cov_female, x_test, y_test)
    print(miss_LDA, miss_QDA)
