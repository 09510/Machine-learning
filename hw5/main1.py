

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import cdist



def Load():
    x = []
    y = []
    with open('./data/input.data', 'r', encoding='utf-8') as file:
        for line in file:
            x_i, y_i = line.split(' ')
            x.append(float(x_i))
            y.append(float(y_i))
    x = np.array(x, dtype=np.float64).reshape(-1, 1)
    y = np.array(y, dtype=np.float64).reshape(-1, 1)
    return x, y

def Objective(theta, X, Y, beta):
    theta = theta.ravel()
    kernel = RQK(X, X,theta[0],theta[1],theta[2]) + np.identity(len(X), dtype=np.float64) * (1 / beta)
    result = np.sum(np.log(np.diagonal(np.linalg.cholesky(kernel)))) + 0.5 * Y.T @ np.linalg.inv(kernel) @ Y + 0.5 * len(X) * np.log(2 * np.pi)
    return result



def RQK(X1, X2,alpha,sigma,lengh):
    kernel = (sigma ** 2) * ((cdist(X1, X2, 'sqeuclidean') / 2 * alpha * (lengh ** 2)) + 1) ** (-alpha)
    return kernel

def GP(x,y,x_test):
    
    
    kernel = RQK(x, x,alpha,sigma,lengh) 
    kernel_star = RQK(x, x_test,alpha,sigma,lengh)
    kernel_star_star = RQK(x_test, x_test,alpha,sigma,lengh) 

    
    C = kernel + np.identity(len(x), dtype=np.float64) * (1 / beta)
    C_inv = np.linalg.inv(C)

    mean = kernel_star.T @ C_inv @ y
    var = kernel_star_star + np.identity(len(x_test), dtype=np.float64) * (1 / beta) - kernel_star.T @ C_inv @ kernel_star
    var = 1.96*np.sqrt(np.diag(var))#95%信賴區間

    x_test = x_test.ravel()
    mean = mean.ravel()
    
    plt.plot(x_test, mean, color='b')
    plt.scatter(x, y, color='k')
    

    plt.plot(x_test, mean + var, color='g')
    plt.plot(x_test, mean - var, color='g')
    plt.fill_between(x_test, mean + var, mean - var, color='g', alpha=0.3)

    plt.xlim(-60, 60)
    plt.show()

alpha=1
beta=5
sigma=1
lengh=1


if __name__ == '__main__':

    x,y=Load()
    

    #part1
    x_test = np.linspace(-60.0, 60.0, 1000).reshape(-1, 1)
    GP(x,y,x_test)





    #part2
    optimal=100000
    opt_ini=[]
    test_set=[30,20,10,1,0.1,0.01,0.001,0.0001]

    for sigma_test in test_set:
        for alpha_test in test_set:
            for lengh_test in test_set:
                opt = minimize(Objective, [alpha_test,sigma_test,lengh_test], bounds=((1e-8, 1e6), (1e-8, 1e6), (1e-8, 1e6)), args=(x, y, beta))
                if opt.fun < optimal:
                    print('-')
                    opt_ini=[sigma_test,alpha_test,lengh_test]
                    alpha=opt.x[0]
                    sigma=opt.x[1]
                    lengh=opt.x[2]
                    optimal=opt.fun

    print(opt_ini)
    print(alpha,sigma,lengh)

    GP(x,y,x_test)

