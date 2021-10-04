import csv
import sys
import time
import numpy as np
from libsvm.svmutil import *



def load_data():
    x_test=[]
    with open('./data/X_test.csv') as cvsfile:
        rows=csv.reader(cvsfile)
        for row in rows:
            x_test.append([float(v) for v in row])

    x_train=[]
    with open('./data/X_train.csv') as cvsfile:
        rows=csv.reader(cvsfile)
        for row in rows:
            x_train.append([float(v) for v in row])

    y_test=[]
    with open('./data/Y_test.csv') as csvfile:
        rows=csv.reader(csvfile)
        for row in rows:
            y_test.append(int(row[0]))
    

    y_train=[]
    with open('./data/Y_train.csv') as csvfile:
        rows=csv.reader(csvfile)
        for row in rows:
            y_train.append(int(row[0]))


    return x_train , y_train , x_test , y_test



def grid(x_train , y_train , x_test , y_test):
    gamma=[0.001,0.01, 0.5, 0.1, 1,5,10,15]
    cost=[ 0.001,0.01,0.1, 1,5,10,15]

    result=np.zeros((3,len(gamma),len(cost)))

    for i_gamma in range(len(gamma)): 
        for i_cost in range(len(cost)):
            for kernel_t in range(3):
                
                para='-q -v 3 -t '
                para+=str(kernel_t)
                para+=' -g '
                para+=str(gamma[i_gamma])
                para+=' -c ' 
                para+=str(cost[i_cost])
                print(para)
                acc=svm_train(y_train,x_train,para)
                result[kernel_t][i_gamma][i_cost]=acc
                print(acc)

    return result



if __name__ == '__main__':
    x_train , y_train , x_test , y_test = load_data()
    result = grid(x_train , y_train , x_test , y_test)
    print(result)
    ind = np.unravel_index(np.argmax(result, axis=None), result.shape)
    
    print(result[ind[0]][ind[1]][ind[2]])