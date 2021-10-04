import csv
import numpy as np
from scipy.spatial.distance import cdist
from libsvm.svmutil import *



def load_data():
    x_test=[]
    with open('./data/X_test.csv') as cvsfile:
        rows=csv.reader(cvsfile)
        for row in rows:
            x_test.append([float(v) for v in row])
        x_test=np.asarray(x_test,dtype='float')

    x_train=[]
    with open('./data/X_train.csv') as cvsfile:
        rows=csv.reader(cvsfile)
        for row in rows:
            x_train.append([float(v) for v in row])
        #print(x_train)
        x_train=np.asarray(x_train,dtype='float')
        #print(x_train)

    
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

def new_kernel(x,y):
    gamma=1

    linear_k=x @ y.T
    RBF_k=np.exp(-gamma*cdist(x,y, 'sqeuclidean'))

    k=linear_k+RBF_k
    k=np.hstack((np.arange(1,len(x)+1).reshape(-1,1),k))

    return k





if __name__ == '__main__':
    x_train , y_train , x_test , y_test = load_data()

    

    kernel=new_kernel(x_train,x_train)

    prob=svm_problem(y_train,kernel,isKernel=True)


    model=svm_train(prob,'-t 4 -q')

    kernel_test=new_kernel(x_test, x_train)
    p_label,p_acc,p_vals=svm_predict(y_test,kernel_test,model,'-q')
    print('linear kernel + RBF kernel: {:.2f}%'.format(p_acc[0]))