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


if __name__ == '__main__':
    x_train , y_train , x_test , y_test = load_data()
    
    model=svm_train(y_train,x_train,'-q -t 0')
    p_label,p_acc,p_vals=svm_predict(y_test,x_test,model,'-q')
    linear_accuracy=p_acc[0]
    print("linear kernel: " + str(linear_accuracy)+"%")

    model=svm_train(y_train,x_train,'-q -t 1')
    p_label,p_acc,p_vals=svm_predict(y_test,x_test,model,'-q')
    poly_accuracy=p_acc[0]
    print("polynomial kernel: " + str(poly_accuracy)+"%")

    model=svm_train(y_train,x_train,'-q -t 2')
    p_label,p_acc,p_vals=svm_predict(y_test,x_test,model,'-q')
    rbf_accuracy=p_acc[0]
    print("RBF : " + str(rbf_accuracy)+"%")



    



