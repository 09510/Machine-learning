import numpy as np
from numba import jit
import os
np.set_printoptions(threshold=np.inf)

def read_train():
    mnist_dir_image="./train-images.idx3-ubyte"
    mnist_dir_label="./train-labels.idx1-ubyte"
   

    with open(mnist_dir_image, 'rb') as image:
        #讀int,4個byte,順序排列,四次
        magic, num, row, col = np.fromfile(image, dtype=np.dtype('>i4'), count=4)

        #讀全部的 unsign byte,順序排列
        images = np.fromfile(image, dtype=np.dtype('>B'), count=-1)

    

    with open(mnist_dir_label, 'r') as label: 
        #讀int,4個byte,順序排列,四次
        magic, num = np.fromfile(label, dtype=np.dtype('>i4'), count=2) 
        
        #讀全部的 unsign byte,順序排列
        labels = np.fromfile(label, dtype=np.dtype('>B'), count=-1) 
    

    pixels = row*col
    images = images.reshape(num, pixels)
    images=images//128

    return num, images, labels, pixels


@jit
def print_confusion_matrix(train_data, train_label, Pi, lamda, relation):
    error = data_num
    confusion_matrix = np.zeros(shape=(10, 4), dtype=np.int)       # TP FP TN FN
    for this_data in range(data_num):
        #print(this_data)
        binomial_coun = np.full((10), 1, dtype=np.float64)
        for class_num in range(10):
            for this_pixel in range(28*28):
                if train_data[this_data][this_pixel] == 1:
                    binomial_coun[class_num] *= Pi[class_num][this_pixel]
                else:
                    binomial_coun[class_num] *= (1 - Pi[class_num][this_pixel])
            binomial_coun[class_num] *= lamda[class_num][0]

        predict_cluster = np.argmax(binomial_coun)
        predict_label = np.where(relation==predict_cluster)
        for num_idx in range(10):
            if num_idx == train_label[this_data]:
                if num_idx == predict_label[0]:
                    error -= 1
                    confusion_matrix[num_idx][0] += 1
                else:
                    confusion_matrix[num_idx][3] += 1
            else:
                if num_idx == predict_label[0]:
                    confusion_matrix[num_idx][1] += 1
                else:
                    confusion_matrix[num_idx][2] += 1

    for num_idx in range(10):
        print("Confusion matrix {}:".format(num_idx))
        print("\t\tPredict number {}\tPredict not number {}".format(num_idx, num_idx))
        print("Is number {}\t\t{}\t\t\t{}".format(num_idx, confusion_matrix[num_idx][0], confusion_matrix[num_idx][3]))
        print("Isn't number {}\t\t{}\t\t\t{}".format(num_idx, confusion_matrix[num_idx][1], confusion_matrix[num_idx][2]))
        print("Sensitivity (Successfully predict number {}): {}".format(num_idx, confusion_matrix[num_idx][0] / (confusion_matrix[num_idx][0] + confusion_matrix[num_idx][3])))
        print("Specificity (Successfully predict not number {}): {}".format(num_idx, confusion_matrix[num_idx][2] / (confusion_matrix[num_idx][2] + confusion_matrix[num_idx][1])))
        print("---------------------------------------------------------------\n")

    return error

def print_Pi_digit(Pi):
    Pi_new = Pi.copy()
    for num_idx in range(class_num):
        print("\nclass: ", num_idx)
        for pixel_idx in range(28*28):
            if pixel_idx % 28 == 0 and pixel_idx != 0:
                print("")
            if Pi_new[num_idx][pixel_idx] >= 0.5:
                print("1", end=" ")
            else:
                print("0", end=" ")
        print("")

def cal_diff(Pi,last_Pi):
    diff=np.sum(abs(Pi-last_Pi))
    return diff


def imagination_digit(class_label_decide, Pi):

    for this_class in range(10):
        print("\n\n\n")
        print("Labeled class : ",this_class)
        real_class=class_label_decide[this_class]

        for j in range(28*28):
            if j % 28 ==0:
                print("")
            if Pi[real_class][j] >= 0.5:
                print("1",end=" ")
            else:
                print("0",end=" ")



#判斷每個class代表哪個數字



@jit
def label_decide(train_data,label,lamda,Pi):

    table = np.zeros(shape=(10, 10), dtype=np.int)
    relation = np.full((10), -1, dtype=np.int)
    for n in range(60000):
        #print(n)
        temp = np.zeros(shape=10, dtype=np.float64)
        for k in range(10):
            temp1 = np.float64(1.0)
            for i in range( 28 * 28):
                if train_data[n][i] == 1:
                    temp1 *= Pi[k][i]
                else:
                    temp1 *= (1 - Pi[k][i])
            temp[k] = lamda[k][0] * temp1
        table[label[n]][np.argmax(temp)] += 1

    #print(table)
    for i in range(10):
        ind = np.unravel_index(np.argmax(table, axis=None), table.shape)
        relation[ind[0]] = ind[1]
        for j in range(10):
            table[ind[0]][j] = -1 
            table[j][ind[1]] = -1 
        #print(ind[0],ind[1],table)
        #print(relation)
    return relation


def Check(LAMBDA, condition):
    if 0 in LAMBDA:
        return 0
    else:
        return condition + 1


@jit
def E_step(train_data,lamda,Pi,Wi):
    print("do E_step")
    for this_data in range(data_num):
        binomial_count=np.full((class_num,1),1,dtype=np.float64)
        #wi= sigma( lamda_i * pi^xi * (1-pi)^(1-xi))/marginal
        for this_class in range(class_num):
            for this_pixel in range(pixel_num):
                if train_data[this_data][this_pixel]==1:
                    binomial_count[this_class][0] *= Pi[this_class][this_pixel]
                else:
                    binomial_count[this_class][0] *= (1-Pi[this_class][this_pixel])
            binomial_count[this_class][0]*=lamda[this_class][0]
        marginal=np.sum(binomial_count)
        if marginal==0:
            marginal=1
        for this_class in range(class_num):
            Wi[this_data][this_class]=binomial_count[this_class][0]/marginal

    

    return Wi

def M_step(train_data,Wi):
    print("do M_step")

    w_sum=np.sum(Wi,axis=0)
    Pi_update=np.random.random_sample((class_num, pixel_num))
    lamda_update = np.random.random_sample((class_num, 1))

    for this_class in range(class_num):
        lamda_update[this_class][0]=w_sum[this_class]/data_num

    for this_class in range(class_num):
        for this_pixel in range(pixel_num):
            sigma_wx= np.dot(train_data[:,this_pixel],Wi[:,this_class])
            sigma_w=w_sum[this_class]
            if sigma_w==0:
                sigma_w=1
            Pi_update[this_class][this_pixel] = sigma_wx / sigma_w




    return Pi_update,lamda_update



class_num = 10
pixel_num = 28 * 28
data_num = 60000
Epoch=1000



if __name__ == "__main__":
    num, images, labels, pixels = read_train()
    
    condition=0
    

    


    lamda = np.random.random_sample((class_num, 1))
    Pi = np.random.random_sample((class_num, pixel_num))
    Pi_last = np.zeros((class_num, pixel_num), dtype=np.float64)
    Wi  = np.random.random_sample((data_num, class_num))
    '''
    try:
        Wi = np.load("Wi.npy")
    except:
        print(" ")
    try:
        Pi = np.load("Pi.npy")
    except:
        print(" ")
    try:
        lamda = np.load("lamda.npy")
    except:
        print(" ")
    '''
    #print_Pi_digit(Pi)

    try:
        Wi = np.load("Wi.npy")
        Pi = np.load("Pi.npy")
        lamda = np.load("lamda.npy")
    except:

        for i in range(Epoch):
            Wi = E_step(train_data=images,lamda=lamda,Pi=Pi,Wi=Wi)
            Pi,lamda = M_step(train_data=images,Wi=Wi)
            condition=Check(lamda,condition)
            if condition == 0:
                lamda = np.random.random_sample((class_num, 1))
                Pi = np.random.random_sample((class_num, pixel_num))

            
            np.save("Wi.npy", Wi)
            np.save("Pi.npy", Pi)
            np.save("lamda.npy", lamda)
            
            print_Pi_digit(Pi)
            diff=cal_diff(Pi=Pi,last_Pi=Pi_last)
            #print(lamda)
            #print(condition)
            print("No. of Iteration: {",condition,"}, Difference: {",diff,"}")
            print("--------------------------------------------------------------")
            if diff<10 and condition>=10:
                break
            
            Pi_last=Pi
    
    
    #class_label_decide = label_decide(train_data=images, label=labels, lamda=lamda, Pi=Pi)
    
    
    try:
        print("load class_label_decide")
        class_label_decide = np.load("class_label_decide.npy")
    except:
        print("train class_label_decide")
        class_label_decide = label_decide(train_data=images, label=labels, lamda=lamda, Pi=Pi)
        np.save("class_label_decide.npy", class_label_decide)
    
    #print(class_label_decide)
    #print_Pi_digit(Pi)
    imagination_digit(class_label_decide=class_label_decide,Pi=Pi)
    
    print("---------------------------------------------------------------\n")
    
    
    
    error = print_confusion_matrix(train_data=images, train_label=labels, Pi=Pi, lamda=lamda, relation=class_label_decide)
    
    
    #print_Pi_digit(Pi)
    #imagination_digit(class_label_decide=class_label_decide,Pi=Pi)
    print("Total iteration to converge: {}".format(condition))
    print("Total error rate: {}".format(error / data_num))
    


