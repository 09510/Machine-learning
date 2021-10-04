import numpy as np
import os
import matplotlib.pyplot as plt
import time



def read_train():
    mnist_dir_image="data/train-images.idx3-ubyte"
    mnist_dir_label="data/train-labels.idx1-ubyte"
   

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

    return num, images, labels, pixels



def read_test():
    mnist_dir_image="./data/t10k-images.idx3-ubyte"
    mnist_dir_label="./data/t10k-labels.idx1-ubyte"

    with open(mnist_dir_image, 'rb') as image:
        #讀int,4個byte,順序排列,四次
        magic, num, row, col = np.fromfile(image, dtype=np.dtype('>i'), count=4)
        #讀全部的 unsign byte,順序排列
        images = np.fromfile(image, dtype=np.dtype('>B'), count=-1)

    with open(mnist_dir_label, 'rb') as label:
        #讀int,4個byte,順序排列,四次
        magic, num = np.fromfile(label, dtype=np.dtype('>i'), count=2)
        #讀全部的 unsign byte,順序排列 
        labels = np.fromfile(label, dtype=np.dtype('>B'), count=-1)

    pixels = row*col
    images = images.reshape(num, pixels)

    return num, images, labels, pixels


def calculate_prior_discrete(num, image, label,pixel):

    prior = np.zeros((10), dtype=np.int32)
    image_all_bin = np.zeros((10, pixel, 32), dtype=np.int32)

    try:
        print("load prior")
        prior = np.load("prior.npy")
    except:
        print("train prior")
        for i in range(num):
            this_label=label[i]
            prior[this_label]=prior[this_label]+1
        np.save("prior.npy", prior)

    try:
        print("load image_all_bin")
        image_all_bin = np.load("image_all_bin.npy")
    except:
        print("train image_all_bin")
        for i in range(num):
            this_label=label[i]
            for j in range(pixel):
                #print("i: ",i,"  j: ",j)
                color=image[i][j]
                color=color // 8
                image_all_bin[this_label][j][color] += 1
            np.save("image_all_bin.npy", image_all_bin)


    
    

    return prior, image_all_bin 


def calculate_prior_continuous(num, image, label,pixel):
    prior = np.zeros((10), dtype=np.int32)
    var = np.zeros((10, pixel), dtype=np.float)
    mean = np.zeros((10, pixel), dtype=np.float)
    mean_square = np.zeros((10, pixel), dtype=np.float)

    try:
        print("load prior")
        prior = np.load("prior.npy")
    except:
        print("train prior")
        for i in range(num):
            this_label=label[i]
            prior[this_label]=prior[this_label]+1
        np.save("prior.npy", prior)
    
    

    try:
        print("load mean")
        mean=np.load("mean.npy")
    except:
        for i in range(num):
            this_label=label[i]
            for j in range(pixel):
                #print(image[i][j])
                color=image[i][j]
                color=color
                mean[this_label][j]+=color
                #print(mean[1][1])
        
        for i in range(10):
            for j in range(pixel):
                mean[i][j]=mean[i][j]/prior[i]
                print(mean[i][j])
        np.save("mean.npy",mean)
        '''
        for i in range(10):
            for j in range(pixel):
                print(mean[i][j])
        '''




    try:
        print("load mean_square")
        mean_square=np.load("mean_square.npy")
        
    except:
        for i in range(num):
            this_label=label[i]
            for j in range(pixel):
                color=image[i][j]
                mean_square[this_label][j]+=color ** 2
        
        for i in range(10):
            for j in range(pixel):
                mean_square[i][j]=mean_square[i][j]/prior[i]
        
        np.save("mean_square.npy",mean_square)
        
        for i in range(10):
            for j in range(pixel):
                print(mean_square[i][j])






    try:
        print("load varient")
        var=np.load("var.npy")
    except:
        for i in range(10):
            for j in range(pixel):
                #E(V)=E(mean_square)-E(u)^2
                var[i][j]=mean_square[i][j]-pow(mean[i][j],2)
                if var[i][j]<=100:
                    var[i][j]=100

        np.save("var.npy",var)
        
        for i in range(10):
            for j in range(pixel):
                print(var[i][j])
        

    return prior, mean , mean_square ,var

        
def imagination_digit_discrete(image_all_bin):
    '''
    for i in range(10):
        for j in range(28*28):
            for k in range(32):
                print(image_all_bin[i][j][k])

    '''
    for i in range(10):
        
        print("\n\n\n")
        print("數字 :",i)
        for j in range(28*28):
            
            white=0
            black=0
            for t in range(16):
                white += image_all_bin[i][j][t]
            for t in range(16, 32):
                black += image_all_bin[i][j][t]


            total=black-white
            if j%28==0:
                print("") 
            #print(white,black,total)
            if total > 0:
                print("1",end=" ")
            else:
                print("0",end=" ")

        

               

def imagination_digit_continus(mean):
    '''
    for i in range(10):
        for j in range(28*28):
            for k in range(32):
                print(image_all_bin[i][j][k])

    '''
    for i in range(10):
        print("\n\n\n")
        print("數字 :",i)
        for j in range(28*28):

            if j%28==0:
                print("") 
            if mean[i][j]>128:
                print("1",end=" ")
            else:
                print("0",end=" ")

        






def Normalize(predict):
    total=np.sum(predict)
    predict=predict/total
    return predict

def CheckAns(predict, ans):
    print("")
    print("Postirior (in log scale):")

    for i in range(10):
        print(i,":",predict[i])

    predict_ans=np.argmin(predict)
    print("prediction: ",predict_ans,"  ,Ans: ",ans)
    if predict_ans==ans:
        return 0
    else:
        return 1
    

def Test_Discrete(num,image,label,pixel,t_num,t_image,t_label,t_pixel,image_all_bin,prior):
    print("start test")
    error=0

    for i in range(t_num):
        
        predict = np.zeros((10), dtype=np.float)
        this_image_pixel = np.zeros((t_pixel), dtype=np.int32)
        this_image_pixel = t_image[i]
        predict=np.log(prior/num)
        #print(predict)
        #print(this_image_pixel)
        for j in range(10):

            for k in range(pixel):
                # consider likelihood
                this_image_pixel_color=this_image_pixel[k]//8
                #print(this_image_pixel_color)
                likelihood = image_all_bin[j][k][this_image_pixel_color]
                if likelihood == 0:
                    likelihood = 0.000001
                predict[j] += np.log(likelihood / prior[j])

        predict=Normalize(predict=predict)
        print(error,"/",i)
        error=error+CheckAns(predict=predict,ans=t_label[i])
    

    return error/t_num

    
def Test_Continuous(num,image,label,pixel,t_num,t_image,t_label,t_pixel,var,mean,mean_square,prior):
    print("start test")
    error=0

    for i in range(t_num):
        predict = np.zeros((10), dtype=np.float)
        this_image_pixel = np.zeros((t_pixel), dtype=np.int32)
        this_image_pixel = t_image[i]
        predict=np.log(prior/num)
        #print(predict)
        #print(this_image_pixel)

        for j in range(10):
            for k in range(pixel):
                # consider likelihood
                this_image_pixel_color=this_image_pixel[k]
                #print(this_image_pixel_color)
                part_i = np.log(2*np.pi*var[j][k])
                part_ii=pow( (this_image_pixel_color-mean[j][k]),2)/var[j][k]
                likelihood=-0.5*(part_i+part_ii)  
                
                predict[j] += likelihood

        predict=Normalize(predict=predict)
        print(error,"/",i)
        error=error+CheckAns(predict=predict,ans=t_label[i])
    return error/t_num

    




def Train_Discrete(num,image,label,pixel,t_num,t_image,t_label,t_pixel):
    print("use discrete")

    prior = np.zeros((10), dtype=np.int32)
    image_all_bin = np.zeros((10, pixel, 32), dtype=np.int32)

    prior,image_all_bin=calculate_prior_discrete(num,image,label,pixel)
    error_rate=Test_Discrete(num=num,image=image,label=label,pixel=pixel,t_num=t_num,t_image=t_image,t_label=t_label,t_pixel=t_pixel,image_all_bin=image_all_bin,prior=prior)
    print("")
    print("")
    print("Imagination of numbers in Bayesian classifier:")
    imagination_digit_discrete(image_all_bin=image_all_bin)
    print("")
    print("")
    print("Error Rate:",error_rate)
    
def Train_Continuous(num,image,label,pixel,t_num,t_image,t_label,t_pixel):
    print("use continuous")
    prior = np.zeros((10), dtype=np.int32)
    var = np.zeros((10, pixel), dtype=np.float)
    mean = np.zeros((10, pixel), dtype=np.float)
    mean_square = np.zeros((10, pixel), dtype=np.float)

    prior,mean,mean_square,var=calculate_prior_continuous(num,image,label,pixel)
    erro_rate=Test_Continuous(num=num,image=image,label=label,pixel=pixel,t_num=t_num,t_image=t_image,t_label=t_label,t_pixel=t_pixel,var=var,mean=mean,mean_square=mean_square,prior=prior)
    print("")
    print("")
    print("Imagination of numbers in Bayesian classifier:")
    imagination_digit_continus(mean=mean)

    print("")
    print("")
    print("Error Rate:",erro_rate)




    









if __name__=='__main__':

    train_num, train_images, train_labels, pixels = read_train()
    test_num, test_images, test_labels, text_pixels = read_test()

    train_type=input("請輸入 mode  0:discrete  1:continuous :")

    t1=time.time()
    if train_type == "0":
        Train_Discrete(num=train_num,image=train_images,label=train_labels,pixel=pixels,t_num=test_num,t_image=test_images,t_label=test_labels,t_pixel=text_pixels)
    elif train_type == "1":
        Train_Continuous(num=train_num,image=train_images,label=train_labels,pixel=pixels,t_num=test_num,t_image=test_images,t_label=test_labels,t_pixel=text_pixels)
    else:
        print("wrong")
    
    t2=time.time()
    print(t2-t1)






    



    
