import numpy as np
import math
import matplotlib.pyplot as plt

def Univariate_Gaussian_Generator(mean,s):
    z=np.random.uniform(0,1,12)
    z=np.sum(z)
    z=z-6
    z=np.float(z)
    star=s**0.5


    return mean + star * z

def sigmoid(input_X,input_W):
    #print("X",input_X,"\nW",input_W)
    wx=input_X @ input_W
    #print("\n wx",wx)
    s_output=[]
    for i in range(len(wx)):
        temp=[]
        temp.append(1.0 / (1.0 + np.exp(-1.0 * wx[i])))
        s_output.append(temp)
    s_output=np.array(s_output)
    s_output=s_output.reshape(-1,1)
    #print(s_output)
    return s_output

def draw(D1x,D1y,D2x,D2y):
    plt.subplot(131)
    plt.title("Ground Truth")
    plt.scatter(D1x, D1y, c='r')
    plt.scatter(D2x,D2y,c='b')  


    plt.subplot(132)
    plt.title("Gradient descent")
    predict=sigmoid(input_W=gradient_result,input_X=X)
    for i in range(len(X)):
        if predict[i]>=0.5:
            plt.scatter(X[i][0],X[i][1],c='b')
        else:
            plt.scatter(X[i][0],X[i][1],c='r')

    

    
    plt.subplot(133)
    plt.title("Newton's Method")
    predict=sigmoid(input_W=newton_result,input_X=X)
    for i in range(len(X)):
        if predict[i]>=0.5:
            plt.scatter(X[i][0],X[i][1],c='b')
        else:
            plt.scatter(X[i][0],X[i][1],c='r')

    plt.show()

def generate_D1_D2():
    D1x = []
    D1y = []
    D2x = []
    D2y = []
    X = []
    y =[]
    for i in range(0, n):
        record = []
        x1 = Univariate_Gaussian_Generator(mx1, vx1)
        y1 = Univariate_Gaussian_Generator(my1, vy1)
        D1x.append(x1)
        D1y.append(y1)
        record.append(x1)
        record.append(y1)
        record.append(1.0)
        X.append(record)
        y.append([0.0])

        record = []
        x2 = Univariate_Gaussian_Generator(mx2, vx2)
        y2 = Univariate_Gaussian_Generator(my2, vy2)
        D2x.append(x2)
        D2y.append(y2)
        record.append(x2)
        record.append(y2)
        record.append(1.0)
        X.append(record)
        y.append([1.0])
    return X, y, D1x, D1y, D2x, D2y


def confusion_matrix_gradient():

    one_p_one=0
    one_p_two=0
    two_p_one=0
    two_p_two=0


    sensitive=0
    specificity=0

    predict=sigmoid(input_W=gradient_result,input_X=X)

    for i in range(len(y)):
        if predict[i]>=0.5:
            if y[i]==1:
                #print("right")
                two_p_two+=1
            else:
                #print("wrong")
                one_p_two+=1
        else:
            if y[i]==0:
                #print("wrong")
                one_p_one+=1
            else:
                #print("right")
                two_p_one+=1


    sensitive=one_p_one/(one_p_one + one_p_two)
    specificity=two_p_two/(two_p_one + two_p_two)

    print("\nConfusion matrix :")
    print("\t\tPredict cluster 1 \tPredict cluster 2 ")
    print("Is cluster 1 \t\t{}\t\t\t{}".format(one_p_one, one_p_two))
    print("Is cluster 2\t\t{}\t\t\t{}" .format(two_p_one, two_p_two))
    print("Sensitivity (Successfully predict cluster 1): {}".format(sensitive))
    print("Specificity (Successfully predict cluster 2): {}".format(specificity ))
    print("---------------------------------------------------------------\n")

def confusion_matrix_newton():

    one_p_one=0
    one_p_two=0
    two_p_one=0
    two_p_two=0


    sensitive=0
    specificity=0

    predict=sigmoid(input_W=newton_result,input_X=X)

    for i in range(len(y)):
        if predict[i]>=0.5:
            if y[i]==1:
                #print("right")
                two_p_two+=1
            else:
                #print("wrong")
                one_p_two+=1
        else:
            if y[i]==0:
                #print("wrong")
                one_p_one+=1
            else:
                #print("right")
                two_p_one+=1


    sensitive=one_p_one/(one_p_one + one_p_two)
    specificity=two_p_two/(two_p_one + two_p_two)

    print("\nConfusion matrix :")
    print("\t\tPredict cluster 1 \tPredict cluster 2 ")
    print("Is cluster 1 \t\t{}\t\t\t{}".format(one_p_one, one_p_two))
    print("Is cluster 2\t\t{}\t\t\t{}" .format(two_p_one, two_p_two))
    print("Sensitivity (Successfully predict cluster 1): {}".format(sensitive))
    print("Specificity (Successfully predict cluster 2): {}".format(specificity ))
    print("---------------------------------------------------------------\n")
n=50
mx1=1
my1=1
mx2=3
my2=3
vx1=2
vy1=2
vy2=4
vx2=4

w =[[0.0], [0.0], [0.0]]
new_w = [[0.0], [0.0], [0.0]]

gradient_result=[[0.0], [0.0], [0.0]]
newton_result=[[0.0], [0.0], [0.0]]


if __name__== "__main__" :
    
    n = int(input("Number of data points: "))
    mx1=float(input("mx1 : "))
    my1=float(input("my1 : "))
    mx2=float(input("mx2 : "))
    my2=float(input("my2 : "))
    vx1=float(input("vx1 : "))
    vy1=float(input("vy1 : "))
    vx2=float(input("vx2 : "))
    vy2=float(input("vy2 : "))
    
    



    learning_rate=0.01

    X, y, D1x, D1y, D2x, D2y=generate_D1_D2()
    X=np.array(X)
    y=np.array(y)
    w=np.array(w)
    new_w=np.array(new_w)

    
    

    #grandent decent

    print("Gradent descent:\n\nw:\n")
    while(True):
        gradent=X.T @ (y-sigmoid(input_W=w , input_X=X))
        new_w=w + gradent*learning_rate

        #print(np.sum(abs(new_w-w)))
        if np.sum(abs(new_w-w))<=1e-1:
            break
        w=new_w
    gradient_result=w

    print(gradient_result)
    
    confusion_matrix_gradient()


    #newton

    print("Newton's method:\n\nw:\n")
    w =[[0.0], [0.0], [0.0]]
    new_w = [[0.0], [0.0], [0.0]]
    
    w=np.array(w)
    new_w=np.array(new_w)

    
    XT = X.T
    
    iter=0
    #print(np.shape(X))
    while True:
        iter+=1
        D = np.zeros((np.size(X,0),np.size(X,0)))
        for x in range(np.size(X,0)):
            e=np.exp(-X[x] @ w)
            if math.isinf(e):
                D[x,x]=0
            else:
                D[x,x]=e/(1+e)**2

        #print(D)
        Hessian = X.T @ (D @ X)
        gradent=X.T @ (y-sigmoid(input_W=w , input_X=X))
        #print(Hessian)
        if np.linalg.det(Hessian)==0:
            new_w = w + gradent*learning_rate
            print("==============================================")
        else:
            new_w = w + learning_rate*(np.linalg.inv(Hessian) @ gradent)/len(X)
            
            #print(new_w)
        #print(np.sum(abs(new_w-w)))
        if np.sum(abs(new_w-w))<=1e-4 or iter>=1000:
            break
        w=new_w
    
    newton_result=w

    
    print(newton_result)
    confusion_matrix_newton()
    
    draw(D1x=D1x,D1y=D1y,D2x=D2x,D2y=D2y)




    


    
