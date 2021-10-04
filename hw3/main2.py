from main import Polynomial_Linear_Generator
import numpy as np
import math
import matplotlib.pyplot as plt




def first_iterate(this_n,this_w,this_var,this_b):
    prior_mean = 0
    prior_var_inv = this_b
    x,y = Polynomial_Linear_Generator(n = this_n, w = this_w, a = var)
    design = design_matrix(this_n,x)

    #Fist iteration
    at_a=np.dot(design.T,design)
    posterior_var_inv = this_var * at_a + prior_var_inv * np.eye(this_n)
    ata_inverse=np.linalg.inv(posterior_var_inv)
    posterior_mean = this_var * np.dot(ata_inverse, design.T) * y
    predictive_distribution_mean = np.dot(design, posterior_mean)
    predictive_distribution_variance = 1 / this_var + np.dot(np.dot(design, np.linalg.inv(posterior_var_inv)), design.T)



    print("Add data point (",x,",", y,"):")
    print("")
    print("Posterior mean:")
    print(posterior_mean)
    print("")
    print("Posterior covariance:")
    print(np.linalg.inv(posterior_var_inv))
    print("")
    print("Predictive distribution ~ N(",predictive_distribution_mean,",",predictive_distribution_variance,")")
    print("--------------------------------------------------")

    return x,y,posterior_mean,posterior_var_inv
    

def else_iterate(this_n,this_w,this_var , posterior_mean , posterior_var_inv):
    end=False
    prior_mean = posterior_mean.copy()
    prior_var_inv = posterior_var_inv.copy()
    x,y = Polynomial_Linear_Generator(n = this_n, w = this_w, a = var)
    design = design_matrix(this_n,x)
    all_x.append(x)
    all_y.append(y)

    #Fist iteration
    at_a=np.dot(design.T,design)
    posterior_var_inv = this_var * at_a + prior_var_inv 

    #posterior_mean = this_var * np.dot(np.linalg.inv(posterior_var_inv), design.T) * y
    posterior_mean = np.dot(np.linalg.inv(posterior_var_inv), (this_var * design.T * y + np.dot(prior_var_inv, prior_mean)))

    predictive_distribution_mean = np.dot(design, posterior_mean)
    predictive_distribution_variance = 1 / this_var + np.dot(np.dot(design, np.linalg.inv(posterior_var_inv)), design.T)
    
    #print(predictive_distribution_variance)
    print("Add data point (",x,",", y,"):")
    print("")
    print("Posterior mean:")
    print(posterior_mean)
    print("")
    print("Posterior covariance:")
    print(np.linalg.inv(posterior_var_inv))
    print("")
    print("Predictive distribution ~ N(",predictive_distribution_mean,",",predictive_distribution_variance,")")
    print("--------------------------------------------------")


    if (abs(np.sum(prior_mean - posterior_mean)) < 1e-4) and (abs(np.sum(np.linalg.inv(prior_var_inv) - np.linalg.inv(posterior_var_inv))) < 1e-6):
        end=True
    return x, y, posterior_mean,posterior_var_inv,end



def generate_y(w,x):
    y=[]
    
    for acount in range(len(x)):
        now=0
        for i in range(len(w)):
            now+=pow(x[acount],i) *w[i]
        y.append(now)
    
    return y







def draw(this_w,this_var,this_n,posterior_mean,posterior_var_inv):
    

    
    ground_x = np.linspace(-2.0, 2.0, 30)
    predict_x = np.linspace(-2.0, 2.0, 30)
    
    #ground truth
    fig = plt.figure()
    plt.ylim(-15, 25)
    plt.xlim(-2.0, 2.0)
    plt.title("Ground Truth")
    ground_y=generate_y(w=this_w,x=ground_x)
    
    #print(ground_y)

    plt.plot(ground_x, ground_y, color = 'black')
    #mean + variance
    for i in range(len(ground_y)):
        ground_y[i]+=var
    plt.plot(ground_x, ground_y, color = 'red')
     #mean + variance - 2 * variance
    for i in range(len(ground_y)):
        ground_y[i]-= 2 * var
    plt.plot(ground_x, ground_y, color = 'red')

    


    #predict result
    fig = plt.figure()
    plt.xlim(-2.0, 2.0)
    plt.ylim(-15, 25)
    plt.title("Predict result")
    predict_y = generate_y(w=posterior_mean,x=predict_x)
    predict_y_upper_bound = generate_y(w=posterior_mean,x=predict_x)
    predict_y_lower_bound = generate_y(w=posterior_mean,x=predict_x)

    for i in range(len(predict_x)):
        predict_design_matrix = design_matrix(this_n,predict_x[i])
        predict_predictive_distribution_variance = 1 / this_var + np.dot(np.dot(predict_design_matrix, np.linalg.inv(posterior_var_inv)), predict_design_matrix.T)
        predict_y_upper_bound[i] += predict_predictive_distribution_variance[0]
        predict_y_lower_bound[i] -= predict_predictive_distribution_variance[0]

    plt.plot(predict_x, predict_y, color = 'black')
    plt.plot(predict_x, predict_y_upper_bound, color = 'red')
    plt.plot(predict_x, predict_y_lower_bound, color = 'red')
    plt.scatter(all_x, all_y)




    #10 incomes
    fig = plt.figure()
    plt.xlim(-2.0, 2.0)
    plt.ylim(-15, 25)
    plt.title("After 10 incomes")
    predict_y = generate_y(w=ten_mean,x=predict_x)
    predict_y_upper_bound = generate_y(w=ten_mean,x=predict_x)
    predict_y_lower_bound = generate_y(w=ten_mean,x=predict_x)

    for i in range(len(predict_x)):
        predict_design_matrix = design_matrix(this_n,predict_x[i])
        predict_predictive_distribution_variance = 1 / ten_var + np.dot(np.dot(predict_design_matrix, np.linalg.inv(ten_var_inv)), predict_design_matrix.T)
        predict_y_upper_bound[i] += predict_predictive_distribution_variance[0]
        predict_y_lower_bound[i] -= predict_predictive_distribution_variance[0]

    plt.plot(predict_x, predict_y, color = 'black')
    plt.plot(predict_x, predict_y_upper_bound, color = 'red')
    plt.plot(predict_x, predict_y_lower_bound, color = 'red')
    plt.scatter(ten_x, ten_y)


    #50 incomes
    fig = plt.figure()
    plt.xlim(-2.0, 2.0)
    plt.ylim(-15, 25)
    plt.title("After 50 incomes")
    predict_y = generate_y(w=fifty_mean,x=predict_x)
    predict_y_upper_bound = generate_y(w=fifty_mean,x=predict_x)
    predict_y_lower_bound = generate_y(w=fifty_mean,x=predict_x)

    for i in range(len(predict_x)):
        predict_design_matrix = design_matrix(this_n,predict_x[i])
        predict_predictive_distribution_variance = 1 / fifty_var + np.dot(np.dot(predict_design_matrix, np.linalg.inv(fifty_var_inv)), predict_design_matrix.T)
        predict_y_upper_bound[i] += predict_predictive_distribution_variance[0]
        predict_y_lower_bound[i] -= predict_predictive_distribution_variance[0]

    plt.plot(predict_x, predict_y, color = 'black')
    plt.plot(predict_x, predict_y_upper_bound, color = 'red')
    plt.plot(predict_x, predict_y_lower_bound, color = 'red')
    plt.scatter(fifty_x, fifty_y)

    plt.show()    


def design_matrix(n,x):
    A = []
    for i in range(n):
        A.append(x ** i)

    return np.array(A).reshape(1, -1)


def calculate_var(x,y,pos_mean):
    
    square=0
    total=0
    for i in range(len(x)):
        yi=0
        for j in range(len(pos_mean)):
            yi+=pos_mean[j] * (x[i]**j)
        res=y[i]-yi

        total+=res
        square+=res**2
    square_mean=square/len(x)
    total_mean=total/len(x)

    varience=square_mean-pow(total_mean,2)
    if varience==0:
        varience=0.00001
    return varience

all_x = []
all_y = []
ten_x=[]
ten_y=[]
fifty_x=[]
fifty_y=[]
ten_mean = np.array([])
fifty_mean = np.array([])
ten_var_inv = np.array([])
fifty_var_inv = np.array([])

ten_var=0
fifty_var=0

b=0
n=0
var=0
w=""


if __name__ == "__main__":
    
    b=input("b: ")
    n=input("n: ")
    var=input("a: ")
    w=input("w:")
    w=w.split(",")
    '''
    b=100
    n=4
    var=1
    w=[1,2,3,4]
    '''


    b=float(b)
    n= int(n)
    var=float(var)
    for  i in range(len(w)):
        w[i]=float(w[i])



    this_n = n
    this_mean = 0
    this_var = 0.001
    last_var=0
    this_w = w
    this_b = b

    



    count = 1
    x , y , posterior_mean , posterior_var_inv =first_iterate(this_b=this_b,this_n=this_n,this_var=this_var,this_w=this_w)
    all_x.append(x)
    all_y.append(y)
    
    while True:
        print(count)
        last_var=this_var
        this_var=1/calculate_var(all_x,all_y,pos_mean=posterior_mean)
        count+=1
        x ,y , posterior_mean ,posterior_var_inv ,end = else_iterate(this_w=this_w,this_var=this_var,this_n=this_n,posterior_mean=posterior_mean,posterior_var_inv=posterior_var_inv)
        all_x.append(x)
        all_y.append(y)

        if count == 10:
            ten_var=this_var
            ten_x=all_x.copy()
            ten_y=all_y.copy()
            ten_mean=posterior_mean.copy()
            ten_var_inv=posterior_var_inv.copy()
        elif count ==50:
            fifty_var=this_var
            fifty_x=all_x.copy()
            fifty_y=all_y.copy()
            fifty_mean=posterior_mean.copy()
            fifty_var_inv=posterior_var_inv.copy()


        #代表已經收斂
        if (end and  count>1000 and abs(this_var-last_var)<1e-4)or count>50000:
            break
    draw(this_n=this_n,this_var=this_var,this_w=this_w,posterior_mean=posterior_mean,posterior_var_inv=posterior_var_inv)

    
    
    
        
    

