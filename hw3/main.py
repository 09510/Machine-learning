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

def Polynomial_Linear_Generator(n,a,w):
    print("Polynomial basis linear model data generator")
    x=np.random.uniform(-1.0, 1.0, 1)
    y=0

    e=Univariate_Gaussian_Generator(0,a)
    point=0
    for i in range(n):
        point+=w[i] * (x ** i)
        
    y=point+e

    return x,y
    
def variance(n, total, square):
    mean_square = square / n
    mean = total / n
    var=mean_square - mean ** 2

    return var



if __name__ == "__main__":

    print("start")

    m=input("m : ")
    s=input("s : ")
    print("Data point source function: N(", m , "," , s , ")")
    
    m=np.float(m)
    s=np.float(s)
    
    n = 0
    total = 0
    square_total = 0
    this_varience = 0
    last_varience = 0
    # n = add = addsq = 0
    count=0

    while True:
        count+=1
        point = Univariate_Gaussian_Generator(mean=m, s=s)
        print("Add data point:",point)

        n += 1
        total += point
        square_total += point ** 2
        this_varience=variance(n=n,total=total,square=square_total)
        print("Mean = ", total/n ," Variance = ",this_varience)
        if count>=100:
            if abs(this_varience - last_varience) <= 1e-7:
                break

        last_varience=this_varience