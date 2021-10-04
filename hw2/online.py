import os
import matplotlib.pyplot as plt


def read_file(path):
    content = []
    with open (path, 'r') as fp:
        for this_result in fp:
            content.append(this_result.strip())
    return content

def How_many_1_N(this_result):
    l=len(this_result)
    m=0
    for i in range(l):
        if this_result[i] == '1':
            m=m+1
    
    return m,l

def Facotrial(n):
    return n * Facotrial(n - 1) if n > 1 else 1

def calculate(m,N):
    likelihood=Facotrial(N)/(Facotrial(m)*Facotrial(N-m))
    likelihood=likelihood*pow(m/N,m)*pow(1-m/N,(N-m))

    return likelihood


def draw_prior(a,b):
    if a==0 or b==0:
        print("")
    else:
        x=[]
        y=[]
        p=0.01
        for i in range(99):
            prior=pow(p,a-1) * pow(1-p,b-1) * Facotrial(a+b-1)/( Facotrial(a-1) * Facotrial(b-1) )
            y.append(prior)
            x.append(p)
            p+=0.01
        #plt.ylim(0,2)
        plt.title("Prior")
        plt.plot(x, y)
        plt.show()

def draw_likelihood(N,m):

    x=[]
    y=[]
    p=0
    for i in range(100):
        like=Facotrial(N) / (Facotrial(m) * Facotrial(N - m) )
        like=like *pow(p,m) * pow( 1-p , N-m )
        y.append(like)
        x.append(p)
        p+=0.01
    #plt.ylim(0,2)
    plt.title("likelihood")
    plt.plot(x, y)
    plt.show()

def draw_posterior(N,m,a,b):

    x=[]
    y=[]
    p=0
    for i in range(100):
        post=Facotrial(N + a + b-1) / (Facotrial(m+a-1) * Facotrial(N - m+ b-1 ) )
        post=post *pow(p,m + a - 1) * pow( 1-p , N - m + b - 1 )
        y.append(post)
        x.append(p)
        p+=0.01
    #plt.ylim(0,2)
    plt.title("Posterior")
    plt.plot(x, y)
    plt.show()

if __name__=='__main__':

    path="./testfile.txt"

    a=input("input a : ")
    b=input("input b : ")
    content = read_file(path=path)

    a=int(a)
    b=int(b)
    '''
    draw_prior(2,2)
    draw_likelihood(1,1)
    draw_posterior(1,1,2,2)
    #print(content)
    '''
    

    for i in range(len(content)):
        print("case",i+1 , ":  ",content[i])
        m,N=How_many_1_N(content[i])
        likelihood=calculate(m=m,N=N)
        print("Likelihood:",likelihood)
        print("Beta Prior:     a:",a,"  b:",b)
        a_p=a+m
        b_p=b+(N-m)

        print("Beta Posterior: a:",a_p,"  b:",b_p)
        print("\n")
        draw_prior(a,b)
        draw_likelihood(N=N,m=m)
        draw_posterior(a=a,b=b,N=N,m=m)

        a=a_p
        b=b_p
        


    




