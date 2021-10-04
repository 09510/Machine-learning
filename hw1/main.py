import matplotlib.pyplot as plt
import numpy as np




def import_plot(path):
    
    if path=='1':
        path="require.txt"
    
    '''
    print("import txt")
    '''
    content = np.loadtxt(path,dtype=float,delimiter=',')
    
    return content




def LU_Inverse(martrix):
    #print("do LU")

    martrix=np.array(martrix)
    row=np.size(martrix,0)
    col=np.size(martrix,1)
    
    #做LU分解
    (L,U)=LU_Decomp(martrix)

    #用方塊迭代法找到上三角與下三角矩陣的反矩陣
    L_inverse = iterate_inverse(L)
    U_inverse = iterate_inverse(U)


    martrix_inverse = np.dot(U_inverse,L_inverse)
    '''
    print(martrix_inverse.round(3))
    '''
    return martrix_inverse
    


def LU_Decomp(martrix):
    #print("do lu_decomp~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    martrix=np.array(martrix,dtype = float)
    row=np.size(martrix,0)
    col=np.size(martrix,1)


    #LA=U
    A=martrix[:,:]
    L=np.ndarray([row,col],dtype = float)
    U=np.ndarray([row,col],dtype = float)
    L=np.identity(row)#L初始為I
    U=A#初始化U為A,
            
    



    #列運算
    for i in range(0,row):
        for j in range(i+1,row):
            if U[j,i]!=0:
                coff=U[j,i]
                basic=U[i,i]
                coff=coff/basic
                '''
                debug="第 "+str(j)+" 列不是0，所以用第 "+str(i)+" 列乘上 "+str(coff)+" 倍後減去該"
                print(U[j,:])
                print(U[i,:])
                print(debug)
                '''
                
                #計算L
                L[j,i]=coff
                '''
                print(L.round(3))
                '''
                #計算U
                U[j,:]=U[j,:]-U[i,:]*coff
                '''
                print(U.round(3))
                '''

    return (L,U)





def iterate_inverse(martrix):
    #print("迭代求反矩陣")
    
    #初始化與賦值
    martrix=np.array(martrix)
    row=np.size(martrix,0)
    col=np.size(martrix,1)
    martrix_inverse=np.ndarray([col,row])
    



    #1，2階矩陣直接運算後回傳
    if row==1:
        martrix_inverse=martrix[0,0]
        martrix_inverse=1/martrix_inverse
        return martrix_inverse
    elif row==2:
        '''
        print(martrix.round(2))
        '''
        martrix_inverse[0,0]=martrix[1,1]
        martrix_inverse[0,1]=martrix[0,1]*-1
        martrix_inverse[1,0]=martrix[1,0]*-1
        martrix_inverse[1,1]=martrix[0,0]
        
        det=martrix[1,1]*martrix[0,0]
        martrix_inverse=martrix_inverse/det
        return martrix_inverse
    

    
    #開始方塊迭代
    A=np.ndarray([1,1])
    C=np.ndarray([1,col-1])
    B=np.ndarray([row-1,col-1])
    D=np.ndarray([1,col-1])

    A[0,0]=martrix[0,0]
    C=martrix[0,1:]
    B=martrix[1:,1:]
    D=martrix[1:,0]

    

    is_up=False
    for i in range(0,col):
        for j in range(i+1,row):
            if martrix[i,j]!=0:
                '''
                print("up!!!!!!!!")
                '''
                is_up=True


    if is_up:
        '''
        print("上三角")
        '''
        A_inverse=iterate_inverse(A)
        B_inverse=iterate_inverse(B)

        martrix_inverse[0,0]=A_inverse
        martrix_inverse[0,1:]=-1*np.dot(A_inverse,C)
        martrix_inverse[0,1:]=np.dot(martrix_inverse[0,1:],B_inverse)
        martrix_inverse[1:,1:]=B_inverse
        martrix_inverse[1:,0]=0
        '''
        print(martrix_inverse.round(2))
        '''
        return martrix_inverse
    else:
        '''
        print("下三角")
        print(A)
        print(B)
        print(C)
        print(D)
        '''
        A_inverse=iterate_inverse(A)
        B_inverse=iterate_inverse(B)

        martrix_inverse[0,0]=A_inverse
        martrix_inverse[1:,0]=-1*np.dot(B_inverse,D)
        martrix_inverse[1:,0]=np.dot(martrix_inverse[1:,0],A_inverse)
        martrix_inverse[1:,1:]=B_inverse
        martrix_inverse[0,1:]=0
        '''
        print(martrix_inverse.round(2))
        '''
        return martrix_inverse





def LSE(n,lamda,content):
    '''
    print("do LSE.")
    '''
    data_acount=np.size(content,0)


    A=np.ndarray([data_acount,n])
    b=np.ndarray([data_acount,1])
    

    for j in range(n):
        for i in range(data_acount): 
            A[i,j]=pow(content[i,0],(n-j-1))
    
                
    for i in range(data_acount): 
            b[i,0]=(content[i,1])

    

    I=np.identity(n)
    
    #x=(ATA)^-1ATb  
    x=np.dot(A.T,A)
    x=x+I*lamda
    x=LU_Inverse(x)
    x=np.dot(x,A.T)
    x=np.dot(x,b)
    '''
    print(x)
    '''
    return x




def Newton(n,content):
    '''
    print("do Newton's Method.")
    '''
    #設定初始值
    x_0=np.random.rand(n,1)
    x_1=np.random.rand(n,1)
    #設定誤差範圍
    error=0.00000001
    err=100
    data_acount=np.size(content,0)

    A=np.ndarray([data_acount,n])
    b=np.ndarray([data_acount,1])

    for j in range(n):
        for i in range(data_acount): 
            A[i,j]=pow(content[i,0],(n-j-1))
    
                
    for i in range(data_acount): 
            b[i,0]=(content[i,1])

    i=0

    while err>error:
        
        #算一階導數
        f_one=np.dot(A.T,A)
        f_one=f_one*2
        f_one=np.dot(f_one,x_0)
        
        aa=np.dot(A.T,b)
        aa=aa*2

        f_one=f_one-aa

        #算二階導術
        f_two=np.dot(A.T,A)
        f_two=f_two*2
        #牛頓推進
        step=LU_Inverse(f_two)
        step=np.dot(step,f_one)

        x_1=x_0-step
        
        #算error
        L=np.dot(A.T,A)
        L=np.dot(L,x_1)
        L=L*2

        atb=np.dot(A.T,b)
        atb=atb*2

        L=L-atb
        '''
        print(x_1)
        '''
        err=L.sum()
        if err<0:
            err=-1*err
        i=i+1
        if i==100:
            break

        x_0=x_1

    return x_0


def calculate_loss(A,b,x):
    #print(A)
    #xt at a x
    j=np.dot(x.T,A.T)
    j=np.dot(j,A)
    j=np.dot(j,x)

    #xt at b
    k=np.dot(x.T,A.T)
    k=np.dot(k,b)

    #bt a x
    l=np.dot(b.T,A)
    l-np.dot(l,x)

    #bt b
    p=np.dot(b.T,b)

    y=j-k
    y=y-l
    y=y+p

    return y
    

 

def pow(num,exp):
    x=1
    for i in range(0,exp):
        x=x*num
    return x


def result(name,weight,lamda,content):
    if name=="LSE":
        print(name+" : ")
        M=np.size(weight)
        line="Fitting line : "
        error="Total error : "
        for i in range(0,M):
            w=weight[i]
            if line!="Fitting line : ":
                if w<0:
                    line=line+" - "
                else:
                    line=line+" + "
            w=str(w)
            if w[1]=='-':
                w=w[2:-1]
            else:
                w=w[1:-1]


            line=line+w+"X^"+str(M-i-1)
        
        err=count_error(weight,lamda,content)
        error=error+str(err)
        print(line)
        print(error)
    elif name=="Newton":
        print("Newton's Method : ")
        M=np.size(weight)
        line="Fitting line : "
        error="Total error : "
        for i in range(0,M):
            w=weight[i]
            if line!="Fitting line : ":
                if w<0:
                    line=line+" - "
                else:
                    line=line+" + "
            w=str(w)
            if w[1]=='-':
                w=w[2:-1]
            else:
                w=w[1:-1]


            line=line+w+"X^"+str(M-i-1)
        
        
        err=count_error(weight,lamda,content)
        error=error+str(err)
        print(line)
        print(error)




def count_error(weight,lamda,content):
    data_acount=np.size(content,0)


    A=np.ndarray([data_acount,n])
    b=np.ndarray([data_acount,1])
    

    for j in range(n):
        for i in range(data_acount): 
            A[i,j]=pow(content[i,0],(n-j-1))
    
                
    for i in range(data_acount): 
            b[i,0]=(content[i,1])

    
    #(AX-b)^2+L|w|^2
    err=np.dot(A,weight)
    err=err-b
    err=np.dot(err.T,err)
    '''
    re=np.dot(weight.T,weight)
    re=weight*lamda

    err=err+re
    '''
    total_err=err.sum()

    return total_err





def draw(plot,Weight_1,Weight_2,name_1,name_2):
    '''
    print("Drawing~~~~~~~~~~~~~")
    '''
    maximum=np.max(plot[:,0])
    minmum=np.min(plot[:,0])

    leg=(maximum-minmum)/10

    maximum=maximum+leg
    minmum=minmum-leg



    #第一個子圖
    plt.subplot(211)
    plt.title(name_1)

    x = np.linspace(minmum,maximum,10)
    y=[f(i,Weight_1) for i in x]
    plt.plot(x,y)
    plt.scatter(plot[:,0],plot[:,1],c='red')


    #第二個子圖
    plt.subplot(212)
    plt.title(name_2)

    x = np.linspace(minmum,maximum,10)
    y=[f(i,Weight_2) for i in x]
    plt.plot(x,y)
    plt.scatter(plot[:,0],plot[:,1],c='red')

    

    plt.show()


def f(x,weight):
    '''
    print(weight)
    '''
    M=np.size(weight)
    y=0
    '''
    print(x)
    '''
    for i in range(0,M):
        c=weight[i]
        exp=M-i-1
        y=y+pow(x,exp)*c
    '''
    print(y)  
    '''
    return y

    


if __name__ == "__main__":
    
    #輸入
    print("===============start=============== ")
    path=input("input path:")
    n=input("input number of polynomial basis n: ")
    lamda=input("input lambda: ")

    n=int(n)
    lamda=int(lamda)

    #匯入資料
    content=import_plot(path)

    #用LSE找到fitting line 的權重
    weight_LSE=LSE(n=n,lamda=lamda,content=content)
    result("LSE",weight_LSE,lamda,content)
    
    print("")


    #用牛頓法找到fitting line 的權重
    weight_Newton=Newton(n=n,content=content)
    result("Newton",weight_Newton,lamda,content)


    #畫圖
    draw(content,weight_LSE,weight_Newton,"LSE","Newton")
    


    
