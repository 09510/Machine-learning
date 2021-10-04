  
from PIL import Image
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import matplotlib.pyplot as plt
import numba as nb
import time
import os

np.set_printoptions(threshold=np.inf)
if not os.path.isdir('./visualization'):
    os.mkdir('./visualization')
if not os.path.isdir('./visualization2'):
    os.mkdir('./visualization2')


def read_input(filename):
    img = Image.open(filename)
    width, height = img.size
    pixel = np.array(img.getdata()).reshape((width*height, 3))

    coord = np.array([]).reshape(0, 2)
    for i in range(num):
        row_x = np.full(num, i)
        row_y = np.arange(num)
        row = np.array(list(zip(row_x, row_y))).reshape(1*num, 2)
        coord = np.vstack([coord, row])

    return pixel, coord

def get_gif(pics_dir,n):
    imgs = []
    files = os.listdir(pics_dir)
    for i in files:
        pic_name = os.path.join(pics_dir, i)
        print(pic_name)
        temp = Image.open(pic_name)
        imgs.append(temp)
    
    save_name =storename+"k_"+str(k)+"ini_method_"+initial_method + '{}.gif'.format(pics_dir)
    
    imgs[0].save(save_name, save_all=True, append_images=imgs, duration=400)
    return save_name

def k_mean_plus(data):
    
    mean=np.zeros((k,2))


    mean[0,0]=np.random.randint(low=0,high=100)
    mean[0,1]=np.random.randint(low=0,high=100)

    for u_num in range(1,k):
        all_distance=np.zeros((10000))

        for i in range(10000):
            x=i // 100
            y=i % 100
            this_dis=np.zeros(u_num)
            for j in range(u_num):
                this_dis[j]=(mean[j,0]-x) ** 2 + (mean[j,1]-y) ** 2
            
            all_distance[i]=this_dis[np.argmin(this_dis)]
        
        max_distance=np.argmax(all_distance)
        mean[u_num,0]=max_distance  // 100
        mean[u_num,1]=max_distance %100
    
    rand=np.zeros((k))

    for i in range(k):
        rand[i]=mean[i,0] + mean[i,1] *100

    return rand


def c_k_mean_plus(data):
    
    mean=np.zeros((k,2))
    total=256 ** 2 * 3

    mean[0,0]=np.random.randint(low=0,high=100)
    mean[0,1]=np.random.randint(low=0,high=100)

    for u_num in range(1,k):
        all_distance=np.zeros((10000))

        for i in range(10000):
            x=i // 100
            y=i % 100
            this_dis=np.zeros(u_num)
            for j in range(u_num):
                this_dis[j]=((mean[j,0]-x) ** 2 + (mean[j,1]-y) ** 2) *total / ( (data[j,0]-data[int(mean[j,0]),0]) ** 2 
                                + (data[j,1]-data[int(mean[j,1]),1]) ** 2 +(data[j,2]-data[int(mean[j,1]),2]) ** 2)
            
            all_distance[i]=this_dis[np.argmin(this_dis)]
        
        max_distance=np.argmax(all_distance)
        mean[u_num,0]=max_distance // 100
        mean[u_num,1]=max_distance %100
    
    rand=np.zeros((k))

    for i in range(k):
        rand[i]=mean[i,0] + mean[i,1] *100

    return rand

def find_center(center,i,index_of_pixel):

    index=index_of_pixel[i]
    x=index[0]
    y=index[1]

    min_dis=10000000000
    min_center=-1

    for center_num in range(k):


        c_x=index_of_pixel[int(center[center_num])][0]
        c_y=index_of_pixel[int(center[center_num])][1]

        this_dis=((x-c_x) ** 2) + ((y-c_y) ** 2)

        if this_dis<min_dis:
            min_dis=this_dis
            min_center=center_num

    return min_center

def initial(data,index_of_pixel, initial_method):
    this_ri=np.copy(ri)
    this_aik=np.copy(aik)
    this_Ck=np.zeros((k))

    
    
    if initial_method == 'random':
        rand=np.random.randint(low=0,high=10000,size=(1 * k))
        print(rand)

        for i in range(pixel_num):
            cluster=find_center(center=rand,i=i,index_of_pixel=index_of_pixel)  
            this_ri[i]=cluster       
            this_aik[i][cluster] = 1
            this_Ck[cluster] += 1
        return this_ri,this_aik,this_Ck,rand
    elif initial_method == "kmean++":
        rand=k_mean_plus(data)
        print(rand)
        for i in range(pixel_num):
            cluster=find_center(center=rand,i=i,index_of_pixel=index_of_pixel)  
            this_ri[i]=cluster       
            this_aik[i][cluster] = 1
            this_Ck[cluster] += 1
        return this_ri,this_aik,this_Ck,rand
    elif initial_method=="c-kmean++":
        rand=c_k_mean_plus(data)
        print(rand)
        for i in range(pixel_num):
            cluster=find_center(center=rand,i=i,index_of_pixel=index_of_pixel)  
            this_ri[i]=cluster       
            this_aik[i][cluster] = 1
            this_Ck[cluster] += 1
        return this_ri,this_aik,this_Ck,rand

        

            



def kernel(pixel,coord):
    spatial_sq_dists = squareform(pdist(coord, 'sqeuclidean'))
    spatial_rbf = np.exp(-gamma_s * spatial_sq_dists)


    color_sq_dists = squareform(pdist(pixel, 'sqeuclidean'))
    color_rbf = np.exp(-gamma_c * color_sq_dists)
    kernel = spatial_rbf * color_rbf


    return kernel
def visualization(ri,iteration):

    img = Image.open(filename)
    width, height = img.size
    pixel = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixel[j, i] = color[int(ri[i * num + j])]
    img.save(storename+"method_"+initial_method+"_K-" +str(k)+ '_'+str(time.time())+'_' + str(iteration) + '.png')

@nb.jit
def E_step(aik,Ck,all_kernel):
    #print("E_STEP")
    objective=np.zeros((10000,k))
    


    #gram_martrix
    third=np.zeros((k))
    for j in range(k):
        for m in range(10000):
            for n in range(10000):
                third[j] += aik[m][j] * aik[n][j] *all_kernel[m][n]
    

    for i in range(10000):
        for j in range(k):
            first=all_kernel[i][i]
            second=0
            for m in range(10000):
                second += aik[m][j]*all_kernel[i][m]
            
            
            objective[i][j]=first - 2 * second / (Ck[j]+1) + third[j]/((Ck[j]+1)**2)
        
    
    this_ri=calculate_ri(objective)

    return this_ri

#@nb.jit
def M_step(this_ri):
    #print("M_STEP")
    this_Ck=np.zeros((k),dtype='int64')
    this_aik=np.zeros((10000,k),dtype='int64')
    for i in range(pixel_num):
        i_cluster=int(this_ri[i])
        this_aik[i][i_cluster] = 1 
        this_Ck[i_cluster] += 1
    
    return this_aik,this_Ck

#@nb.jit
def calculate_ri(objective):
    ri=np.zeros((10000))
    miniz=np.argmin(objective,axis=1)
    for i in range(10000):
        ri[i]=miniz[i]
        
    return ri
    

def calculate_error(this_ri,pre_ri):
    error=0
    for i in range(len(this_ri)):

        if this_ri[i]!=pre_ri[i]:
            error+=1
    
    return error
    

def cluster(pixel,index_of_pixel):
    #initial
    this_ri,this_aik,this_Ck,mu= initial(pixel,index_of_pixel,initial_method)
    all_kernel=kernel(pixel,index_of_pixel)
    
    for i in range(len(mu)):
        x=mu[i] // 100
        y=mu[i] % 100

        print("mean is :" ,x,y)

    itera=1
    for i in range(epochs):
        itera+=1

        pre_ri=np.copy(this_ri)
        #E_step
        this_ri=E_step(this_aik,this_Ck,all_kernel)
        visualization(this_ri,iteration=i)
        #M_step
        this_aik,this_Ck=M_step(this_ri)

        error=calculate_error(this_ri,pre_ri)
        if(error<=10):
            break


    
    visualization(this_ri,iteration='final_iteration')

    return itera


k=5
epochs = 200
gamma_c = 1/(255*255)
gamma_s = 1/(100*100)

num = 100
color = [(0,0,0), (100, 0, 0), (0, 255, 0), (255,255,255),(255,0,0),(0,0,255),(255,0,255),(0,255,255),(255,255,0)]
gif_num=0


Ck=np.zeros((k),dtype='int64')
aik=np.zeros((10000,k),dtype='int64')
ri=np.zeros((10000),dtype='int64')
initial_method='kmean++'
pixel_num=10000

filename=""
storename=""



if __name__ == "__main__":


    filename = 'data/image1.png'
    storename = 'visualization/image1'
    pixel, index_of_pixel = read_input(filename)


    gif_num=cluster(pixel,index_of_pixel)
    get_gif('visualization',gif_num)

    
    filename = 'data/image2.png'
    storename = 'visualization2/image2'
    pixel, index_of_pixel = read_input(filename)


    gif_num=cluster(pixel,index_of_pixel)
    get_gif('visualization2',gif_num)

    