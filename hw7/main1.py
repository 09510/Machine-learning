  
import numpy as np
from PIL import Image
import os, re, sys
import scipy.spatial.distance
import time
import matplotlib.pyplot as plt

SHAPE = (50, 50)
kernels = ['linear kernel', 'polynomial kernel', 'rbf kernel']

def readPGM(filename):
    #print(filename)
    image = Image.open(filename)
    image = image.resize(SHAPE, Image.ANTIALIAS)
    image = np.array(image)
    label = int(re.findall(r'subject(\d+)', filename)[0])
    return image.ravel().astype(np.float64), label

def readData(path):            
    data = []
    filename = []
    label = []
    for pgm in os.listdir(path):
        res = readPGM(f'{path}/{pgm}')
        data.append(res[0])
        filename.append(pgm)
        label.append(res[1])
    return np.asarray(data), np.asarray(filename), np.asarray(label)




def draw( title, W):

    folder = f"{title}_{time.time()}"
    os.mkdir(folder)
    os.mkdir(f'{folder}/{title}')
    
    plt.clf()
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            plt.subplot(5, 5, idx + 1)
            plt.imshow(W[:, idx].reshape(SHAPE), cmap='gray')
            plt.axis('off')
    plt.savefig(f'./{folder}/{title}/{title}.png')

    for i in range(W.shape[1]):
        plt.clf()
        plt.title(f'{title}_{i + 1}')
        plt.imshow(W[:, i].reshape(SHAPE), cmap='gray')
        plt.savefig(f'./{folder}/{title}/{title}_{i + 1}.png')
    

    
    


def Recognition(train, train_label, test, test_label ):
    test_num = test.shape[0]
    train_num = train.shape[0]

    all_dist = []
    for i in range(test_num):
        dist = []
        # 計算與每個點之間的距離
        for j in range(train_num):
            this_distance=np.sum((train[j] - test[i]) ** 2)
            dist.append((this_distance, train_label[j]))
        # 依據距離排序
        dist.sort(key=lambda l: l[0])
        #記錄此test data 與其他nebighbor 的關係記錄下來
        all_dist.append(dist)
    k=3
    error = 0
    for i in range(test_num):
        dist = all_dist[i]
        #找到最近的k個當neighbor
        neighbor=dist[:k]
        n_list=[]
        for j in range(k):
            n_list.append(neighbor[j][1])
        neighbor=np.array(n_list)

        # 統計neighbor 的 lable
        neighbor_label, count = np.unique(neighbor, return_counts=True)
        # 最多的當result
        most_n=np.argmax(count)
        predict = neighbor_label[most_n]
        #驗證答案
        if predict != test_label[i]:
            error += 1
    print(f'accuracy: {(1 - error / test_num):>.3f} ({test_num-error}/{test_num})')
    print("==========================================================")


def kernelPCA(X, kernel_type):
    # 計算kernel
    if kernel_type == 1:
        kernel=X.T @ X
    elif kernel_type == 2:
        g=0.1
        cof=100
        degree=3
        kernel = g * (X.T @ X) + cof
        kernel = np.power(kernel,degree)
    else:
        g=1
        kernel = -1 * g * scipy.spatial.distance.cdist(X.T, X.T, 'sqeuclidean')
        kernel = np.exp(kernel)

    gram_matrix = kernel
    # 計算KC
    n = gram_matrix.shape[0]
    one = np.ones((n, n)) / n
    k_cov = gram_matrix - one @ gram_matrix - gram_matrix @ one + one @ gram_matrix @ one
    
    # 把這個KC做eigen
    eigen_val, eigen_vec = np.linalg.eigh(k_cov)
    for i in range(eigen_vec.shape[1]):
        eigen_vec[:, i] = eigen_vec[:, i] / np.linalg.norm(eigen_vec[:, i])
    
    idx = np.argsort(eigen_val)[::-1]
    W = eigen_vec[:, idx]
    W = W[:, :dims].real
    
    
    return  W

def PCA(X):
   

    # 找到mean
    mu = np.mean(X, axis=0)
    # 先標準化，mu變為0
    st_X = X - mu
    # 算covariance
    cov = st_X.T @ st_X
    # 找到cov的eigen
    eigen_val, eigen_vec = np.linalg.eigh(cov)

    # 排序後，取前 dims個 特徵向量回傳
    for i in range(eigen_vec.shape[1]):
        eigen_vec[:, i] = eigen_vec[:, i] / np.linalg.norm(eigen_vec[:, i])
    idx = np.argsort(eigen_val)[::-1]
    W = eigen_vec[:, idx]
    W = W[:, :dims].real
    
    
    return W





def LDA(X, label):
    (n, d) = X.shape
    label = np.asarray(label)

    c = np.unique(label)
    S_w = np.zeros((d, d), dtype=np.float64)
    S_b = np.zeros((d, d), dtype=np.float64)
    
    mu = np.mean(X, axis=0)

    # SW
    for i in c:
        X_i = X[np.where(label == i)[0], :]
        #mj
        mu_i = np.mean(X_i, axis=0)
        #(xi-mj)
        st_X_i = X_i-mu_i
        S_w += st_X_i.T @ st_X_i
    
    # SB
    for i in c: 
        X_i = X[np.where(label == i)[0], :]
        #mj
        mu_i = np.mean(X_i, axis=0)
        #(m-mj)
        st_mu_i = mu_i - mu
        #ni
        ni = X_i.shape[0]
        S_b += ni * (st_mu_i.T @ st_mu_i)


    eigen_val, eigen_vec = np.linalg.eig(np.linalg.pinv(S_w) @ S_b)
    for i in range(eigen_vec.shape[1]):
        eigen_vec[:, i] = eigen_vec[:, i] / np.linalg.norm(eigen_vec[:, i])
    idx = np.argsort(eigen_val)[::-1]
    W = eigen_vec[:, idx][:, :dims].real
    return W


def kernelLDA(X, label, kernel_type):
    label = np.asarray(label)
    c = np.unique(label)
    
    # compute kernel
    if kernel_type == 1:
        kernel=X.T @ X
    # polynomial
    elif kernel_type == 2:
        g=5
        cof=10
        degree=2
        kernel = g * (X.T @ X) + cof
        kernel = np.power(kernel,degree)
    # RBF
    else:
        g=1e-7
        kernel = -1 * g * scipy.spatial.distance.cdist(X.T, X.T, 'sqeuclidean')
        kernel = np.exp(kernel)
    
    n = kernel.shape[0]
    mu = np.mean(kernel, axis=0)
    N = np.zeros((n, n), dtype=np.float64)
    M = np.zeros((n, n), dtype=np.float64)
    
    # compute M
    for i in c:
        K_i = kernel[np.where(label == i)[0], :]
        l = K_i.shape[0]
        mu_i = np.mean(K_i, axis=0)
        M +=  ((mu_i - mu).T @ (mu_i - mu))
    # compute N
    for i in c:
        K_i = kernel[np.where(label == i)[0], :]
        l = K_i.shape[0]
        N += K_i.T @ (np.eye(l) - (np.ones((l, l), dtype=np.float64) / l)) @ K_i
    
    eigen_val, eigen_vec = np.linalg.eig(np.linalg.pinv(N) @ M)
    for i in range(eigen_vec.shape[1]):
        eigen_vec[:, i] = eigen_vec[:, i] / np.linalg.norm(eigen_vec[:, i])
    idx = np.argsort(eigen_val)[::-1]
    W = eigen_vec[:, idx][:, :dims].real

    return W

def show_reconstruction(X,X_recover,num,H,W):
    
    randint=np.random.choice(X.shape[0],num)
    for i in range(num):
        plt.subplot(2,num,i+1)
        plt.imshow(X[randint[i],:].reshape(H,W),cmap='gray')
        plt.subplot(2,num,i+1+num)
        plt.imshow(X_recover[randint[i],:].reshape(H,W),cmap='gray')
    plt.savefig(f'./reconstruct/reconstruction{str(SHAPE[0])}.png')
    plt.show()

dims=25

if __name__=="__main__":
    X, X_filename, X_label = readData('./Yale_Face_Database/Training')
    test, test_filename, test_label = readData('./Yale_Face_Database/Testing')


    data = np.vstack((X, test))
    filename = np.hstack((X_filename, test_filename))
    label = np.hstack((X_label, test_label))

    kernel_type=2
    
    
    # PCA
    # Q1
    W = PCA(data)
    draw( 'pca_eigenface', W)

    mu = np.mean(data, axis=0)
    re = (data) @ W @ W.T 
    show_reconstruction(data,re,10,SHAPE[0],SHAPE[1])



    # Q2
    W = PCA(X)
    print(W.shape)
    X_proj = X @ W
    test_proj = test @ W
    print('PCA:')
    Recognition(X_proj, X_label, test_proj,test_label)
    

    # Q3

    W  = kernelPCA(X, kernel_type)
    print(W.shape)
    X_proj = X @ W
    test_proj = test @ W
    
    print('Kernel PCA:')
    Recognition(X_proj, X_label, test_proj,test_label)
    
    
    
    # LDA
    # Q1
    
    
    U = LDA(data, label)
    print(U.shape)
    draw( 'lda_fisherface', U)
    mu = np.mean(data, axis=0)
    re = (data-mu) @ U @ U.T  +mu
    show_reconstruction(data,re,10,SHAPE[0],SHAPE[1])

    
    
    # Q2
    U=LDA(X,X_label)
    print(U.shape)
    X_proj = X @ U
    test_proj = test @ U
    print('LDA:')
    Recognition(X_proj, X_label, test_proj,test_label)
    
    # Q3

    U  = kernelLDA(X, X_label, kernel_type)
    print(U.shape)
    X_proj = X @ U
    test_proj = test @ U
    print('Kernel LDA:')
    Recognition(X_proj, X_label, test_proj,test_label)