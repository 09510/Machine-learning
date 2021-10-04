from PIL import Image
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import matplotlib.pyplot as plt
import time
import os
import scipy.spatial


np.set_printoptions(suppress=True,threshold=np.inf)
num = 100
epochs = 150
K = 5
gamma_c = 1/1000
gamma_s = 1/1000
color = [(0,0,0), (100, 0, 0), (0, 255, 0), (255,255,255),(255,0,0),(0,0,255),(255,0,255),(0,255,255),(255,255,0)]


Initial_method='random'
filename = ''
storename = ''
gif_num=0


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

def compute_kernel(color, coord):

	spatial_sq_dists = squareform(pdist(coord, 'sqeuclidean'))
	spatial_rbf = np.exp(-gamma_s * spatial_sq_dists)

	color_sq_dists = squareform(pdist(color, 'sqeuclidean'))
	color_rbf = np.exp(-gamma_c * color_sq_dists)

	kernel = spatial_rbf * color_rbf

	return kernel



# 初始化
def initial(data, initial_method):
	prev_ri = np.random.randint(K, size=data.shape[0])

	if initial_method == 'random':
		mu=np.zeros((K,K))
		rand = np.random.randint(10000, size=K)

		for u_num in range(K):
			this_data=data[rand[u_num],:]
			for dim in range(K):
				mu[u_num][dim]=this_data[dim]

		return rand,mu, prev_ri
	elif initial_method == 'Kmeans++':
		mu=np.zeros((K,K))
		rand = np.random.randint(10000, size=K)



		center=[np.random.choice(range(10000))]
		centroids = [data[center[-1], :]]
		for i in range(K - 1):
			dist = scipy.spatial.distance.cdist(data, centroids, 'euclidean').min(axis=1)
			prob = dist / np.sum(dist)
			center.append(np.random.choice(range(10000), p=prob))
			centroids.append(data[center[-1]])

		centroids = np.array(centroids)
		return center, centroids, prev_ri


	



#E_step
def E_step(data, mu):
	ri = np.zeros((10000))
	for dataidx in range(10000):
		distance = np.zeros(K)
		for cluster in range(K):
			
			minus=data[dataidx,:]- mu[cluster,:]
			minus=abs(minus)
			distance[cluster]=np.square(minus).sum(axis=0)
			
		ri[dataidx] = np.argmin(distance)
	return ri


#M_step
def M_step(data, mu, ri):


	new_mu = np.zeros(mu.shape)
	total = np.zeros(mu.shape)
	plus = np.ones(mu.shape[1])

	for dataidx in range(10000):
		new_mu[int(ri[dataidx])] += data[dataidx]
		total[int(ri[dataidx])] += plus

	#防止zero devide
	for i in range(K):
		if total[i][0] == 0:
			total[i] += plus

	new_mu=new_mu / total

	print(new_mu)

	return new_mu


# converge condition
def calculate_error(ri, prev_ri):
	error = 0
	for i in range(10000):
		if ri[i] != prev_ri[i]:
			error+=1

	return error


def visualization(ri,iteration,ini_m):

	
	img = Image.open(filename)
	width, height = img.size
	pixel = img.load()
	for i in range(img.size[0]):
		for j in range(img.size[1]):
			pixel[j, i] = color[int(ri[i * num + j])]
	img.save(storename+"method_"+Initial_method+"_K-" +str(K)+ '_'+str(time.time())+'_' + str(iteration) + '.png')


def draw_eigenspace(filename, storename, iteration, ri, initial_method, data):
	
	fig=plt.figure()
	ax=fig.add_subplot(111,projection='3d')
	markers=['o','^','s']
	
	for marker,i in zip(markers,np.arange(3)):
		ax.scatter(data[ri==i,0],data[ri==i,1],data[ri==i,2],marker=marker)

	ax.set_xlabel('1st dim')
	ax.set_ylabel('2nd dim')
	ax.set_zlabel('3rd dim')
	plt.show()


def K_Means(data, filename, storename):

	initial_method=Initial_method
	center, mean, ri = initial(data, initial_method)
	mu=np.zeros((K,K))
	print(center)
		
	for u_num in range(K):
		mu[u_num,:]=data[int(center[u_num]),:]

	print(mu)
	error = -10000
	prev_error = -10001

	iteration = 0
	while(iteration <= epochs):
		iteration += 1
		prev_ri = ri
		
		
		#Estep
		ri = E_step(data, mu)
		visualization(ri,iteration,initial_method)
		#Mstep
		mu = M_step(data, mu, ri)
		
		error = calculate_error(ri, prev_ri)
		print(error)
		if error == prev_error:
			break
		prev_error = error
	
	draw_eigenspace(filename, storename, iteration, ri, initial_method, data)
	return iteration



def normalized_cut(pixel, coord):

	if filename == 'data/image1.png':
		print("img1")
		try:
			W=np.load('W.npy')
			D=np.load('D.npy')
			D_root=np.load('D_root.npy')
			L_sym=np.load('L_sym.npy')
			eigen_values=np.load('eigen_values.npy')
			eigen_vectors=np.load('eigen_vectors.npy')
		except:
			W = compute_kernel(pixel, coord)
			D = np.diag(np.sum(W, axis=1))
			D_root = np.diag(np.power(np.diag(D), -0.5))
			L_sym = np.eye(W.shape[0]) - D_root @ W @ D_root
			eigen_values, eigen_vectors = np.linalg.eig(L_sym)
			np.save('W', W)
			np.save('D', D)
			np.save('D_root', D_root)
			np.save('L_sym',L_sym)
			np.save('eigen_values',eigen_values)
			np.save('eigen_vectors',eigen_vectors)
	elif filename == 'data/image2.png':
		print("img2")
		try:
			W=np.load('W2.npy')
			D=np.load('D2.npy')
			D_root=np.load('D_root2.npy')
			L_sym=np.load('L_sym2.npy')
			eigen_values=np.load('eigen_values2.npy')
			eigen_vectors=np.load('eigen_vectors2.npy')
		except:
			W = compute_kernel(pixel, coord)
			D = np.diag(np.sum(W, axis=1))
			D_root = np.diag(np.power(np.diag(D), -0.5))
			L_sym = np.eye(W.shape[0]) - D_root @ W @ D_root
			eigen_values, eigen_vectors = np.linalg.eig(L_sym)

			np.save('W2', W)
			np.save('D2', D)
			np.save('D_root2', D_root)
			np.save('L_sym2',L_sym)
			np.save('eigen_values2',eigen_values)
			np.save('eigen_vectors2',eigen_vectors)

	K_eigen_v = np.argsort(eigen_values)[1: K+1]
	U = eigen_vectors[:, K_eigen_v].real.astype(np.float32)


	# normalized
	normalize_sum = np.power(U, 2)
	normalize_sum = np.sum(normalize_sum, axis=1) ** 0.5
	normalize_sum = normalize_sum.reshape(-1, 1)


	T = U.copy()
	for i in range(normalize_sum.shape[0]):
		if normalize_sum[i][0] == 0:
			normalize_sum[i][0] = 1
		T[i][0] /= normalize_sum[i][0]
		T[i][1] /= normalize_sum[i][0]
	return T

def ratio_cut(pixel, coord):
	if filename == 'data/image1.png':
		print("img1")
		try:
			W_r=np.load('W_r.npy')
			D_r=np.load('D_r.npy')
			L_r=np.load('L_r.npy')
			eigen_values_r=np.load('eigen_values_r.npy')
			eigen_vectors_r=np.load('eigen_vectors_r.npy')
		except:

			W_r = compute_kernel(pixel, coord)
			D_r = np.diag(np.sum(W_r, axis=1))
			L_r = D_r - W_r
			eigen_values_r, eigen_vectors_r = np.linalg.eig(L_r)
			
			np.save('W_r',W_r )
			np.save('D_r', D_r)
			np.save('L_r', L_r)
			np.save('eigen_values_r',eigen_values_r)
			np.save('eigen_vectors_r',eigen_vectors_r)
	elif filename == 'data/image2.png':
		print("img2")
		try:
			W_r=np.load('W_r2.npy')
			D_r=np.load('D_r2.npy')
			L_r=np.load('L_r2.npy')
			eigen_values_r=np.load('eigen_values_r2.npy')
			eigen_vectors_r=np.load('eigen_vectors_r2.npy')
		except:

			W_r = compute_kernel(pixel, coord)
			D_r = np.diag(np.sum(W_r, axis=1))
			L_r = D_r - W_r
			eigen_values_r, eigen_vectors_r = np.linalg.eig(L_r)
			
			np.save('W_r2',W_r )
			np.save('D_r2', D_r)
			np.save('L_r2', L_r)
			np.save('eigen_values_r2',eigen_values_r)
			np.save('eigen_vectors_r2',eigen_vectors_r)


	
	idx = np.argsort(eigen_values_r)[1: K+1]
	U = eigen_vectors_r[:, idx].real.astype(np.float32)

	return U

def get_gif(pics_dir,n):
    imgs = []
    files = os.listdir(pics_dir)
    for i in files:
        pic_name = os.path.join(pics_dir, i)
        temp = Image.open(pic_name)
        imgs.append(temp)
    
    save_name =storename+"k_"+str(K)+"ini_method_"+Initial_method + '{}.gif'.format(pics_dir)
    
    imgs[0].save(save_name, save_all=True, append_images=imgs, duration=400)
    return save_name

if not os.path.isdir('./visualization'):
    os.mkdir('./visualization')
if not os.path.isdir('./visualization2'):
    os.mkdir('./visualization2')


if __name__ == '__main__':

    

	K=3
	Initial_method = "Kmeans++"
	cut_type="R"
	
	filename = 'data/image1.png'
	storename = 'visualization/image1'
	pixel1, coord1 = read_input(filename)

	#獲得特徵向量,(10000,K)
	if cut_type=="N":
		T = normalized_cut(pixel1, coord1)
	else:
		print("ratio")
		T = ratio_cut(pixel1,coord1)
	
	#把t當成是10000筆的3維資料直接做k_mean，所以公式也完全參照pdf
	gif_num=K_Means(T, filename, storename)
	get_gif('visualization',gif_num)

	###########################################################
	print("===================================================")
	filename = 'data/image2.png'
	storename = 'visualization2/image2'
	pixel2, coord2 = read_input(filename)
	if cut_type=="N":
		T = normalized_cut(pixel2, coord2)
	else:
		T=ratio_cut(pixel2,coord2)

	gif_num=K_Means(T, filename, storename)
	get_gif('visualization2',gif_num)