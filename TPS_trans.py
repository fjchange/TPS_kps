import numpy as np
from PIL import Image,ImageFilter
import time

image_mean=np.array([[117.59180662198212, 106.95174844888551, 96.32777887186238]],dtype=float)

def cal_z(source_kps,target_kps):
    '''

    :param source_kps: source_kps with form [num_kps,2], the same size with target_kps
    :param target_kps: target_kps with form [num_kps,2]
    :return: get W,a,c the parameters of the interpolate function, we can use the parameters to reconstruct the function
            then
    '''
    n = source_kps.shape[0]
    P = np.ones([n, 3], dtype=float)
    for i in range(n):
        P[i, 1], P[i, 2] = source_kps[i]
    P_1 = P.reshape([n, 1, 3])
    P_2 = P.reshape([1, n, 3])
    d = np.sqrt(np.sum((P_1 - P_2) ** 2, 2))
    r = d * d * np.log(d * d + 1e-5)

    L = np.zeros([n + 3, n + 3], dtype=float)
    L[:n, :-3] = r
    L[:n, -3:] = P
    L[n:, :-3] = P.transpose()
    L_inverse=np.linalg.inv(L)
    tar_kps=np.concatenate([target_kps,np.zeros([3,2],dtype=float)])
    W=np.matmul(L_inverse,tar_kps)

    return W

def cal_new_i_j(W,i,j,source_kps):
    pts_2 = np.array([i, j], dtype=float)
    pts_3 = np.reshape(np.array([1, i, j], dtype=float), [3, 1])
    # mapped=W_[-3]+W_[-2]*i+W_[-1]*j
    src_kps = source_kps - pts_2
    src_kps = np.log(np.sum(src_kps ** 2, axis=1) + 1e-5) * np.sum(src_kps ** 2, axis=1)
    new_i = np.matmul(W[:, 0][-3:], pts_3)[0]
    new_i += np.sum(W[:, 0][:-3] * src_kps)
    new_i = int(new_i+0.5)

    new_j = np.matmul(W[:, 1][-3:], pts_3)[0]
    new_j += np.sum(W[:, 1][:-3] * src_kps)
    new_j = int(new_j+0.5)
    return (new_i,new_j)

def interpolate(W,image,source_kps):
    '''
    default trans into the same size
    :param W: the parameters of the function
    :param image: the input image, numpy array, with np.float32 type without norm
    :return: the trans images
    '''
    #when multi points are mapped into a same point, do sample work

    #image_mean=np.zeros([1,3])
    #new_image=np.matmul(np.ones([image.shape[0],image.shape[1],1],dtype=float),image_mean)
    new_image=np.empty([image.shape[0],image.shape[1],image.shape[2]])
    map_matrix={}

    # height,width=image.shape[0],image.shape[1]
    x=np.empty(shape=[image.shape[0],image.shape[1]])
    y=np.empty(shape=[image.shape[0],image.shape[1]])

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            key=cal_new_i_j(W,i,j,source_kps)
            #x[i,j],y[i,j]=cal_new_i_j(W,i,j,source_kps)
            y[i,j],x[i,j]=key
            if key in map_matrix.keys():
                map_matrix[key].append(image[i][j])
            else:
                map_matrix[key]=[image[i][j]]
    #now we have the mapping matrix, create the new image
    for key in map_matrix:
        average=np.mean(np.array(map_matrix[key]),axis=0)
        #new_image[int((key[0]-min_x)/(float)(max_x-min_x)*(image.shape[0]-2)+0.5)][int((key[1]-min_y)/(float)(max_y-min_y)*(image.shape[1]-2)+0.5)]=average
        if key[0]<0 or key[0]>=224 or key[1]<0 or key[1]>=224:continue
        new_image[key[0]][key[1]]=average
    return new_image
    '''
    return output
    '''

def main():
    source_kps=np.load('/home/solink/kiwi_fung/horse/npy/_04_Aug16_png/horse+head7.npy')
    target_kps=np.load('/home/solink/kiwi_fung/aflw/npy/2/image00010_6234.npy')
    image=Image.open('/home/solink/kiwi_fung/horse/im/_04_Aug16_png/horse+head7.jpg')
    image_np=np.array(image,dtype=float)

    src_kps=[]
    tar_kps=[]
    for i in range(source_kps.shape[0]):
        if source_kps[i][-1]!=-1 and target_kps[i][-1]!=-1:
            src_kps.append(source_kps[i][:2])
            tar_kps.append(target_kps[i][:2])
    src_kps=np.array(src_kps,dtype=float)
    tar_kps=np.array(tar_kps,dtype=float)

    W=cal_z(src_kps,tar_kps)
    new_image=interpolate(W,image_np,src_kps).reshape(image_np.shape)
    new_jpg=Image.fromarray(new_image.astype('uint8'),mode='RGB')


    #new_jpg.show(title='trans_image')
    new_jpg.save('trans2.jpg')
