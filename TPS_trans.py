import numpy as np
from PIL import Image

def get_radius(x,y):
    return np.sqrt(x**2+y**2)

def U(item):
    x,y=item
    return np.log(get_radius(x,y)**2+1e-5)*get_radius(x,y)**2

def L_con(source_kps):
    n=source_kps.shape[0]
    K=np.zeros([n,n],dtype=np.float32)
    for i in range(n):
        for j in range(n):
            K[i][j]=U(source_kps[i]-source_kps[j])
    P=np.ones([n,3],dtype=np.float32)
    for i in range(n):
        P[i,1],P[i,2]=source_kps[i]
    P_t=np.transpose(P)
    P_t=np.concatenate([P_t,np.zeros([3,3])],axis=-1)
    L=np.concatenate([np.concatenate([K,P],axis=-1),P_t],axis=0)
    return L

def cal_z(source_kps,target_kps):
    '''

    :param source_kps: source_kps with form [num_kps,2], the same size with target_kps
    :param target_kps: target_kps with form [num_kps,2]
    :return: get W,a,c the parameters of the interpolate function, we can use the parameters to reconstruct the function
            then
    '''
    L=L_con(source_kps)
    L_inverse=np.linalg.inv(L)
    tar_kps=np.concatenate([target_kps,np.zeros([3,2],dtype=np.float32)])
    W=np.matmul(L_inverse,tar_kps)

    return W

def cal_map(W_,i,j,source_kps):
    mapped=W_[-3]+W_[-2]*i+W_[-1]*j
    for t in range(W_.shape[0]-3):
        mapped+=W_[t]*U(source_kps[t]-np.array([i,j],dtype=np.float32))
    return mapped

def interpolate(W,image,source_kps):
    '''
    default trans into the same size
    :param W: the parameters of the function
    :param image: the input image, numpy array, with np.float32 type without norm
    :return: the trans images
    '''
    #when multi points are mapped into a same point, do sample work

    W1 = np.reshape(W[:, 0], [-1])
    W2 = np.reshape(W[:, 1], [-1])

    #image_mean=np.reshape(np.mean(np.mean(image,axis=0),axis=0),[1,3])
    image_mean=np.zeros([1,3])
    new_image=np.matmul(np.ones([image.shape[0],image.shape[1],1],dtype=np.float32),image_mean)

    #the function is to cal the offset of the given points

    map_matrix={}
    x=[]
    y=[]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            new_i=int(np.asscalar(np.floor(cal_map(W1,i,j,source_kps))))
            new_j=int(np.asscalar(np.floor(cal_map(W2,i,j,source_kps))))


            if (new_i,new_j)in map_matrix.keys():
                map_matrix[(new_i,new_j)].append(image[i][j])
            else:
                map_matrix[(new_i,new_j)]=[image[i][j]]
                x.append(new_i)
                y.append(new_j)

    min_x,min_y,max_x,max_y=min(x),min(y),max(x),max(y)

    #too many information loss because they are mapped out of the boundary
    rescale_map_matrix={}

    for key in map_matrix:
        new_i,new_j=key
        new_i=int((new_i-min_x)/(float)(max_x-min_x)*image.shape[0])
        new_j=int((new_j-min_y)/(float)(max_y-min_y)*image.shape[1])

        if (new_i,new_j) in rescale_map_matrix.keys():
            rescale_map_matrix[(new_i,new_j)]=rescale_map_matrix[(new_i,new_j)]+map_matrix[key]
        else:
            rescale_map_matrix[(new_i,new_j)]=map_matrix[key]

    #now we have the mapping matrix, create the new image
    for key in rescale_map_matrix:
        len_=len(rescale_map_matrix[key])
        total=np.zeros([1,3],dtype=np.float32)
        for i in range(len_):
            total=np.add(total,rescale_map_matrix[key][i])
        average=np.floor(total/len_)
        if key[0]<0 or key[1]<0 or key[0]>=image.shape[0] or key[1]>=image.shape[1]:continue
        else :
            new_image[key[0]][key[1]]=average

    return new_image

def test():
    source_kps=np.load('/home/solink/kiwi_fung/horse/npy/_04_Aug16_png/horse+head7.npy')
    target_kps=np.load('/home/solink/kiwi_fung/aflw/npy/2/image00010_6234.npy')
    image=Image.open('/home/solink/kiwi_fung/horse/im/_04_Aug16_png/horse+head7.jpg')
    image_np=np.array(image,dtype=np.float32)

    src_kps=[]
    tar_kps=[]
    for i in range(source_kps.shape[0]):
        if source_kps[i][-1]!=-1 and target_kps[i][-1]!=-1:
            src_kps.append(source_kps[i][:2])
            tar_kps.append(target_kps[i][:2])
    src_kps=np.array(src_kps,dtype=np.float32)
    tar_kps=np.array(tar_kps,dtype=np.float32)

    W=cal_z(src_kps,tar_kps)
    new_image=interpolate(W,image_np,src_kps)
    new_jpg=Image.fromarray(new_image.astype('uint8'),mode='RGB')
    new_jpg.show(title='trans_image')
    new_jpg.save('trans.jpg')

if __name__=='__main__':
    test()
