import numpy as np
from PIL import Image
from finetune import  build_dataset
def TPS_STN(U,source_kps,target_kps, out_size):
    """
    TPS trans as the idea of STN
    U:image 
    source_kps,target_kps:[num_kps,2]
    out_size=(out_height,out_width)
    
    ----------
    Reference :
      https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py
      https://github.com/iwyoo/TPS_STN-tensorflow
      https://arxiv.org/pdf/1603.03915.pdf
    """

    def _repeat(x, n_repeats):
        rep = np.transpose(
            np.expand_dims(np.ones(shape=np.stack([n_repeats, ])), 1), [1, 0])
        rep = rep.astype(np.int32)
        x = np.matmul(np.reshape(x, (-1, 1)), rep)
        return np.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        # constants
        num_batch = np.shape(im)[0]
        height = np.shape(im)[1]
        width = np.shape(im)[2]
        channels = np.shape(im)[3]


        height_f =float(height)
        width_f =float(width)
        out_height = out_size[0]
        out_width = out_size[1]
        zero = np.zeros([], dtype=int)
        max_y = int(np.shape(im)[1] - 1)
        max_x = int(np.shape(im)[2] - 1)

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0) * (width_f) / 2.0
        y = (y + 1.0) * (height_f) / 2.0

        # do sampling
        x0 = np.floor(x).astype(np.int32)
        x1 = x0 + 1
        y0 = np.floor(y).astype(np.int32)
        y1 = y0 + 1

        x0 = np.clip(x0, zero, max_x)
        x1 = np.clip(x1, zero, max_x)
        y0 = np.clip(y0, zero, max_y)
        y1 = np.clip(y1, zero, max_y)
        dim2 = width
        dim1 = width * height
        base = _repeat(np.arange(num_batch)*dim1 , out_height * out_width)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = np.reshape(im, np.stack([-1, channels]))
        im_flat = im_flat.astype(np.float32)
        Ia = im_flat[idx_a]
        Ib = im_flat[idx_b]
        Ic = im_flat[idx_c]
        Id = im_flat[idx_d]

        # and finally calculate interpolated values
        x0_f = x0.astype(np.float32)
        x1_f = x1.astype(np.float32)
        y0_f = y0.astype(np.float32)
        y1_f = y1.astype(np.float32)
        wa = np.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = np.expand_dims(((x1_f - x) * (y - y0_f)), 1)
        wc = np.expand_dims(((x - x0_f) * (y1_f - y)), 1)
        wd = np.expand_dims(((x - x0_f) * (y - y0_f)), 1)
        output = np.sum([wa * Ia, wb * Ib, wc * Ic, wd * Id],axis=0)
        return output

    def _meshgrid(height, width, fp):
        # coordinate trans to [-1,1]
        x_t = np.matmul(
            np.ones(shape=np.stack([height, 1])),
            np.transpose(np.expand_dims(np.linspace(-1.0, 1.0, width,dtype=np.float32), 1), [1, 0]))
        y_t = np.matmul(
            np.expand_dims(np.linspace(-1.0, 1.0, height,dtype=np.float32), 1),
            np.ones(shape=np.stack([1, width])))

        x_t_flat = np.reshape(x_t, (1, -1))
        y_t_flat = np.reshape(y_t, (1, -1))

        x_t_flat_b = np.expand_dims(x_t_flat, 0)  # [1, 1, h*w]
        y_t_flat_b = np.expand_dims(y_t_flat, 0)  # [1, 1, h*w]

        num_batch = np.shape(fp)[0]

        px = np.expand_dims(fp[:, :, 0], 2)  # [n, nx*ny, 1]
        py = np.expand_dims(fp[:, :, 1], 2)  # [n, nx*ny, 1]

        d = (x_t_flat_b - px)**2 + (y_t_flat_b - py)**2
        r = d* np.log(d+ 1e-6)  # [n, nx*ny, h*w]
        x_t_flat_g = np.tile(x_t_flat_b, np.stack([num_batch, 1, 1]))  # [n, 1, h*w]
        y_t_flat_g = np.tile(y_t_flat_b, np.stack([num_batch, 1, 1]))  # [n, 1, h*w]
        ones = np.ones_like(x_t_flat_g)  # [n, 1, h*w]

        grid = np.concatenate([ones, x_t_flat_g, y_t_flat_g,r], 1)  # [n, nx*ny+3, h*w]
        return grid

    def _transform(T, fp, input_dim, out_size):
        num_batch,height,width,num_channels = input_dim.shape

        out_height = out_size[0]
        out_width = out_size[1]
        grid = _meshgrid(out_height, out_width, fp)  # [2, h*w]

        # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
        T_g = np.matmul(T, grid)  # MARK
        x_s = T_g[:, 0, :]
        y_s = T_g [:, 1, :]
        x_s_flat = np.reshape(x_s, [-1])
        y_s_flat = np.reshape(y_s, [-1])

        input_transformed = _interpolate(
            input_dim, x_s_flat, y_s_flat, out_size)

        output = np.reshape(
            input_transformed,
            np.stack([num_batch, out_height, out_width, num_channels]))
        return output

    def cal_z(source_kps, target_kps):
        '''

        :param source_kps: source_kps with form [num_kps,2], the same size with target_kps
        :param target_kps: target_kps with form [num_kps,2]
        :return: get W,a,c the parameters of the interpolate function, we can use the parameters to reconstruct the function
                then
        '''
        n = source_kps.shape[0]
        P=np.concatenate([np.ones(shape=[n,1],dtype=np.float32),target_kps],axis=-1)
        P_1 = P.reshape([n, 1, 3])
        P_2 = P.reshape([1, n, 3])
        d = np.sqrt(np.sum((P_1 - P_2) ** 2, 2))
        r = d * d * np.log(d*d + 1e-6)

        L = np.zeros([n + 3, n + 3], dtype=np.float32)
        L[:n, 3:] = r
        L[:n, :3] = P
        L[n:, 3:] = P.transpose()
        try:
            L_inverse = np.linalg.inv(L)
        except np.linalg.linalg.LinAlgError:
            return np.array(0,dtype=int)

        tar_kps = np.concatenate([source_kps, np.zeros([3, 2], dtype=np.float32)])
        W = np.matmul(L_inverse, tar_kps)

        return np.expand_dims(W.transpose(),0)
        T=cal_z(source_kps,target_kps)
        
    if  T.dtype==int:
        return T
    source_kps=np.expand_dims(source_kps,axis=0)
    output = _transform(T, source_kps, U, out_size).reshape([out_size[0],out_size[1],3])
    return output

def main():
    source_kps = np.load('/home/solink/kiwi_fung/horse/npy/_04_Aug16_png/horse+head7.npy')
    target_kps = np.load('/home/solink/kiwi_fung/aflw/npy/2/image00010_6234.npy')
    image = Image.open('/home/solink/kiwi_fung/horse/im/_04_Aug16_png/horse+head7.jpg')
    image_np = np.expand_dims(np.array(image, dtype=np.float32),0)

    src_kps = []
    tar_kps = []
    for i in range(source_kps.shape[0]):
        if source_kps[i][-1] != -1 and target_kps[i][-1] != -1:
            src_kps.append((source_kps[i][:2]-112.0)/112.0)
            tar_kps.append((target_kps[i][:2]-112.0)/112.0)
    src_kps = np.array(src_kps, dtype=np.float32)
    tar_kps = np.array(tar_kps, dtype=np.float32)

    tps=TPS_STN(image_np,src_kps,tar_kps,(224,224))
    new_jpg = Image.fromarray(tps.astype('uint8'), mode='RGB')
    #new_jpg=new_jpg.transpose(Image.FLIP_TOP_BOTTOM)
    # new_jpg.show(title='trans_image')
    new_jpg.save('trans6.jpg')
