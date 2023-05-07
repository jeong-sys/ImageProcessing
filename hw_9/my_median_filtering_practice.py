import numpy as np
import cv2
import time

def my_filtering(src, filter, pad_type='zero'):
    (h, w) = src.shape
    (f_h, f_w) = filter.shape

    src_pad = my_padding(src, (f_h//2, f_w//2), pad_type)
    dst = np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            val = np.sum(src_pad[row:row+f_h, col:col+f_w]*filter)
            dst[row, col] = val
    return dst

def my_padding(src, pad_shape, pad_type='zero'):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h+2*p_h, w+2*p_w))
    pad_img[p_h:p_h+h, p_w:p_w+w] = src

    if pad_type == 'repetition':
        print('repetition padding')
        #########################################################
        # TODO                                                  #
        # repetition padding 완성                                #
        #########################################################
        #up
        pad_img[:p_h, p_w:p_w+w] = src[0, :]
        #down
        pad_img[p_h+h:, p_w:p_w+w] = src[h-1, :]

        #left
        pad_img[:,:p_w] = pad_img[:, p_w:p_w + 1]
        #right
        pad_img[:, p_w+w:] = pad_img[:, p_w+w-1:p_w+w]

    else:
        print('zero padding')

    return pad_img

def my_median_filtering(src, msize):
    h, w = src.shape

    dst = np.zeros((h, w))
    ######################################################
    # TODO                                               #
    # median filtering 코드 작성                          #
    ######################################################

    for row in range(h):
        for col in range(w):
            r_start = np.clip(row-msize//2, 0, h)
            r_end = np.clip(row+msize//2, 0, h)

            c_start = np.clip(col-msize//2, 0, w)
            c_end = np.clip(col+msize//2, 0, w)
            mask = src[r_start:r_end, c_start:c_end]
            dst[row, col] = np.median(mask)

    return dst.astype(np.uint8)

def add_SnP_noise(src, prob):
    h, w = src.shape

    # np.ranomd.rand = 0~1 사이의 값이 나옴
    noise_prob = np.random.rand(h, w)
    dst = np.zeros((h, w), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            if noise_prob[row, col] < prob:
                # pepper noise
                dst[row, col] = 0
            elif noise_prob[row, col] > 1 - prob:
                # salt noise
                dst[row, col] = 255
            else:
                dst[row, col] = src[row, col]

    return dst

def main():
    np.random.seed(seed=100)
    fname = 'Lena.png'
    src = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    # src = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)
    snp_noise = add_SnP_noise(src, prob=0.05)

    average_start = time.time()
    # nxn average mask
    mask_size = 5
    mask = np.ones((mask_size,mask_size)) / (mask_size**2)

    dst_aver = my_filtering(snp_noise, mask)
    dst_aver = dst_aver.astype(np.uint8)
    print('average filtering time : ', time.time()-average_start)

    median_start = time.time()
    dst_median = my_median_filtering(snp_noise, mask_size)
    print('median filtering time : ', time.time()-median_start)

    cv2.imshow('original', src)
    cv2.imshow('Salt and Pepper noise', snp_noise)
    cv2.imshow('noise removal(average filter)', dst_aver)
    cv2.imshow('noise removal(median filter)', dst_median)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

