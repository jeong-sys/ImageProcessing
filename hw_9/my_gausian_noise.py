import numpy as np
import cv2

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


def my_normalize(src):
    dst = src.copy()
    dst *= 255
    dst = np.clip(dst, 0, 255)
    return dst.astype(np.uint8)

def add_gaus_noise(src, mean=0, sigma=0.1):
    ######################################################
    # TODO                                               #
    # gaussian noise image들을 더하는 코드 작성             #
    ######################################################
    dst = src/255
    h, w = dst.shape
    noise = np.random.normal(mean, sigma, size=(h, w))
    dst+=noise

    return my_normalize(dst)



"""def main():
    np.random.seed(seed=100)
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)
    dst_noise = add_gaus_noise(src, mean=0, sigma=0.3)

    cv2.imshow('original', src)
    cv2.imshow('add gaus noise', dst_noise)
    cv2.waitKey()
    cv2.destroyAllWindows()"""


"""def main():
    np.random.seed(seed=100)
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)
    dst_noise = add_gaus_noise(src, mean=0, sigma=0.1)

    h, w = src.shape
    num = 100

    imgs = np.zeros((num, h, w))
    for i in range(num):
        imgs[i] =  add_gaus_noise(src, mean=0, sigma=0.1)

    dst = np.mean(imgs, axis=0).astype(np.uint8)

    cv2.imshow('original', src)
    cv2.imshow('add gaus noise', dst_noise)
    cv2.imshow('noise removal', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()"""

def main():
    np.random.seed(seed=100)
    fname = 'Lena.png'
    src = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    # src = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)
    dst_noise = add_gaus_noise(src, mean=0, sigma=0.1)

    # nxn average mask
    mask_size = 5
    mask = np.ones((mask_size,mask_size)) / (mask_size**2)

    dst_avg = my_filtering(dst_noise, mask)
    dst_avg = dst_avg.astype(np.uint8)

    # gaussian mean
    h, w = src.shape
    num = 100

    imgs = np.zeros((num, h, w))
    for i in range(num):
        imgs[i] = add_gaus_noise(src, mean=0, sigma=0.1)
    dst_gaus = np.mean(imgs, axis=0).astype(np.uint8)



    cv2.imshow('original', src)
    cv2.imshow('add gaus noise', dst_noise)
    cv2.imshow('average filtering', dst_avg)
    cv2.imshow('gaussian noise removal', dst_gaus)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
