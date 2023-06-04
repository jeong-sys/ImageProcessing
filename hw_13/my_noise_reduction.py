import numpy as np
import cv2
import time


def my_get_Gaussian2D_mask(msize, sigma=1):
    y, x = np.mgrid[-(msize // 2):(msize // 2) + 1,
           -(msize // 2):(msize // 2) + 1]

    gaus2D = 1 / (2 * np.pi * sigma**2) * \
             np.exp(-(( x**2 + y**2 )/(2 * sigma**2)))

    gaus2D /= np.sum(gaus2D)
    return gaus2D


def my_normalize(src):
    dst = src.copy()
    dst *= 255
    dst = np.clip(dst, 0, 255)
    return dst.astype(np.uint8)


def my_padding(src, pad_shape):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h+2*p_h, w+2*p_w))
    pad_img[p_h:p_h+h, p_w:p_w+w] = src
    return pad_img


def my_filtering(src, mask):
    (h, w) = src.shape
    (m_h, m_w) = mask.shape
    pad_img = my_padding(src, (m_h//2, m_w//2))
    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            val = np.sum(pad_img[row:row + m_h, col:col + m_w] * mask)
            dst[row, col] = val
    return dst


def add_gaus_noise(src, mean=0, sigma=0.1):
    #src : 0 ~ 255, dst : 0 ~ 1
    dst = src/255
    h, w, c = dst.shape
    noise = np.random.normal(mean, sigma, size=(h, w, c))
    dst += noise
    return my_normalize(dst)


def my_bilateral(src, msize, sigma, sigma_r):
    ####################################################################################################
    # TODO                                                                                             #
    # my_bilateral 완성                                                                                 #
    # mask를 만들 때 4중 for문으로 구현 시 감점(if문 fsize*fsize개를 사용해서 구현해도 감점) 실습영상 설명 참고      #
    ####################################################################################################
    (h, w) = src.shape
    m_s = msize // 2
    img_pad = my_padding(src, pad_shape=(m_s, m_s))
    dst = np.zeros((h, w))

    y, x = np.mgrid[-m_s:m_s + 1, -m_s:m_s + 1]

    for i in range(h):
        print('\r%d / %d ...' %(i,h), end="")
        for j in range(w):
            k = y + i
            l = x + j
            mask = np.exp( -(((i - k)**2) / (2 * sigma**2)) -(((j-l)**2) / (2 * sigma**2)) ) * np.exp( -(((img_pad[i+m_s, j+m_s] - img_pad[k+m_s, l+m_s])**2)/(2*sigma_r**2)) )
            mask = mask/mask.sum()
            dst[i, j] = np.sum(img_pad[i:i + msize, j:j + msize] * mask)
    # return dst
    return my_normalize(dst)


def my_median_filtering(src, msize):
    h, w = src.shape

    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            r_start = np.clip(row - msize // 2, 0, h)
            r_end = np.clip(row + msize // 2, 0, h)

            c_start = np.clip(col - msize // 2, 0, w)
            c_end = np.clip(col + msize // 2, 0, w)
            mask = src[r_start:r_end, c_start:c_end]
            dst[row, col] = np.median(mask)
    ######################################################
    # TODO                                               #
    # median filtering 코드 작성                          #
    ######################################################

    return dst.astype(np.uint8)


if __name__ == '__main__':
    src = cv2.imread('canoe.png')
    np.random.seed(seed=100)

    noise_image = add_gaus_noise(src, mean=0, sigma=0.1)
    src_noise = noise_image / 255

    ######################################################
    # TODO                                               #
    # RGB에서 Bilateral, Gaussian, Median filter 진행     #
    ######################################################
    # RGB
    b_channel, g_channel, r_channel = cv2.split(src_noise)


    b_bilateral = my_bilateral(b_channel, 5, 20, 300)
    g_bilateral = my_bilateral(g_channel, 5, 20, 300)
    r_bilateral = my_bilateral(r_channel, 5, 20, 300)

    b_gaussian = my_filtering(b_channel, my_get_Gaussian2D_mask(5, sigma=20))
    g_gaussian = my_filtering(g_channel, my_get_Gaussian2D_mask(5, sigma=20))
    r_gaussian = my_filtering(r_channel, my_get_Gaussian2D_mask(5, sigma=20))

    b_median = my_median_filtering(b_channel, 5)
    g_median = my_median_filtering(g_channel, 5)
    r_median = my_median_filtering(r_channel, 5)

    rgb_bilateral_dst = cv2.merge([b_bilateral, g_bilateral, r_bilateral])
    rgb_gaussian_dst = cv2.merge([b_gaussian, g_gaussian, r_gaussian])
    rgb_median_dst = cv2.merge([b_median, g_median, r_median])

    ######################################################
    # TODO                                               #
    # YUV에서 Bilateral, Gaussian, Median filter 진행     #
    ######################################################
    # YUV
    src_noise = src_noise.astype(np.uint8)
    yuv = cv2.cvtColor(src_noise, cv2.COLOR_BGR2YUV)
    y_channel, u_channel, v_channel = cv2.split(yuv)

    y_bilateral = my_bilateral(y_channel, 5, 20, 300)
    u_bilateral = my_bilateral(u_channel, 5, 20, 300)
    v_bilateral = my_bilateral(v_channel, 5, 20, 300)

    y_gaussian = my_filtering(y_channel, my_get_Gaussian2D_mask(5, sigma=20))
    y_gaussian = np.clip(y_gaussian, 0, 255).astype(np.uint8)
    u_gaussian = my_filtering(u_channel, my_get_Gaussian2D_mask(5, sigma=20))
    u_gaussian = np.clip(u_gaussian, 0, 255).astype(np.uint8)
    v_gaussian = my_filtering(v_channel, my_get_Gaussian2D_mask(5, sigma=20))
    v_gaussian = np.clip(v_gaussian, 0, 255).astype(np.uint8)

    y_median = my_median_filtering(y_channel, 5)
    u_median = my_median_filtering(u_channel, 5)
    v_median = my_median_filtering(v_channel, 5)

    yuv_bilateral_dst = cv2.merge([y_bilateral, u_bilateral, v_bilateral])
    yuv_gaussian_dst = cv2.merge([y_gaussian, u_gaussian, v_gaussian])
    yuv_median_dst = cv2.merge([y_median, u_median, v_median])

    cv2.imshow('original.png', src)
    cv2.imshow('RGB bilateral', rgb_bilateral_dst)
    cv2.imshow('RGB gaussian', rgb_gaussian_dst)
    cv2.imshow('RGB median', rgb_median_dst)

    cv2.imshow('YUV bilateral', cv2.cvtColor(yuv_bilateral_dst, cv2.COLOR_YUV2BGR))
    cv2.imshow('YUV gaussian', cv2.cvtColor(yuv_gaussian_dst, cv2.COLOR_YUV2BGR))
    cv2.imshow('YUV median', cv2.cvtColor(yuv_median_dst, cv2.COLOR_YUV2BGR))

    cv2.waitKey()
    cv2.destroyAllWindows()

