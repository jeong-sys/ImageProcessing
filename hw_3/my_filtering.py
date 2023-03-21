import cv2
import numpy as np

def my_get_Gaussian2D_mask(msize, sigma=1):
    #########################################
    # ToDo
    # 2D gaussian filter 만들기
    #########################################
    y, x = np.mgrid[???:???, ???:???]
    '''
    y, x = np.mgrid[-1:2, -1:2]
    y = [[-1,-1,-1],
         [ 0, 0, 0],
         [ 1, 1, 1]]
    x = [[-1, 0, 1],
         [-1, 0, 1],
         [-1, 0, 1]]
    '''
    # 파이 => np.pi 를 쓰시면 됩니다.
    # 2차 gaussian mask 생성
    gaus2D = ???
    # mask의 총 합 = 1
    gaus2D /= ???

    return gaus2D

def my_get_Gaussian1D_mask(msize, sigma=1):
    #########################################
    # ToDo
    # 1D gaussian filter 만들기
    #########################################
    x = np.full((1, msize), [range(-(msize // 2), (msize // 2) + 1)])
    '''
    x = np.full((1, 3), [-1, 0, 1])
    x = [[ -1, 0, 1]]

    x = np.array([[-1, 0, 1]])
    x = [[ -1, 0, 1]]
    '''

    # 파이 => np.pi 를 쓰시면 됩니다.
    gaus1D = ???

    # mask의 총 합 = 1
    gaus1D /= ???
    return gaus1D

def my_mask(ftype, fshape, sigma=1):
    if ftype == 'average':
        print('average filtering')
        ###################################################
        # TODO                                            #
        # mask 완성                                       #
        ###################################################
        mask = ???
        mask = ???

        #mask 확인
        print(mask)

    elif ftype == 'sharpening':
        print('sharpening filtering')
        ##################################################
        # TODO                                           #
        # mask 완성                                      #
        ##################################################

        base_mask = np.zeros(fshape)
        base_mask[fshape[0]//2, fshape[1]//2] = 2
        aver_mask = ???
        aver_mask = ???
        mask = ???

        #mask 확인
        print(mask)

    elif ftype == 'gaussian2D':
        print('gaussian filtering')
        ##################################################
        # TODO                                           #
        # mask 완성                                      #
        ##################################################
        mask = my_get_Gaussian2D_mask(fshape, sigma=sigma)
        #mask 확인
        print(mask)

    elif ftype == 'gaussian1D':
        print('gaussian filtering')
        ##################################################
        # TODO                                           #
        # mask 완성                                      #
        ##################################################
        mask = my_get_Gaussian1D_mask(fshape, sigma=sigma)
        #mask 확인
        print(mask)

    return mask

def my_zero_padding(src, pad_shape):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h+2*p_h, w+2*p_w))
    pad_img[p_h:p_h+h, p_w:p_w+w] = src
    return pad_img

def my_filtering(src, mask):
    #########################################################
    # TODO                                                  #
    # dst 완성                                              #
    # dst : filtering 결과 image                            #
    #########################################################
    h, w = src.shape
    m_h, m_w = mask.shape
    pad_img = my_zero_padding(src, (m_h//2, m_w//2))
    dst = ???
    
    """
    반복문을 이용하여 filtering을 완성하기
    """
    for row in range(h):
        for col in range(w):
            val = ???
            val = np.clip(val, 0, 255) #범위를 0~255로 조정
            ??? = val

    dst = (dst+0.5).astype(np.uint8) #uint8의 형태로 조정

    return dst

if __name__ == '__main__':
    src = cv2.imread('../baby.jpg', cv2.IMREAD_GRAYSCALE)

    # 3x3 filter
    average_mask = my_mask('average', (3, 3))
    sharpening_mask = my_mask('sharpening', (3, 3))

    #원하는 크기로 설정
    #dst_average = my_filtering(src, 'average', (5,5))
    #dst_sharpening = my_filtering(src, 'sharpening', (5,5))

    # 11x13 filter
    #dst_average = my_filtering(src, 'average', (5,3), 'repetition')
    #dst_sharpening = my_filtering(src, 'sharpening', (5,3), 'repetition')
    #dst_average = my_filtering(src, 'average', (11,13))
    #dst_sharpening = my_filtering(src, 'sharpening', (11,13))


    dst_average = my_filtering(src, average_mask)
    dst_sharpening = my_filtering(src, sharpening_mask)

    # Gaussian filter
    gaussian2d_mask = my_mask('gaussian2D', (3, 3), sigma=1)
    gaussian1d_mask = my_mask('gaussian1D', 3, sigma=1)

    dst_gaussian2d = my_filtering(src, gaussian2d_mask)

    dst_gaussian1d = my_filtering(src, gaussian1d_mask.T)
    dst_gaussian1d = my_filtering(dst_gaussian1d, gaussian1d_mask)


    cv2.imshow('original', src)
    cv2.imshow('average filter', dst_average)
    cv2.imshow('sharpening filter', dst_sharpening)
    cv2.imshow('gaussian2D filter', dst_gaussian2d)
    cv2.imshow('gaussian1D filter', dst_gaussian1d)
    cv2.waitKey()
    cv2.destroyAllWindows()
