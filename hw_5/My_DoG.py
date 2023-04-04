import cv2
import numpy as np

# library add
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_filtering_true import my_filtering


def get_DoG_filter(fsize, sigma=1):
    ###################################################
    # TODO                                            #
    # DoG mask 완성                                    #
    ###################################################
    y, x = np.mgrid[-(fsize//2):(fsize//2)+1, -(fsize//2):(fsize//2)+1]

    DoG_x = ???
    DoG_y = ???

    return DoG_x, DoG_y

def main():
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)
    DoG_x, DoG_y = get_DoG_filter(fsize=3, sigma=1)

    ###################################################
    # TODO                                            #
    # DoG mask sigma값 조절해서 mask 만들기              #
    ###################################################
    # DoG_x, DoG_y filter 확인
    x, y = get_DoG_filter(fsize=256, sigma=10)
    x = ???
    y = ???

    dst_x = my_filtering(src, DoG_x)
    dst_y = my_filtering(src, DoG_y)

    ###################################################
    # TODO                                            #
    # dst_x, dst_y 를 사용하여 magnitude 계산            #
    ###################################################
    dst = ???

    """
    한글이나 영어로 작성하기
    ID, name(학번 이름): 2022012340, Hon Gil Dong    
    department(학과): computer science
    """

    cv2.imshow('DoG_x filter - 2022012340 Hong Gil Dong', x)
    cv2.imshow('DoG_y filter - 2022012340 Hong Gil Dong', y)
    cv2.imshow('dst_x - 2022012340 Hong Gil Dong', dst_x/255)
    cv2.imshow('dst_y - 2022012340 Hong Gil Dong', dst_y/255)
    cv2.imshow('dst - 2022012340 Hong Gil Dong', dst/255)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

