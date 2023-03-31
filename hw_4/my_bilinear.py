import cv2
import numpy as np

"""
해당 부분에 여러분 정보 입력해주세요.
한밭대학교 컴퓨터공학과  20211939 허유정
"""

def my_bilinear(src, scale):
    #########################
    # TODO                  #
    # my_bilinear 완성      #
    #########################
    (h, w) = src.shape
    h_dst = int(h * scale + 0.5)
    w_dst = int(w * scale + 0.5)

    dst = np.zeros((h_dst, w_dst))

    # bilinear interpolation 적용
    for row in range(h_dst):
        for col in range(w_dst):
            y = row / scale
            x = col / scale

            m = ???
            n = ???

            t = ???
            s = ???

            """
            픽셀 위치가 이미지를 넘어서는 경우를 막기위해서 조건문을 사용
            각 조건문을 생각하여 코드를 완성하기
            Hint: 4가지에 대한 경우를 생각해야함
            1. m+1, n+1 모두 이미지를 넘어서는 경우
            2. m+1이 이미지를 넘어서는 경우 
            3. n+1이 이미지를 넘어서는 경우
            4. 그외
            """
            value = ???

            dst[row, col] = value

    return dst

if __name__ == '__main__':
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)

    scale = 3
    #이미지 크기 ??x??로 변경
    my_dst_mini = my_bilinear(src, 1/scale)
    my_dst_mini = my_dst_mini.astype(np.uint8)

    #이미지 크기 512x512로 변경(Lena.png 이미지의 shape는 (512, 512))
    my_dst = my_bilinear(my_dst_mini, scale)
    my_dst = my_dst.astype(np.uint8)

    # 출력 윈도우에 학번과 이름을 써주시기 바립니다.
    cv2.imshow('[20211939 허유정]original', src)
    cv2.imshow('[20211939 허유정]my bilinear mini', my_dst_mini)
    cv2.imshow('[20211939 허유정]my bilinear', my_dst)

    cv2.waitKey()
    cv2.destroyAllWindows()