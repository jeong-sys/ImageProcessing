import cv2
import numpy as np

"""
해당 부분에 여러분 정보 입력해주세요.
한밭대학교 컴퓨터공학과  20211939 허유정(Heo You Jeong)
"""

def my_nearest_neighbor(src, scale):
    (h, w) = src.shape
    h_dst = int(h * scale + 0.5)
    w_dst = int(w * scale + 0.5)

    dst = np.zeros((h_dst, w_dst), np.uint8)
    for row in range(h_dst):
        for col in range(w_dst):
            r = min(int(row / scale + 0.5), h - 1)
            c = min(int(col / scale + 0.5), w - 1)
            dst[row, col] = src[r, c]

    return dst

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

            m = int(y)
            n = int(x)

            t = abs(y-m)
            s = abs(x-n)

            """
            픽셀 위치가 이미지를 넘어서는 경우를 막기위해서 조건문을 사용
            각 조건문을 생각하여 코드를 완성하기
            Hint: 4가지에 대한 경우를 생각해야함
            1. m+1, n+1 모두 이미지를 넘어서는 경우
            2. m+1이 이미지를 넘어서는 경우 
            3. n+1이 이미지를 넘어서는 경우
            4. 그외
            """

            if m + 1 >= h and n + 1 >= w:
                value = (1 - s) * (1 - t) * src[m][n] + s * (1 - t) * src[m][n] + (1 - s) * t * src[m][n] + s * t * \
                        src[m][n]
            elif m + 1 >= h:
                value = (1 - s) * (1 - t) * src[m][n] + s * (1 - t) * src[m][n + 1] + (1 - s) * t * src[m][n] + s * t * \
                        src[m][n + 1]
            elif n + 1 >= w:
                value = (1 - s) * (1 - t) * src[m][n] + s * (1 - t) * src[m][n] + (1 - s) * t * src[m + 1][n] + s * t * \
                        src[m + 1][n]


            if m + 1 >= h and n + 1 >= w:
                value = src[h - 1, w - 1]
            elif m + 1 >= h and n + 1 < w:
                value = ((1 - s) * src[h - 1, n]) + (s * src[h - 1, n + 1])
            elif m + 1 < h and n + 1 >= w:
                value = ((1 - t) * src[m, w - 1]) + (t * src[m + 1, w - 1])
            else:
                value = (1-s) * (1-t) * src[m][n] + s * (1-t) * src[m][n+1] + (1-s) * t * src[m+1][n] + s * t * src[m+1][n+1]

            value = int(value)
            dst[row, col] = value

    return dst

if __name__ == '__main__':
    fname = 'Lena.png'
    src = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

    scale = 3
    #이미지 크기 ??x??로 변경
    near_my_dst_mini = my_nearest_neighbor(src, 1/scale)
    near_my_dst_mini = near_my_dst_mini.astype(np.uint8)
    my_dst_mini = my_bilinear(src, 1/scale)
    my_dst_mini = my_dst_mini.astype(np.uint8)

    #이미지 크기 512x512로 변경(Lena.png 이미지의 shape는 (512, 512))
    near_my_dst = my_nearest_neighbor(near_my_dst_mini, scale)
    near_my_dst = near_my_dst_mini.astype(np.uint8)
    my_dst = my_bilinear(my_dst_mini, scale)
    my_dst = my_dst.astype(np.uint8)

    # 출력 윈도우에 학번과 이름을 써주시기 바립니다.
    cv2.imshow('[20211939 Heo you jeong]original', src)
    cv2.imshow('[20211939 Heo you jeong]my nearest mini', near_my_dst_mini)
    cv2.imshow('[20211939 Heo you jeong]my nearest', near_my_dst)
    cv2.imshow('[20211939 Heo you jeong]my bilinear mini', my_dst_mini)
    cv2.imshow('[20211939 Heo you jeong]my bilinear', my_dst)

    cv2.waitKey()
    cv2.destroyAllWindows()