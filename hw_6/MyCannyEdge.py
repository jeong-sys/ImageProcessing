import cv2
import numpy as np

def my_padding(src, filter):
    (h, w) = src.shape
    (h_pad, w_pad) = filter.shape
    h_pad = h_pad // 2
    w_pad = w_pad // 2
    padding_img = np.zeros((h+h_pad*2, w+w_pad*2))
    padding_img[h_pad:h+h_pad, w_pad:w+w_pad] = src
    return padding_img

# filter와 image를 입력받아 filtering수행
def my_filtering(src, filter):
    (h, w) = src.shape
    (m_h, m_w) = filter.shape
    pad_img =my_padding(src, filter)
    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row + m_h, col:col + m_w] * filter)
    return dst

def convert_uint8(img):
    #이미지 출력을 위해서 타입을 변경 수행
    return ((img - np.min(img)) / np.max(img - np.min(img)) * 255).astype(np.uint8)

def get_DoG_filter(fsize, sigma=1):
    y, x = np.mgrid[-(fsize//2):(fsize//2)+1, -(fsize//2):(fsize//2)+1]

    DoG_x = -(x / sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    DoG_y = -(y / sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    return DoG_x, DoG_y

# Ix와 Iy의 magnitude를 구함
def calcMagnitude(Ix, Iy):
    # Ix와 Iy의 magnitude를 계산
    magnitude = np.sqrt(Ix ** 2 + Iy ** 2)
    return magnitude

# Ix와 Iy의 angle을 구함
def calcAngle(Ix, Iy):
    angle = np.rad2deg(np.arctan(Iy / (Ix+1e-6)))
    return angle

# non-maximum supression 수행
def non_maximum_supression(magnitude, angle):
    ####################################################################################
    # TODO                                                                             #
    # non_maximum_supression 완성                                                       #
    # largest_magnitude     : non_maximum_supression 결과(가장 강한 edge만 남김)           #
    ####################################################################################
    (h, w) = magnitude.shape
    # angle의 범위 : -90 ~ 90
    largest_magnitude = np.zeros((h, w))
    for row in range(1, h - 1):
        for col in range(1, w - 1):
            degree = angle[row, col]

            # gradient의 degree는 edge와 수직방향이다.
            if 0 <= degree and degree < 45:
                rate = np.tan(np.deg2rad(degree))
                left_magnitude = ???
                right_magnitude = ???
                if magnitude[row, col] == max(left_magnitude, magnitude[row, col], right_magnitude):
                    largest_magnitude[row, col] = magnitude[row, col]

            elif 45 <= degree and degree <= 90:
                rate = ???
                up_magnitude = ???
                down_magnitude = ???
                if magnitude[row, col] == max(up_magnitude, magnitude[row, col], down_magnitude):
                    largest_magnitude[row, col] = magnitude[row, col]

            elif -45 <= degree and degree < 0:
                rate = ???
                left_magnitude = ???
                right_magnitude = ???
                if magnitude[row, col] == max(left_magnitude, magnitude[row, col], right_magnitude):
                    largest_magnitude[row, col] = magnitude[row, col]

            elif -90 <= degree and degree < -45:
                rate = ???
                up_magnitude = ???
                down_magnitude = ???
                if magnitude[row, col] == max(up_magnitude, magnitude[row, col], down_magnitude):
                    largest_magnitude[row, col] = magnitude[row, col]

            else:
                print(row, col, 'error!  degree :', degree)

    return largest_magnitude

def show_strong_weak_edge(src):
    # strong edge / weak edge 이미지 출력용
    h, w = src.shape[:2]
    dst = src.copy()
    dst -= dst.min()
    dst /= dst.max()
    dst *= 255
    dst = dst.astype(np.uint8)
    strong_edge_img = np.zeros((h, w), dtype=np.uint8)
    weak_edge_img = np.zeros((h, w), dtype=np.uint8)

    high_threshold_value, _ = cv2.threshold(dst, 0, 255, cv2.THRESH_OTSU)
    low_threshold_value = high_threshold_value * 0.4

    for row in range(h):
        for col in range(w):
            if dst[row, col] >= high_threshold_value:
                strong_edge_img[row, col] = 255
            elif dst[row, col] < high_threshold_value and dst[row, col] >= low_threshold_value:
                weak_edge_img[row, col] = 55
    return strong_edge_img, weak_edge_img

# double_thresholding 수행 high threshold value는 내장함수(otsu방식 이용)를 사용하여 구하고 low threshold값은 (high threshold * 0.4)로 구한다
def double_thresholding(src):
    dst = src.copy()

    # dst => 0 ~ 255
    dst -= dst.min()
    dst /= dst.max()
    dst *= 255
    dst = dst.astype(np.uint8)
    (h, w) = dst.shape

    high_threshold_value, _ = cv2.threshold(dst, 0, 255, cv2.THRESH_OTSU)

    print('highthreshold')
    print(high_threshold_value)
    low_threshold_value = high_threshold_value * 0.4

    for row in range(h):
        for col in range(w):
            if dst[row, col] >= high_threshold_value:
                dst[row, col] = 255
            elif dst[row, col] < low_threshold_value:
                dst[row, col] = 0
            else:
                #######################################
                # TODO                                #
                # High 값 보다 작고 Low 값 보다 큰 경우   #
                #######################################

    return dst

def my_canny_edge_detection(src, fsize=3, sigma=1):
    """
    한글이나 영어로 작성하기
    ID, name(학번 이름): 2022012340, Hon Gil Dong
    department(학과): computer science
    """
    # low-pass filter를 이용하여 blur효과
    # high-pass filter를 이용하여 edge 검출
    # gaussian filter -> sobel filter 를 이용해서 2번 filtering
    DoG_x, DoG_y = get_DoG_filter(fsize, sigma)
    Ix = my_filtering(src, DoG_x)
    Iy = my_filtering(src, DoG_y)

    # magnitude와 angle을 구함
    magnitude = calcMagnitude(Ix, Iy)
    cv2.imshow('magnitude - 2022012340 Hong Gil Dong', convert_uint8(magnitude))

    angle = calcAngle(Ix, Iy)

    # non-maximum suppression 수행
    larger_magnitude = non_maximum_supression(magnitude, angle)
    cv2.imshow('NMS - 2022012340 Hong Gil Dong', convert_uint8(larger_magnitude))

    # strong / weak edge 확인용
    strong_edge_img, weak_edge_img = show_strong_weak_edge(larger_magnitude)
    cv2.imshow('strong edge - 2022012340 Hong Gil Dong', strong_edge_img)
    cv2.imshow('weak edge - 2022012340 Hong Gil Dong', weak_edge_img)

    # double thresholding 수행
    dst = double_thresholding(larger_magnitude)
    return dst

def main():
    src = cv2.imread('../img/Lena.png', cv2.IMREAD_GRAYSCALE)

    dst = my_canny_edge_detection(src)

    cv2.imshow('original - 2022012340 Hong Gil Dong', src)
    cv2.imshow('my canny edge detection - 2022012340 Hong Gil Dong', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

