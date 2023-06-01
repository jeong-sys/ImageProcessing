import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_threshold(img, th=120):
    ######################################################
    # TODO                                               #
    # 실습시간에 배포된 코드 사용                             #
    ######################################################
    dst = np.zeros(img.shape, img.dtype)

    dst[img >= th] = 1
    dst[img < th] = 0

    return dst

def my_otsu_threshold(img):
    hist, bins = np.histogram(img.ravel(),256,[0,256])
    p = hist / np.sum(hist) + 1e-7
    ######################################################
    # TODO                                               #
    # Otsu 방법을 통해 threshold 구한 후 이진화 수행          #
    # cv2의 threshold 와 같은 값이 나와야 함                 #
    ######################################################
    q1 = []
    q2 = []
    m1 = []
    m2 = []
    sigma_1 = []
    sigma_2 = []
    sigma_3 = []

    for k in range(len(hist)):
        p1_sum = 0.0
        p2_sum = 0.0

        for i in range(k + 1):
            p1_sum += p[i]
        q1.append(p1_sum)

        for j in range(k + 1, len(hist)):
            p2_sum += p[j]
        q2.append(p2_sum)

        try:
            for i in range(k + 1):
                p1_sum += i * p[i]
            p1_sum = p1_sum / q1[k]
            m1.append(p1_sum)

            for j in range(k + 1, len(hist)):
                p2_sum += j * p[j]
            p2_sum = p2_sum / q2[k]
            m2.append(p2_sum)

            for i in range(k + 1):
                p1_sum += ((i ** 2) * p[i])
            p1_sum = (p1_sum / q1[k]) - (m1[k] ** 2)
            sigma_1.append(p1_sum)

            for j in range(k + 1, len(hist)):
                p2_sum += ((j ** 2) * p[j])
            p2_sum = (p2_sum / q2[k]) - (m2[k] ** 2)
            sigma_2.append(p2_sum)

        except ZeroDivisionError:
            pass

    for k in range(len(hist) - 1):
        sigma_3.append((q1[k] * q2[k]) * ((m1[k] - m2[k]) ** 2))

    th = np.argmax(sigma_3)
    dst = apply_threshold(img / 255, th / 255)

    return th, dst


if __name__ == '__main__':
    img = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)

    th_cv2, dst_cv2 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    th_my, dst_my = my_otsu_threshold(img)

    print('Threshold from cv2: {}'.format(th_cv2))
    print('Threshold from my: {}'.format(th_my))

    cv2.imshow('original image', img)
    cv2.imshow('cv2 threshold', dst_cv2)
    cv2.imshow('my threshold', dst_my)

    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


