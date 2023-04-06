import cv2
import numpy as np

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

    print('<mask>')
    print(mask)

    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            val = np.sum(pad_img[row:row + m_h, col:col + m_w] * mask)
            dst[row, col] = val
    return dst