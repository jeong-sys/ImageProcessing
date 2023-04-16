import numpy as np
import cv2


def get_dst_coordinate(src, M):
    h_src, w_src = src.shape[:2]

    """
    실습 참고
    """
    h_src, w_src = src.shape[:2]
    rows=[]
    cols=[]
    row_list = [0, 0, h_src-1, h_src-1]
    col_list = [0, w_src-1, 0, w_src-1]

    for i in range(4):
        P = np.array([col_list[i]],[row_list[i]],[1])
        P_dst = np.dot(M, P)
        cols.append(P_dst[0][0])
        rows.append(P_dst[1][0])

    row_max = max(rows)
    row_min = min(rows)

    col_max = max(cols)
    col_min = min(cols)

    row_max = int(np.ceil(row_max))
    row_min = int(np.floor(row_min))

    col_max = int(np.ceil(col_max))
    col_min = int(np.floor(col_min))
    
    return row_max, row_min, col_max, col_min


def transformation_backward(src, M):
    #######################################
    # TODO                                #
    # Backward 완성하기                     #
    # 비교를 위해서 행렬 M 수정 필요           #
    #######################################
    h, w, c = src.shape
    row_max, row_min, col_max, col_min = get_dst_coordinate(src, M)
    dst = np.zeros((row_max - row_min + 1, col_max - col_min + 1, 3))

    h_, w_ = dst.shape[:2]
    M_inv = np.linalg.inv(M)

    print("backward calc")
    for row_ in range(h_):
        for col_ in range(w_):
            P_dst = M_inv.dot()

            xy = M_inv.dot(P_dst)
            x = xy[0, 0]
            y = xy[1, 0]

            """
            pixel의 값을 가져오기 위해서 bilinear 연산을 통해서 값을 가져옴
            bilinear 연산 구현하기
            """
            if x < 0 or y < 0 or x >= w or y >= h:
                continue

            x1 = int(np.floor(x))
            y1 = int(np.floor(y))

            x2 = x1 + 1
            y2 = y1 + 1

            dx = x - x1
            dy = y - y1

            dst[row_, col_, :] = (1 - dy) * ((1 - dx) * src[y1, x1, :] + dx * src[y1, x2, :]) \
                                 + dy * ((1 - dx) * src[y2, x1, :] + dx * src[y2, x2, :])


    dst = ((dst - np.min(dst)) / np.max(dst - np.min(dst)) * 255 + 0.5)  # normalization
    return dst.astype(np.uint8)


def transformation_forward(src, M):
    #######################################
    # TODO                                #
    # Forward 완성하기                      #
    # 비교를 위해서 행렬 M 수정 필요           #
    #######################################
    h, w, c = src.shape
    row_max, row_min, col_max, col_min = get_dst_coordinate(src, M)
    dst = np.zeros((row_max - row_min + 1, col_max - col_min + 1, 3))

    """
    실습 참고
    """
    h_, w_ = dst.shape[:2]
    count = dst.copy()

    print("forward calc")
    for row in range(h):
        for col in range(w):
            xy_prime = np.dot(M, np.array([[col, row, 1]]).T)
            x_ = xy_prime[0, 0] - col_min
            y_ = xy_prime[1, 0] - row_min

            if x_ <0 or y_ < 0 or x_ >= w_ or y_ >= h_:
                continue

            dst[int(y_), int(x_), :] += src[row, col, :]
            count[int(y_), int(x_), :] += 1

    dst = (dst / count)
    return dst.astype(np.uint8)


def main():
    src = cv2.imread("Lena.png")
    src = cv2.resize(src, None, fx=0.3, fy=0.3)

    ##################################################################
    # TODO                                                           #
    # 행렬 M에 대해서 Forward와 Backward 결과 비교하기                    #
    # 비교를 위해서 행렬 M 수정 필요                                      #
    ##################################################################
    theta = -10

    translation = [[1, 0, 100],
                   [0, 1, 100],
                   [0, 0, 1]]
    rotation = [[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]]
    scaling = [[2, 0, 0],
               [0, 2, 0],
               [0, 0, 1]]

    M = np.dot(np.dot(translation, rotation), scaling)
    ##################################################################

    forward = transformation_forward(src, M)
    backward = transformation_backward(src, M)

    cv2.imshow("backward", forward)
    cv2.imshow("backward", backward)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()