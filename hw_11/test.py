import numpy as np
import time
import cv2


def C(w, n=8):
    if w == 0:
        return (1 / n) ** 0.5
    else:
        return (2 / n) ** 0.5


# inverse DCT
def C_matrix(w, n=8):
    val = np.round(w / (w + 1E-6))
    val = (1 - val) * (1 / n) ** 0.5 + (val) * (2 / n) ** 0.5
    """y, x = w.shape
    val = np.zeros((y, x))
    for row in range(y):
        for col in range(x):
            if w[row,col] == 0:
                val[row,col] = (1 / n) ** 0.5
            else:
                val[row,col] =  (2 / n) ** 0.5"""

    return val


def Quantization_Luminance(scale_factor):
    luminance = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]])
    return luminance * scale_factor


def img2block(src, n=8):
    (h, w) = src.shape

    if h % n != 0:
        h_pad = n - h % n
    else:
        h_pad = 0

    if w % n != 0:
        w_pad = n - w % n
    else:
        w_pad = 0

    dst = np.zeros((h + h_pad, w + w_pad))
    dst[:h, :w] = src

    blocks = []
    for row in range(h // n):
        for col in range(w // n):
            block = dst[n * (row):n * (row + 1), n * (col):n * (col + 1)]
            blocks.append(block)
    ######################################
    # TODO                               #
    # img2block 완성                      #
    # img를 block으로 변환하기              #
    ######################################

    return np.array(blocks)


def DCT(block, n=8):
    ######################################
    # TODO                               #
    # DCT 완성                            #
    ######################################
    y, x = block.shape

    dst = np.zeros(block.shape)
    v, u = dst.shape

    for v_ in range(v):
        for u_ in range(u):
            tmp = 0
            for y_ in range(y):
                for x_ in range(x):
                    tmp += block[y_, x_] * np.cos(((2 * x_ + 1) * u_ * np.pi) / (2 * n)) * \
                           np.cos(((2 * y_ + 1) * v_ * np.pi) / (2 * n))

            dst[v_, u_] = C(u_, n=n) * C(v_, n=n) * tmp

    return np.round(dst)


def my_zigzag_scanning_all(zigzag, block, block_recover):
    row = 0
    col = 0
    val = 0

    right = False
    left = False

    recover = False

    if len(block) == 8:
        recover = False
    else:
        recover = True

    while (True):
        if recover:
            block_recover[row][col] = block[val]
        else:
            zigzag.append(block[row][col])

        val += 1

        # #첫번째 "->"방향으로 이동
        if row == 0 and col == 0:
            col += 1
            # print("한번")

        elif (row == 0 or col == 0):
            right = False
            # 왼쪽아래로 이동
            if (row == 0):
                if (col % 2 == 0):
                    col += 1
                else:
                    col -= 1
                    row += 1
                    left = True

            elif (col == 0):
                left = False
                # 오른쪽 위로 이동
                if (row % 2 == 0):
                    row -= 1
                    col += 1
                    right = True
                else:
                    if row == 7:
                        col += 1
                        right = True
                    row += 1


        elif (row != 0 and col != 0):
            if right:
                right = True
                if row == 0:
                    right = False
                if row >= 1 and col == 7:
                    right = False
                    row += 1
                    left = True
                row -= 1
                col += 1

            if left:
                left = True

                if col == 0:
                    left = False
                else:
                    if col >= 1 and row == 7:
                        left = False
                        col += 1
                        right = True
                    else:
                        col -= 1
                        row += 1

        if row >= 7:
            row = 7
        if col >= 7:
            col = 7

        if val == 64:
            break

    if recover:
        return block_recover
    else:
        return zigzag


def my_zigzag_scanning(block, mode='encoding', block_size=8):
    # print(f'firstblock = ',block)
    ######################################
    # TODO                               #
    # my_zigzag_scanning 완성             #
    ######################################
    row = 0
    col = 0
    idx = 0
    half = False
    zigzag = []
    block_recover = np.zeros((block_size, block_size))
    encoding = True
    zero_count = 0

    if mode == 'decoding':
        encoding = False

    while True:
        if encoding:

            zigzag = my_zigzag_scanning_all(zigzag, block, block_recover)

            for idx in zigzag[::-1]:
                if abs(int(idx)) == 0:
                    zero_count += 1
                else:
                    break
        else:
            if idx >= len(block) or block[idx] == 'EOB':  # decoding 종료조건
                break

            for bc in block:
                if bc != 'EOB':
                    zero_count += 1
            block = block[:len(block) - 1]

            for banboc in range((block_size ** 2) - zero_count):
                block.append(0)

            block_recover = my_zigzag_scanning_all(zigzag, block, block_recover)

        # 종료 조건
        if row == block_size - 1 and col == block_size - 1:
            break

        if half:
            pass
        else:  # half == False
            break

    length = len(zigzag)

    if encoding:
        # 뒤쪽 0 제거
        zigzag = zigzag[:length - zero_count]
        zigzag.append("EOB")
        return zigzag
    else:  # decoding
        return block_recover


def DCT_inv(block, n=8):
    ###################################################
    # TODO                                            #
    # DCT_inv 완성                                     #
    # DCT_inv 는 DCT와 다름.                            #
    ###################################################
    y, x = block.shape

    dst = np.zeros((y, x))
    v, u = dst.shape

    for y_ in range(y):
        for x_ in range(x):
            tmp = 0
            for v_ in range(v):
                for u_ in range(u):
                    tmp += block[v_, u_] * C_matrix(u_, n=n) * C_matrix(v_, n=n) \
                           * np.cos(((2 * x_ + 1) * u_ * np.pi) / (2 * n)) \
                           * np.cos(((2 * y_ + 1) * v_ * np.pi) / (2 * n))

            dst[y_, x_] = tmp

    return np.round(dst)


def block2img(blocks, src_shape, n=8):
    ###################################################
    # TODO                                            #
    # block2img 완성                                   #
    # 복구한 block들을 image로 만들기                     #
    ###################################################
    (h, w) = src_shape

    if h % n != 0:
        h_pad = n - h % n
    else:
        h_pad = 0

    if w % n != 0:
        w_pad = n - w % n
    else:
        w_pad = 0

    dst = np.zeros((h + h_pad, w + w_pad), dtype=np.uint8)

    idx = 0

    for row in range(h // n):
        for col in range(w // n):
            dst[n * row:(row + 1) * n, n * col:(col + 1) * n] = blocks[idx]
            idx += 1

    dst = dst[:h, :w]
    return dst


def Encoding(src, n=8, scale_factor=1):
    #################################################################################################
    # TODO                                                                                          #
    # Encoding 완성                                                                                  #
    # Encoding 함수를 참고용으로 첨부하긴 했는데 수정해서 사용하실 분은 수정하셔도 전혀 상관 없습니다.              #
    #################################################################################################
    print('<start Encoding>')
    # img -> blocks
    blocks = img2block(src, n=n)
    print("block = \n", src[150:158, 89:97])

    # subtract 128
    blocks -= 128
    b = np.double(src[150:158, 89:97]) - 128
    print("b = \n", b)

    # DCT
    blocks_dct = []
    for block in blocks:
        blocks_dct.append(DCT(block, n=n))
    blocks_dct = np.array(blocks_dct)
    # print DCT results
    bd = DCT(b, n=8)
    print("bd = \n", bd)

    # Quantization + thresholding
    Q = Quantization_Luminance(scale_factor)
    QnT = np.round(blocks_dct / Q)
    # print Quantization results
    bq = bd / Q
    print("bq = \n", bq)

    # zigzag scanning
    zz = []
    for i in range(len(QnT)):
        zz.append(my_zigzag_scanning(QnT[i]))

    return zz, src.shape, bq


def Decoding(zigzag, src_shape, bq, n=8, scale_factor=1):
    #################################################################################################
    # TODO                                                                                          #
    # Decoding 완성                                                                                  #
    # Decoding 함수를 참고용으로 첨부하긴 했는데 수정해서 사용하실 분은 수정하셔도 전혀 상관 없습니다.              #
    #################################################################################################
    print('<start Decoding>')
    # print(f'zigzag value\n',zigzag[0:4])
    # zigzag scanning
    blocks = []

    for i in range(len(zigzag)):
        blocks.append(my_zigzag_scanning(zigzag[i], mode='decoding', block_size=n))
    blocks = np.array(blocks)

    # Denormalizing
    Q = Quantization_Luminance(scale_factor=scale_factor)
    blocks = blocks * Q
    # print results Block * Q
    bq2 = bq * Q
    print("bq2 = \n", bq2)

    # inverse DCT
    blocks_idct = []
    for block in blocks:
        blocks_idct.append(DCT_inv(block, n=n))
    blocks_idct = np.array(blocks_idct)
    # print IDCT results
    bd2 = DCT_inv(bq2, n=8)
    print("bd2 = \n", bd2)

    # add 128
    blocks_idct += 128
    # print block value
    b2 = np.round(bd2 + 128)
    print("b2 = \n", b2)

    # block -> img
    dst = block2img(blocks_idct, src_shape=src_shape, n=n)

    return dst, b2


def main():
    scale_factor = 1
    start = time.time()
    # src = cv2.imread('../imgs/Lenna.png', cv2.IMREAD_GRAYSCALE)
    src = cv2.imread('caribou.tif', cv2.IMREAD_GRAYSCALE)

    comp, src_shape, bq = Encoding(src, n=8, scale_factor=scale_factor)
    np.save('comp.npy', comp)
    np.save('src_shape.npy', src_shape)

    # print(comp)
    comp = np.load('comp.npy', allow_pickle=True)
    src_shape = np.load('src_shape.npy')
    recover_img, b2 = Decoding(comp, src_shape, bq, n=8, scale_factor=scale_factor)
    print("scale_factor : ", scale_factor, "differences between original and reconstructed = \n",
          src[150:158, 89:97] - b2)
    # print(recover_img)
    total_time = time.time() - start
    #
    print('time : ', total_time)
    if total_time > 12:
        print('감점 예정입니다.')
    print(recover_img.shape)
    cv2.imshow('recover img', recover_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
