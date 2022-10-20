import cv2
import numpy as np

if __name__ == '__main__':
    n0 = cv2.imread(f'ut/p2/p2.jpg')
    n1 = cv2.imread(f'ut/p2/test2.jpg')
    n = cv2.add(n1,n0)
    img = cv2.addWeighted(n0, 0.5, n1, 0.5, 0)
    cv2.imwrite(f'ut/p2/pp.jpg', img.astype(np.uint8))


    # p3 = cv2.imread('p3.jpg')
    # # t = cv2.resize(n2, dsize=(p3.shape[:-1][::-1]))
    #
    # img0 = cv2.add(n1,n0)
    # img1 = cv2.add(c3,n2)
    # img2 = cv2.addWeighted(c3, 0.5, n2, 0.5, 0)
    # img4 = cv2.addWeighted(p3, 0.5, t, 0.5, 0)
    # cv2.imwrite(f'ut/t.jpg', img1.astype(np.uint8))
    # cv2.imwrite(f'ut/test4.jpg', img4.astype(np.uint8))