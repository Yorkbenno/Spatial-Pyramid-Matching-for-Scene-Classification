import numpy as np

if __name__ == '__main__':
    arr = np.arange(9).reshape((3, 3, 1))
    ar2 = np.arange(9).reshape((3, 3, 1)) * 2
    # print(arr)
    # print(np.max(arr, axis=0))
    # print(np.max(arr, axis=1))
    # print(np.max(arr, axis=2))
    # print(np.max(arr, axis=(0, 1)))
    # print(np.max(arr, axis=(1, 2)))
    # a = np.arange(9).reshape(3, 3)
    # print(np.flip(a))
    # print(np.flipud(np.fliplr(a)))
    print(np.stack((arr, ar2), axis=-1).shape)
    print(np.dstack((arr, ar2)).shape)
