import numpy as np

if __name__ == '__main__':
    arr = np.arange(9).reshape((3, 3, 1))
    ar2 = np.arange(9).reshape((3, 3, 1)) * 2
    print(np.transpose(arr).shape)