import numpy as np
import time
a = np.array([[1, 3, 2],
     [4, 3, 5],
     [2, 9, 7]])
b = np.array([[1, 4, 7],
    [2, 5, 6],
    [5, 2, 8]])
a_1 = np.arange(12).reshape(3,4)
b_1 = np.arange(12).reshape(4,3)
c = np.array([[1, 3, 2],
     [4, 3, 5],
     [2, 9, 7]])
d = np.array([[1, 4, 7],
    [2, 5, 6],
    [5, 2, 8]])
def add(a, b):
    rows = a.shape[0]
    cols = a.shape[1]
    for i in range(rows):
        for j in range(cols):
            a[i][j] += b[i][j]
    return a

def sub(a, b):
    rows = a.shape[0]
    cols = a.shape[1]
    for i in range(rows):
        for j in range(cols):
            a[i][j] -= b[i][j]
    return a
        
def dot(a, b):
    rows_a = a.shape[0]
    cols_a = a.shape[1]
    cols_b = b.shape[1]
    res = np.zeros((rows_a, cols_b))
    for i in range(rows_a):
        for j in range(cols_b):
            for u in range(cols_a):
                res[i][j] += a[i][u] * b[u][j]
    return res

def div(a, b):
    rows = a.shape[0]
    cols = a.shape[1]
    inv_b = np.linalg.inv(b)
    res = np.zeros((rows, cols))
    res = dot(a, inv_b)
    return res

def transpose(a):
    rows = a.shape[0]
    cols = a.shape[1]
    res = [[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]]
    for i in range(rows):
        for j in range(cols):
            res[i][j] = a[j][i]
    return res



if __name__ == "__main__":

    t1 = time.time()

    print("numpy: ")
    print ('两个数组相加：')
    print (np.add(a,b))
    print("time: ", time.time()-t1)

    t2 = time.time()
    print ('两个数组相减：')
    print (np.subtract(a,b))
    print("time: ", time.time()-t2)

    t3 = time.time()
    print ('两个数组相乘：')
    print (a.dot(b))
    print(a_1.dot(b_1))
    print("time: ", time.time()-t3)

    t4 = time.time()
    print ('两个数组相除：')
    inv_b = np.linalg.inv(b)
    print (a.dot(inv_b))
    print("time: ", time.time() - t4)

    t5 = time.time()
    print("矩阵转置：")
    print(a.T)
    # print(transpose(a)) 
    print("time: ", time.time() - t5)

    t6 = time.time()
    print("手动矩阵相加：")
    print(add(c,d))
    print("time: ", time.time() - t6)

    t7 = time.time()
    print("手动矩阵相减：")
    print(sub(c,d))
    print("time: ", time.time() - t7)

    t8 = time.time()
    print("手动矩阵相乘：")
    print(dot(c,d))
    print(dot(a_1, b_1))
    print("time: ", time.time() - t8)

    t9 = time.time()
    print("手动矩阵相除：")
    print(div(c,d))
    print("time: ", time.time() - t9)

    t10 = time.time()
    print("手动矩阵转置：")
    print(transpose(a))
    print("time: ", time.time() - t10)

