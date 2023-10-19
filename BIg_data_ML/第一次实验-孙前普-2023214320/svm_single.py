import numpy as np
import struct
import os
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class SVM():

    def __init__(self, X, y, C , kernel, tol, max_passes=10):
        self.x = X 
        self.y = y
        self.C = C
        self.kernel = kernel
        self.tol = tol
        self.max_passes = max_passes
        self.m, self.n = X.shape
        self.alphas = np.zeros(self.m)
        self.b = 0.0
        self.w = np.zeros(self.n)
    
    def smo_train(self):
        passes = 0
        while(passes < self.max_passes):
            num_change_alphas = 0
            for i in range(self.m):
                Ei = self.E(i)
                # 在精度范围内满足停机条件
                if (self.y[i] * Ei < -self.tol and self.alphas[i] < self.C) or (self.y[i] * Ei > self.tol and self.alphas[i] > 0):
                    #! 随机找一个不等于i的j， 这里可以改进
                    j = self.select_j(i, self.m)
                    Ej = self.E(j)

                    alpha_i_old = self.alphas[i].copy()
                    alpha_j_old = self.alphas[j].copy()
 
                    # 计算alpha的上下界
                    L, H = self.compute_L_H(i, j, alpha_i_old, alpha_j_old)
                    if L == H:
                        continue
                    # 计算eta
                    eta = self.compute_eta(i, j)
                    if eta <= 0:
                        continue

                    #* 先更新alpha_j
                    alphas_j_new = alpha_j_old + self.y[j] * (Ei - Ej) / eta
                    
                    self.alphas[j] = self.clip_alpha(alphas_j_new, L, H)
                
                    if np.abs(self.alphas[j] - alpha_j_old) < 10e-5:
                        continue
                    #* 再更新alpha_i
                    self.alphas[i] = alpha_i_old + self.y[i] * self.y[j] * (alpha_j_old - self.alphas[j])


                    #* 根据b是否在0-C之间选择更新self.b
                    b1_new = self.b - Ei - self.y[i] * self.K(self.x[i], self.x[i]) * (self.alphas[i] - alpha_i_old) - self.y[j] * self.K(self.x[i], self.x[j]) * (self.alphas[j] - alpha_j_old)
                    b2_new = self.b - Ej - self.y[i] * self.K(self.x[i], self.x[j]) * (self.alphas[i] - alpha_i_old) - self.y[j] * self.K(self.x[j], self.x[j]) * (self.alphas[j] - alpha_j_old)
                    self.b = self.judge_b(i, j, b1_new, b2_new)

                    num_change_alphas += 1
                    
                # print("iter: %d i:%d, pairs changed %d" % (passes, i, num_change_alphas))
            
            if(num_change_alphas == 0):
                passes += 1
            else:
                passes = 0
            
            self.w = self.compute_w(self.alphas)
        
        return self.alphas, self.w, self.b
  

    def select_j(self, i, m):
        j = np.random.randint(m)
        while i == j:
            j = np.random.randint(m)
        return j

    def compute_eta(self, i, j):
        return self.K(self.x[i], self.x[i]) + self.K(self.x[j], self.x[j]) - 2.0 * self.K(self.x[i], self.x[j]) 
    
    def K(self, x1, x2):

        if self.kernel == "linear":
            return np.dot(x1, x2)
        elif self.kernel == "poly":
            return (np.dot(x1, x2) + 1) ** 2
        elif self.kernel == "rbf":
            return np.exp(-np.sum((x1 - x2) ** 2) / (2 * self.sigma ** 2))
        else:  
            raise("kernel error")

    def f(self, x, y, alphas, b):
        return np.sum(alphas * y * self.K(self.x, x)) + b    
    
    def E(self, i):
        return self.f(self.x[i], self.y, self.alphas, self.b) - self.y[i]
    
    def compute_L_H(self, i, j, alpha_i_old, alpha_j_old):
        if self.y[i] != self.y[j]:
            L = np.max((0, alpha_j_old - alpha_i_old))
            H = np.min((self.C, self.C + alpha_j_old - alpha_i_old))
        else:
            L = np.max((0, alpha_i_old + alpha_j_old - self.C))
            H = np.min((self.C, alpha_i_old + alpha_j_old))
        return L, H
    
    def clip_alpha(self, alpha, L, H):
        if alpha > H:
            alpha = H
        elif alpha >= L and alpha <= H:
            alpha = alpha
        else:
            alpha = L
        return alpha
    
    def judge_b(self, i, j, b1_new, b2_new):

        if 0 < self.alphas[i] and self.alphas[i] < self.C:
            b = b1_new
        elif 0 < self.alphas[j] and self.alphas[j] < self.C:
            b = b2_new
        else:
            b = b1_new + b2_new / 2.0

        return b

    def compute_w(self, alphas):
        p1 = self.y.reshape(-1, 1) * self.x
        p2 = alphas.reshape(-1, 1) * p1
        return np.sum(p2, axis=0)


    def predict(self, data_x):
        y = np.sign(np.dot(self.x, self.w) + self.b)
        return y
                
def draw_p_l(data_x, data_y, w, b):
    plt.scatter(data_x[:, 0], data_x[:, 1], c=data_y)
    x1 = np.linspace(0, 5, 100)
    x2 = (-b - w[0] * x1) / w[1]
    plt.plot(x1, x2, c='red')
    plt.savefig("svm.png")

if __name__ == "__main__":

    data_x = np.array([[5, 1], [0, 2], [1, 5], [3.0, 2], [1, 2], [3, 5], [1.5, 6], [4.5, 6], [0, 7]])
    data_y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1])
    clf = SVM(data_x, data_y, 1.5, 'linear', 0.001, max_passes=200)
    alphas, w, b = clf.smo_train()
    y = clf.predict(data_x)
    print(y)
    draw_p_l(data_x, data_y, w, b)

    
