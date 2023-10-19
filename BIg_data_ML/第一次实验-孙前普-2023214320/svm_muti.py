import numpy as np
import struct
import os
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class MINST():

    def __init__(self) -> None:
        super().__init__()
        self.train_img, self.train_label_o, self.train_label, self.test_img, self.test_label_o, self.test_label = self.load_minst()
        train_mask = (self.train_label == 0)
        self.train_label[train_mask] = -1
        test_mask = (self.test_label == 0)
        self.test_label[test_mask] = -1
        assert self.train_img.shape[0] == self.train_label.shape[0], "train_img.shape[0] != train_label.shape[0]"
        assert self.test_img.shape[0] == self.test_label.shape[0], "test_img.shape[0] != test_label.shape[0]"

    def load_img(self, load_path):

        with open(load_path, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            img = np.fromfile(f, dtype=np.uint8).reshape(num,-1)
        return img

    def load_label(self, load_path):

        with open(load_path, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            label = np.fromfile(f, dtype=np.uint8)
        return label


    def load_minst(self):
        train_img = self.load_img('minst/train-images-idx3-ubyte')
        train_label_o = self.load_label('minst/train-labels-idx1-ubyte').astype(np.float32)
        train_label = np.eye(10)[self.load_label('minst/train-labels-idx1-ubyte')]
        test_img = self.load_img('minst/t10k-images-idx3-ubyte')
        test_label = np.eye(10)[self.load_label('minst/t10k-labels-idx1-ubyte')].astype(np.float32)
        test_label_o = self.load_label('minst/t10k-labels-idx1-ubyte')

        return train_img, train_label_o, train_label, test_img, test_label_o, test_label

class SVM():

    def __init__(self, train_data, C , kernel, tol, max_passes=10):
        self.C = C
        self.kernel = kernel
        self.tol = tol
        self.max_passes = max_passes
        self.m, self.n = train_data.shape
        self.alphas = np.zeros(self.m)
        self.b = 0.0
        self.w = np.zeros(self.n)
        self.alphas_list = []
        self.bias = []
        self.sigma = 1.0
    
    def smo_train(self):
        passes = 0
        while(passes < self.max_passes):
            num_change_alphas = 0
            for i in range(self.m):
                Ei = self.E(i)
                # 在精度范围内满足停机条件
                # print("self.y[i] : ", self.y[i] * Ei,  "self.alphas[i] : ", self.alphas[i])
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
            
        
        return self.alphas, self.b
  

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

    def predict_binary(self, X, c):
        y_scores = []
        self.y = self.y_original[:, c]
        for x in X:
            #* slef.x是所有数据， x是单个数据
            y_score = np.sum(self.y * self.K(self.x, x) * self.alphas_list[c]) + self.bias[c]
            y_scores.append(y_score)
        return y_scores

    def predict(self, X):
        print("--------------predict--------------")
        all_y_scores = []
        for c in range(self.n_classes):
            y_scores = []
            self.y = self.y_original[:, c]
            for x in X:
                #* slef.x是所有数据， x是单个数据
                y_score = np.sum(self.y * self.K(self.x, x) * self.alphas_list[c]) + self.bias[c]
                y_scores.append(y_score)
                
            all_y_scores.append(y_scores)
        all_y_scores = np.vstack((all_y_scores)).transpose().astype(np.float128)  # [n_samples,n_classes]
        prob = np.exp(all_y_scores) / np.sum(np.exp(all_y_scores), 1, keepdims=True)
        y_pred = np.argmax(prob, axis=-1)
        return y_pred

    def fit_binary(self, y):
        self.y = y
        self.alphas = np.zeros(self.m)
        self.b = 0.0
        alphas, bias = self.smo_train()
        self.alphas_list.append(alphas)
        self.bias.append(bias)

    def fit(self, X, Y):
        self.x = X
        self.y_original = Y.copy()
        self.alphas_list = []  # 用来保存每个二分类器计算得到的alpha参数，因为在多分类问题中
        self.bias = []  # 采用的是ovr策略； bias用来保存每个分类器对应的偏置
        self.n_classes = Y.shape[1]  # 数据集的类别数量
        for c in range(self.n_classes):
            print(f"--------------fit_{c}_class--------------")
            self.fit_binary(Y[:, c])  # 分别为每个类别拟合得到一个二分类器
        self.alphas_list = np.vstack((self.alphas_list))  # [n_classes，n_samples]
        self.bias = np.array(self.bias)  # [n_classes,]


if __name__ == "__main__":
    
    dataset = MINST()
    x_train = dataset.train_img / 255.0
    y_train = dataset.train_label 
    y_train_o = dataset.train_label_o

    x_test = dataset.test_img / 255.0
    y_test = dataset.test_label
    y_test_o = dataset.test_label_o
    
    x_train = x_train[:100]
    y_train = y_train[:100]
    
    x_test = x_test[:1000]
    y_test_o = y_test_o[:1000]
    clf = SVM(x_train, C=10000, kernel="rbf", tol=0.001, max_passes=1)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    print(pred)
    print(y_test_o)
    acc = accuracy_score(y_test_o, pred)
    print("acc : ", acc)
    print("end")