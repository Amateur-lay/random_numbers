import random
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


class Normal_generator():
    def __init__(self, num, mu, sigma, seed=None):
        self.num = num
        self.mu = mu
        self.sigma = sigma
        if seed != None:
            random.seed(seed)

    def generate(self, algo):
        if algo == 1:
            return self.__box_muller__()
        if algo == 2:
            return self.__rejection_1__()
        if algo == 3:
            return self.__rejection_2__()
        if algo == 4:
            return self.__rejection_boxmuller__()

    def __box_muller__(self):
        x = [None] * self.num
        u1 = [random.uniform(0, 1) for i in range(self.num)]
        u2 = [random.uniform(0, 1) for i in range(self.num)]
        for i in range(self.num):
            x[i] = np.sqrt(-2 * np.log(u1[i])) * np.sin(2 * np.pi * u2[i])
        x = np.array(x)
        x = x * self.sigma + self.mu
        return x

    def __p__(self, x):
        return 1 / (self.sigma * np.sqrt(2 * np.pi)) * np.exp(-1 * (x - self.mu) ** 2 / (2 * self.sigma ** 2))

    def __rejection_1__(self):
        M = 1 / (self.sigma * np.sqrt(2 * np.pi))
        x_min = self.mu - 3 * self.sigma
        x_max = self.mu + 3 * self.sigma
        x = [None] * self.num
        for i in range(self.num):
            while 1:
                u1 = random.uniform(0, 1)
                u2 = random.uniform(0, 1)
                cand = x_min + (x_max - x_min) * u1
                y = M * u2
                if y <= self.__p__(cand):
                    x[i] = cand
                    break
        return x

    def __rejection_2__(self):
        x = [None] * self.num
        for i in range(self.num):
            while 1:
                u1 = 1 - random.uniform(0, 1)
                cand = -np.log(u1)
                y = np.exp(-cand) * random.uniform(0, 1)
                if 1 / (np.sqrt(2 * np.pi)) * np.exp(-1 * cand ** 2 / 2) > y:
                    x[i] = cand
                    if random.uniform(0, 1) < 0.5:
                        x[i] = -x[i]
                    break
        x = np.array(x)
        x = x * self.sigma + self.mu
        return x

    def __rejection_boxmuller__(self):
        x = [None] * self.num
        for i in range(self.num):
            while 1:
                r = np.sqrt(-2 * np.log(random.uniform(0, 1)))
                u1 = 2 * random.uniform(0, 1) - 1
                u2 = 2 * random.uniform(0, 1) - 1
                if u1 ** 2 + u2 ** 2 < 1:
                    x[i] = r * u1 / np.sqrt(u1 ** 2 + u2 ** 2)
                    break
        x = np.array(x)
        x = x * self.sigma + self.mu
        return x


if __name__ == '__main__':
    a = Normal_generator(10000, 0, 1)
    x1 = a.generate(1)
    x2 = a.generate(3)
    x3 = a.generate(4)
    fig, axs = plt.subplots(3, 1)
    plt.suptitle("Normal random numbers")
    u1=sns.histplot(x1, kde=True, ax=axs[0])
    u1.set_title("method 1")
    u1.set_xlabel("X")
    u1.set_ylabel("counts/density")
    u1.legend(labels=['density','hist'],loc=1,prop={'size':6})
    u2=sns.histplot(x2, kde=True, ax=axs[1])
    u2.set_title("method 2")
    u2.set_xlabel("X")
    u2.set_ylabel("counts/density")
    u2.legend(labels=['density','hist'],loc=1,prop={'size':6})
    u3=sns.histplot(x3, kde=True, ax=axs[2])
    u3.set_title("method 3")
    u3.set_xlabel("X")
    u3.set_ylabel("counts/density")
    u3.legend(labels=['density','hist'],loc=1,prop={'size':6})
    plt.tight_layout()
    plt.show()
