import random
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

class Possion_generator():
    def __init__(self, num, p_lambda, seed=None):
        self.num = num
        self.p_lambda = p_lambda
        self.table = dict()
        if seed != None:
            random.seed(seed)

    def generate(self, algo):
        if algo == 1:
            return self.__inverse__()
        if algo == 2:
            return self.__inverse_2__()
        if algo == 3:
            return self.__exp_generate__()

    def __inverse__(self):
        x = [None] * self.num
        u = [random.uniform(0, 1) for i in range(self.num)]
        for i in range(self.num):
            p = np.exp(-1 * self.p_lambda)
            f = p
            k = 0
            while 1:
                if u[i] < f:
                    x[i] = k
                    break
                p = (self.p_lambda / (k + 1)) * p
                f = f + p
                k += 1
        return x

    def __inverse_2__(self):
        x = [None] * self.num
        u = [random.uniform(0, 1) for i in range(self.num)]
        for i in range(self.num):
            k = int(np.floor(self.p_lambda))
            while 1:
                if self.__table__(k) < u[i] and self.__table__(k + 1) >= u[i]:
                    x[i] = k + 1
                    break
                elif self.__table__(k + 1) < u[i]:
                    k += 1
                    continue
                else:
                    k -= 1
                    continue
        return x

    def __distribution__(self, k):
        if k < 0:
            return 0
        else:
            return self.__distribution__(k - 1) + np.exp(-1 * self.p_lambda) * self.p_lambda ** (k) / np.math.factorial(
                k)

    def __table__(self, k):
        if self.table.get(k):
            return self.table[k]
        else:
            self.table[k] = self.__distribution__(k)
            return self.table[k]

    def __exp_generate__(self):
        x = [None] * self.num
        for i in range(self.num):
            u = random.uniform(0, 1)
            k = 1
            while u >= np.exp(-self.p_lambda):
                u *= random.uniform(0, 1)
                k += 1
            x[i] = k - 1
        return x


if __name__ == '__main__':
    a = Possion_generator(10000,4)
    x1 = a.generate(1)
    x2 = a.generate(2)
    x3 = a.generate(3)
    fig, axs = plt.subplots(3, 1)
    plt.suptitle("Possion random numbers")
    u1=sns.histplot(x1, kde=False, ax=axs[0])
    u1.set_title("method 1")
    u1.set_xlabel("X")
    u1.set_ylabel("counts/density")
    u1.legend(labels=['hist'],loc=1,prop={'size':6})
    u2=sns.histplot(x2, kde=False, ax=axs[1])
    u2.set_title("method 2")
    u2.set_xlabel("X")
    u2.set_ylabel("counts/density")
    u2.legend(labels=['hist'],loc=1,prop={'size':6})
    u3=sns.histplot(x3, kde=False, ax=axs[2])
    u3.set_title("method 3")
    u3.set_xlabel("X")
    u3.set_ylabel("counts/density")
    u3.legend(labels=['hist'],loc=1,prop={'size':6})
    plt.tight_layout()
    plt.show()

