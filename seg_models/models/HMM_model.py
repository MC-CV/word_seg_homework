import math
import numpy as np


class HMM:
    def __init__(self):
        self.STATES = ["B", "M", "E", "S"]
        self.array_A = {}  # 状态转移概率矩阵
        self.array_B = {}  # 发射概率矩阵
        self.array_E = {}  # 测试集存在的字符，但在训练集中不存在，发射概率矩阵
        self.array_Pi = {}  # 初始状态分布
        self.word_set = set()  # 训练数据集中所有字的集合
        self.count_dic = {}  # ‘B,M,E,S’每个状态在训练集中出现的次数
        self.line_num = 0  # 训练集语句数量

    # 初始化所有概率矩阵
    def Init_Array(self):
        for state0 in self.STATES:
            self.array_A[state0] = {}
            for state1 in self.STATES:
                self.array_A[state0][
                    state1
                ] = 0.0  # {'B': {'B': 0.0, 'M': 0.0, 'E': 0.0, 'S': 0.0}, 'M': {'B': 0.0, 'M': 0.0, 'E': 0.0, 'S': 0.0}, 'E': {'B': 0.0, 'M': 0.0, 'E': 0.0, 'S': 0.0}, 'S': {'B': 0.0, 'M': 0.0, 'E': 0.0, 'S': 0.0}}
        for state in self.STATES:
            self.array_Pi[state] = 0.0  # {'B': 0.0, 'M': 0.0, 'E': 0.0, 'S': 0.0}
            self.array_B[state] = {}  # {'B': {}, 'M': {}, 'E': {}, 'S': {}}
            array_E = {}
            self.count_dic[state] = 0  # {'B': 0, 'M': 0, 'E': 0, 'S': 0}

    # 对训练集获取状态标签
    def get_tag(self, word):
        tag = []
        if len(word) == 1:
            tag = ["S"]
        elif len(word) == 2:
            tag = ["B", "E"]
        else:
            num = len(word) - 2
            tag.append("B")
            tag.extend(["M"] * num)
            tag.append("E")
        return tag

    # 将参数估计的概率取对数，对概率0取无穷小-3.14e+100
    def Prob_Array(self):
        for key in self.array_Pi:
            if self.array_Pi[key] == 0:
                self.array_Pi[key] = -3.14e100
            else:
                self.array_Pi[key] = math.log(self.array_Pi[key] / self.line_num)
        for key0 in self.array_A:
            for key1 in self.array_A[key0]:
                if self.array_A[key0][key1] == 0.0:
                    self.array_A[key0][key1] = -3.14e100
                else:
                    self.array_A[key0][key1] = math.log(
                        self.array_A[key0][key1] / self.count_dic[key0]
                    )
        # print(array_A)
        for key in self.array_B:
            for word in self.array_B[key]:
                if self.array_B[key][word] == 0.0:
                    self.array_B[key][word] = -3.14e100
                else:
                    self.array_B[key][word] = math.log(
                        self.array_B[key][word] / self.count_dic[key]
                    )

    def Dic_Array(self, array_b):
        tmp = np.empty((4, len(array_b["B"])))
        for i in range(4):
            for j in range(len(array_b["B"])):
                tmp[i][j] = array_b[self.STATES[i]][list(self.word_set)[j]]
        return tmp

    # 判断一个字最大发射概率的状态
    def dist_tag(self):
        self.array_E["B"]["begin"] = 0
        self.array_E["M"]["begin"] = -3.14e100
        self.array_E["E"]["begin"] = -3.14e100
        self.array_E["S"]["begin"] = -3.14e100
        self.array_E["B"]["end"] = -3.14e100
        self.array_E["M"]["end"] = -3.14e100
        self.array_E["E"]["end"] = 0
        self.array_E["S"]["end"] = -3.14e100

    # 将字典转换成数组
    def dist_word(self, word0, word1, word2, array_b):
        if self.dist_tag(word0, array_b) == "S":
            self.array_E["B"][word1] = 0
            self.array_E["M"][word1] = -3.14e100
            self.array_E["E"][word1] = -3.14e100
            self.array_E["S"][word1] = -3.14e100
        return

    # Viterbi算法求测试集最优状态序列
    def Viterbi(self, sentence, array_pi, array_a, array_b):
        tab = [{}]  # 动态规划表
        path = {}
        # print()
        if sentence[0] not in array_b["B"]:
            for state in self.STATES:
                if state == "S":
                    array_b[state][sentence[0]] = 0
                else:
                    array_b[state][sentence[0]] = -3.14e100

        for state in self.STATES:
            tab[0][state] = array_pi[state] + array_b[state][sentence[0]]
            # print(tab[0][state])
            # tab[t][state]表示时刻t到达state状态的所有路径中，概率最大路径的概率值
            path[state] = [state]
        for i in range(1, len(sentence)):
            tab.append({})
            new_path = {}
            for state in self.STATES:
                if state == "B":
                    array_b[state]["begin"] = 0
                else:
                    array_b[state]["begin"] = -3.14e100
            for state in self.STATES:
                if state == "E":
                    array_b[state]["end"] = 0
                else:
                    array_b[state]["end"] = -3.14e100
            for state0 in self.STATES:
                items = []
                for state1 in self.STATES:
                    if sentence[i] not in array_b[state0]:  # 所有在测试集出现但没有在训练集中出现的字符
                        if sentence[i - 1] not in array_b[state0]:
                            prob = (
                                tab[i - 1][state1]
                                + array_a[state1][state0]
                                + array_b[state0]["end"]
                            )
                        else:
                            prob = (
                                tab[i - 1][state1]
                                + array_a[state1][state0]
                                + array_b[state0]["begin"]
                            )
                    else:
                        prob = (
                            tab[i - 1][state1]
                            + array_a[state1][state0]
                            + array_b[state0][sentence[i]]
                        )  # 计算每个字符对应STATES的概率
                    items.append((prob, state1))
                best = max(items)  # bset:(prob,state)
                tab[i][state0] = best[0]
                new_path[state0] = path[best[1]] + [state0]
            path = new_path

        prob, state = max(
            [(tab[len(sentence) - 1][state], state) for state in self.STATES]
        )
        return path[state]
