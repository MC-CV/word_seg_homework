# -*- coding: UTF-8 -*-
from tqdm import tqdm
import os
import json
from seg_models.utils import analyse, score
from seg_models.models.HMM_model import HMM
from seg_models.models.max_ngram import MaxProbCut
from seg_models.dataloader import dataprocessing, dict_dataloader
import argparse

root = r"/Users/mengchang/Documents/Courses/Artificial intelligence programming/final/Wordseg"
data_root = os.path.join(root, "icwb2-data")
root_out = os.path.join(root, "seg_models")
output_path = os.path.join(root_out, "outputs")


def parse_args():
    parser = argparse.ArgumentParser(description="train & test HMM model")
    parser.add_argument(
        "-d",
        "--dataset",
        default="pku",
        help="the type of datasets that you chosed(as,cityu,msr,pku)",
    )
    parser.add_argument(
        "-m", "--mode", default="train", help="train or eval or just inference"
    )
    parser.add_argument("-s", "--res", action="store_true", help="save the weights")
    parser.add_argument(
        "-o", "--out", default=output_path, help="where to output the results"
    )
    parser.add_argument(
        "-a",
        "--alg",
        default="HMM",
        help="the algorithms you choose(HMM,forward,backward,binary,n-gram)",
    )
    args = parser.parse_args()
    return args


def train(model, trainset, tp):
    for line in tqdm(trainset):
        line = line.strip()  # '上海 浦东 开发 与 法制 建设 同步'
        model.line_num += 1
        word_list = (
            []
        )  # ['上', '海', '浦', '东', '开', '发', '与', '法', '制', '建', '设', '同', '步']
        for k in range(len(line)):
            if line[k] == " " or line[k] == "\u3000":
                continue
            word_list.append(line[k])
        model.word_set = model.word_set | set(word_list)  # 训练集所有字的集合

        if tp == "pku" or tp == "msr":
            line = line.replace("  ", " ")
        elif tp == "as" or tp == "cityu":
            line = line.replace("\u3000", " ")
        line = line.split(" ")
        while line.__contains__(""):
            line.remove("")

        line_state = (
            []
        )  # 这句话的状态序列 ['B', 'E', 'B', 'E', 'B', 'E', 'S', 'B', 'E', 'B', 'E', 'B', 'E']
        for i in line:
            if i == "":
                continue
            line_state.extend(model.get_tag(i))
        if len(line_state) == 0:
            continue
        model.array_Pi[line_state[0]] += 1  # array_Pi用于计算初始状态分布概率

        for j in range(len(line_state) - 1):
            model.array_A[line_state[j]][line_state[j + 1]] += 1  # array_A计算状态转移概率
        # print((len(line_state),len(word_list)))
        assert len(line_state) == len(word_list)
        for p in range(len(line_state)):
            model.count_dic[line_state[p]] += 1  # 记录每一个状态的出现次数
            for state in model.STATES:

                if word_list[p] not in model.array_B[state]:
                    model.array_B[state][word_list[p]] = 0.0  # 保证每个字都在STATES的字典中

            model.array_B[line_state[p]][
                word_list[p]
            ] += 1  # array_B用于计算发射概率 {'B': {'上': 1.0, '海': 0.0, '浦': 1.0, '东': 0.0, '开': 1.0, '发': 0.0, '与': 0.0, '法': 1.0, '制': 0.0, ...},
            # 'M': {'上': 0.0, '海': 0.0, '浦': 0.0, '东': 0.0, '开': 0.0, '发': 0.0, '与': 0.0, '法': 0.0, '制': 0.0, ...},
            # 'E': {'上': 0.0, '海': 1.0, '浦': 0.0, '东': 1.0, '开': 0.0, '发': 1.0, '与': 0.0, '法': 0.0, '制': 1.0, ...},
            # 'S': {'上': 0.0, '海': 0.0, '浦': 0.0, '东': 0.0, '开': 0.0, '发': 0.0, '与': 1.0, '法': 0.0, '制': 0.0, ...}}就是指每个state对应的值是多少
    return model


def inference(model, testset, tp, dir):
    output = ""
    results_name = tp + "_array_A.json"
    with open(
        os.path.join(root_out, "results", results_name), "r", encoding="utf-8"
    ) as f:
        array_A = json.load(f)
    results_name = tp + "_array_Pi.json"
    with open(
        os.path.join(root_out, "results", results_name), "r", encoding="utf-8"
    ) as f:
        array_Pi = json.load(f)
    results_name = tp + "_array_B.json"
    with open(
        os.path.join(root_out, "results", results_name), "r", encoding="utf-8"
    ) as f:
        array_B = json.load(f)
    for line in tqdm(testset):
        line = line.strip()  # '外商投资企业成为中国外贸重要增长点'
        if line == "":
            output = output + "\n"
            continue
        tag = model.Viterbi(
            line, array_Pi, array_A, array_B
        )  # ['B', 'E', 'B', 'E', 'B', 'E', 'B', 'E', 'B', 'E', 'B', 'E', 'B', 'E', ...]
        seg = analyse.tag_seg(
            line, tag
        )  # ['外商', '投资', '企业', '成为', '中国', '外贸', '重要', '增长', '点']
        temp_list = ""  # 存储
        for i in range(len(seg)):
            if tp == "pku" or tp == "msr":
                temp_list = temp_list + seg[i] + "  "
            elif tp == "as":
                temp_list = temp_list + seg[i] + "\u3000"
            else:
                temp_list = temp_list + seg[i] + " "  # '外商 投资 企业 成为 中国 外贸 重要 增长 点 '
        output = output + temp_list + "\n"
        file_dir = tp + "_output.txt"
        outputfile = open(os.path.join(dir, file_dir), mode="w", encoding="utf-8")
        outputfile.write(output)


def inference_demo(model, testset, tp):
    output = ""
    results_name = tp + "_array_A.json"
    with open(
        os.path.join(root_out, "results", results_name), "r", encoding="utf-8"
    ) as f:
        array_A = json.load(f)
    results_name = tp + "_array_Pi.json"
    with open(
        os.path.join(root_out, "results", results_name), "r", encoding="utf-8"
    ) as f:
        array_Pi = json.load(f)
    results_name = tp + "_array_B.json"
    with open(
        os.path.join(root_out, "results", results_name), "r", encoding="utf-8"
    ) as f:
        array_B = json.load(f)
    line = testset
    line = line.strip()  # '外商投资企业成为中国外贸重要增长点'
    if line == "":
        output = output
    tag = model.Viterbi(
        line, array_Pi, array_A, array_B
    )  # ['B', 'E', 'B', 'E', 'B', 'E', 'B', 'E', 'B', 'E', 'B', 'E', 'B', 'E', ...]
    seg = analyse.tag_seg(
        line, tag
    )  # ['外商', '投资', '企业', '成为', '中国', '外贸', '重要', '增长', '点']
    temp_list = ""  # 存储
    for i in range(len(seg)):
        temp_list = temp_list + seg[i] + "/"

    output = output + temp_list
    return output


class TrainNgram:
    def __init__(self):
        self.word_dict = {}  # 词语频次词典
        self.transdict = {}  # 每个词后接词的出现个数

    """训练ngram参数"""

    def train(self, train_data_path, wordict_path, transdict_path, dataset):
        print("Start training:")
        self.transdict[u"<BEG>"] = {}
        self.word_dict["<BEG>"] = 0

        with open(train_data_path) as f:
            for sentence in tqdm(f.readlines()):
                self.word_dict["<BEG>"] += 1
                sentence = sentence.strip()
                if dataset == "cityu":
                    sentence = sentence.split(" ")
                elif dataset == "as":
                    sentence = sentence.split("\u3000")
                else:
                    sentence = sentence.split("  ")
                sentence_list = []
                # ['７月１４日', '', '下午４时', '', '，', '', '渭南市', '', '富平县庄里粮站', '', '。'], 得到每个词出现的个数
                for pos, words in enumerate(sentence):
                    if words != "":
                        sentence_list.append(words)
                # ['７月１４日', '下午４时', '渭南市', '富平县庄里粮站']
                for pos, words in enumerate(sentence_list):
                    if words not in self.word_dict.keys():
                        self.word_dict[words] = 1
                    else:
                        self.word_dict[words] += 1
                    # 词频统计
                    # 得到每个词后接词出现的个数，bigram <word1, word2>
                    words1, words2 = "", ""
                    # 如果是句首，则为<BEG，word>
                    if pos == 0:
                        words1, words2 = u"<BEG>", words
                    # 如果是句尾，则为<word, END>
                    elif pos == len(sentence_list) - 1:
                        words1, words2 = words, u"<END>"
                    # 如果非句首，句尾，则为 <word1, word2>
                    else:
                        words1, words2 = words, sentence_list[pos + 1]
                    # 统计当前词后接词语出现的次数：{‘我’：{‘是’：1， ‘爱’：2}}
                    if words not in self.transdict.keys():
                        self.transdict[words1] = {}
                    if words2 not in self.transdict[words1]:
                        self.transdict[words1][words2] = 1
                    else:
                        self.transdict[words1][words2] += 1

        self.save_model(self.word_dict, wordict_path)
        self.save_model(self.transdict, transdict_path)

    def save_model(self, word_dict, model_path):
        print("Start Saving:")
        f = open(model_path, "w")
        f.write(str(word_dict))
        f.close()
        print("Finish Saving:")


def demo(input):
    args = parse_args()
    model = HMM()
    type_data = args.dataset
    testset = input
    output = inference_demo(model, testset, type_data)
    return output


def main():
    args = parse_args()
    gold_name = args.dataset + "_test_gold.utf8"
    gold_path = os.path.join(data_root, "gold", gold_name)
    if args.alg == "HMM":
        model = HMM()
        type_data = args.dataset
        trainset, testset, goldset = dataprocessing(type_data, data_root)
        if args.mode == "train":
            model.Init_Array()
            print("Start training:")
            model = train(model, trainset, type_data)
            print("Finish training:")
            model.Prob_Array()  # 对概率取对数保证精度

        if args.res or args.mode == "train":
            assert args.mode == "train"
            print("Start Saving:")
            results_name = type_data + "_array_Pi.json"
            with open(
                os.path.join(root_out, "results", results_name), "w", encoding="utf-8"
            ) as f:
                f.write(json.dumps(model.array_Pi, ensure_ascii=False))
            results_name = type_data + "_array_A.json"
            with open(
                os.path.join(root_out, "results", results_name), "w", encoding="utf-8"
            ) as f:
                f.write(json.dumps(model.array_A, ensure_ascii=False))
            results_name = type_data + "_array_B.json"
            with open(
                os.path.join(root_out, "results", results_name), "w", encoding="utf-8"
            ) as f:
                f.write(json.dumps(model.array_B, ensure_ascii=False))
            print("Finish Saving in :", os.path.join(root_out, "results"))

        if args.mode == "inference" or args.mode == "train":
            print("Start inferencing:")
            inference(model, testset, type_data, args.out)
            print("Finish inferencing:")

        if args.mode == "train" or args.mode == "eval":
            print("Start evaling:")
            cutset = []
            file_dir = type_data + "_output.txt"
            with open(os.path.join(args.out, file_dir), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    cutset.append(line)
                res = analyse.test(goldset, cutset, type_data)
                print(
                    "模型：",
                    args.alg,
                    "数据集：",
                    type_data,
                    "\n准确率:",
                    res[0],
                    "\n召回率:",
                    res[1],
                    "\nF1值",
                    res[2],
                )
            print("Finish evaling:")
    elif args.alg == "forward" or args.alg == "backward" or args.alg == "binary":
        model = dict_dataloader(args.dataset, data_root)
        P, R, F = score(args, gold_path, model)
        print(
            "模型：", args.alg, "数据集：", args.dataset, "\n准确率:", P, "\n召回率:", R, "\nF1值", F
        )
    elif args.alg == "n-gram":
        train_name = args.dataset + "_training.utf8"
        train_data_path = os.path.join(data_root, "training", train_name)
        wordict_path = os.path.join(root_out, "results", "word_dict.model")
        transdict_path = os.path.join(root_out, "results", "trans_dict.model")
        trainer = TrainNgram()
        trainer.train(train_data_path, wordict_path, transdict_path, args.dataset)
        model = MaxProbCut(wordict_path, transdict_path)
        P, R, F = score(args, gold_path, model)
        print(
            "模型：", args.alg, "数据集：", args.dataset, "\n准确率:", P, "\n召回率:", R, "\nF1值", F
        )

    else:
        raise ("please choose the right algorithm.")


if __name__ == "__main__":
    main()

