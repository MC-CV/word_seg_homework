import os
from seg_models.models.max_match_models import CutWords


def dataprocessing(tp, root):
    train_name = tp + "_training.utf8"
    test_name = tp + "_test.utf8"
    gold_name = tp + "_test_gold.utf8"

    trainset = open(
        os.path.join(root, "training", train_name), encoding="utf-8"
    )  # 读取训练集
    testset = open(os.path.join(root, "testing", test_name), encoding="utf-8")  # 读取测试集
    goldset = open(os.path.join(root, "gold", gold_name), encoding="utf-8")

    trainset = list(trainset)
    testset = list(testset)
    goldset = list(goldset)
    return trainset, testset, goldset


def dict_dataloader(tp, data_root):
    dict_name = tp + "_training_words.utf8"
    dict_path = os.path.join(data_root, "gold", dict_name)
    model = CutWords(dict_path)
    return model
