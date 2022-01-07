from classification_utils.bert_features import get_bert_features
from classification_utils.hand_features import get_hand_features
from classification_utils.predict import predict_device
import joblib
import pandas as pd
import numpy as np
from tensorflow import keras
import csv
import os


def get_features(file_name: str, file_dir: str):
    print('[flask says] 开始对文件' + file_name + '做特征工程')
    # get_bert_features(file_name, file_dir)
    # get_hand_features(file_name, file_dir)

    # 然后就可以送入模型做预测了
    # predict_device(file_name, file_dir)
    csv_path = file_dir + '/' + file_name
    return_list = run_model(csv_path)

    return return_list


def OneClassSVM(csv_path):
    df = pd.read_csv(csv_path)
    y = df.pop('label')

    path = os.getcwd()
    clf = joblib.load(path+"/classification_utils/saved_model/OCSVM.m")
    y_pred_train = clf.predict(df)
    print("OneClassSVM says:", y_pred_train)
    return y_pred_train


def NN_classify(csv_path):
    path = os.getcwd()
    model = keras.models.load_model(path+"/classification_utils/saved_model/NN_predict_model.h5")
    df = pd.read_csv(csv_path)
    y = df.pop('label')
    ansDict2 = {0: 'nginx', 1: 'linux', 2: 'mysql', 3: 'zabbix'}
    y_pred = np.argmax(model.predict(df), axis=1)

    ansList = []
    for i in y_pred:
        ansList.append(ansDict2.get(i))

    print('\n', "[NN_classify says]", ansList, '\n')
    return ansList


# 功能函数：输入日志csv，输出特征矩阵csv（最后一列为标签）
def log2fea(csvfile):
    # 基于公式：df=pd.read_csv()
    def getArti(df, dflen):
        # 日志长度
        def getLength(st):
            return len(st)

        for index, row in df.iterrows():
            df.loc[index, 'Length'] = getLength(str(df.loc[index, '_raw']))

        # 单词个数
        def getNumOfWord(st):
            num = 0
            last = True
            for index, ch in enumerate(st):
                if ch.isalpha() and not last:
                    num += 1
                    last = True
                    continue
                elif ch.isalpha():
                    last = True
                else:
                    last = False
            return num

        for index, row in df.iterrows():
            df.loc[index, 'NumOfWord'] = getNumOfWord(str(df.loc[index, '_raw']))

        # 标点个数
        def getNumOfPunctuation(st):
            num = 0
            for index, ch in enumerate(st):
                if not ch.isalnum():
                    num += 1
            return num

        for index, row in df.iterrows():
            df.loc[index, 'NumOfPunctuation'] = getNumOfPunctuation(str(df.loc[index, '_raw']))

        # 数字个数
        def getNumOfNum(st):
            num = 0
            for index, ch in enumerate(st):
                if ch.isdigit():
                    num += 1
            return num

        for index, row in df.iterrows():
            df.loc[index, 'NumOfNum'] = getNumOfNum(str(df.loc[index, '_raw']))

        # 平均单词长度
        def getAverageLength(st):
            num = 0
            currentNum = 0
            last = True
            for index, ch in enumerate(st):
                if ch.isalpha() and not last:
                    currentNum += 1
                    last = True
                elif ch.isalpha():
                    currentNum += 1
                    last = True
                else:
                    num += currentNum
                    currentNum = 0
                    last = False

            return num

        for index, row in df.iterrows():
            df.loc[index, 'AverageLength'] = getAverageLength(str(df.loc[index, '_raw'])) / (
                    getNumOfWord(str(df.loc[index, '_raw'])) + 0.1)

        # 标点种类数
        def getTypeOfPunctuation(st):
            num = 0
            list = []
            for index, ch in enumerate(st):
                if not ch.isalnum():
                    if ch in list:
                        continue
                    else:
                        list.append(ch)
                        num += 1
            return num

        for index, row in df.iterrows():
            df.loc[index, 'TypeOfPunctuation'] = getTypeOfPunctuation(str(df.loc[index, '_raw']))

        # output
        df = df.drop(['_time', '_raw'], axis=1)
        df['LogType'] = 'mysql'  # gai

        df_vec = df
        arr_vec = df_vec.values

        return arr_vec

    # df2arti
    # 测试成功
    def df2arr_arti(vec):
        row = vec.shape[0]  # 应该为日志的条数
        # row = 2  # 测试
        col_all = 6  # 实际上有7个，但是最后一个是日志的类别，不需要
        col = col_all

        arr_arti = np.ones((row, col), dtype=float)
        for i in range(arr_arti.shape[0]):
            for j in range(arr_arti.shape[1]):
                arr_arti[i][j] = vec[i][j]

        return arr_arti

    # 获得标签
    # 测试成功
    def df2label(vec):
        row = vec.shape[0]  # 应该为日志的条数
        # row = 2  # 测试
        col_all = 1  # 实际上有7个，但是只取最后一个，即日志的类别
        col = col_all

        arr_label = np.empty((row, col), dtype=(str, 99))
        for i in range(arr_label.shape[0]):
            for j in range(arr_label.shape[1]):
                arr_label[i][j] = str(vec[i][-1])

        return arr_label

    # 数据处理
    def feaProcessing(csvName):
        # 导入原来的日志特征（和label一起）
        with open(csvName, 'r') as c:
            reader = csv.reader(c)
            result_List = list(reader)
        result_Array = np.array(result_List)

        # 设置特征矩阵相关参数
        row_bert_arti = result_Array.shape[0] - 1  # 实际上row需要-1，减去的是多余的解释行
        col_bert_arti = result_Array.shape[1] - 1  # 减去的是label

        row = row_bert_arti
        col = col_bert_arti

        fea_Matrix = np.ones((row, col))  # 特征矩阵
        lab_Cla = []  # 多分类样本结果

        # 初始化特侦矩阵

        # 先获取人工特征和bert特征
        for i in range(row):
            for j in range(col_bert_arti):
                fea_Matrix[i][j] = result_Array[i + 1][j]

        # 初始化多分类样本结果
        for i in range(row):
            lab_Cla.append(result_Array[i + 1][col_bert_arti])

        # 归一化，让各个特征的范围相同
        # 按照更严格的规则（来自知乎），归一化应该等训练集和测试集划分之后再做，具体如下注释部分
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        fea_Matrix = min_max_scaler.fit_transform(fea_Matrix)

        # 规整成csv文件
        row_collected = fea_Matrix.shape[0]
        col_collected = fea_Matrix.shape[1]

        collected_data = [[] for i in range(row_collected)]

        for i in range(row_collected):
            for j in range(col_collected):
                collected_data[i].append(fea_Matrix[i][j])

        for i in range(row_collected):
            collected_data[i].append(lab_Cla[i])

        return collected_data

    df_test = pd.read_csv(csvfile)
    df_testa = getArti(df_test, df_test.shape[0])
    arr_testa = df2arr_arti(df_testa)  # 获取手工特征矩阵
    arr_label = df2label(df_testa)  # 获取标签

    arr_final = np.hstack((arr_testa, arr_label))  # 拼接矩阵：手工+bert+标签
    list_final = arr_final.tolist()  # 转成list

    file_name = 'allfealab_nginx.csv'

    # 创造,清空,重塑文件
    f = open(file_name, 'w')
    f.close()

    arti_head = ["fea" + str(i + 1) for i in range(arr_testa.shape[1])]
    label_head = ["label"]
    all_head = arti_head + label_head

    with open(file_name, 'a+', newline='') as ff:
        csv_writer = csv.writer(ff)
        csv_writer.writerow(all_head)

    for i in range(len(list_final)):
        with open(file_name, 'a+', newline='') as ff:  # 输出到最终的文件
            csv_writer = csv.writer(ff)
            csv_writer.writerow(list_final[i])

    return file_name  # 返回拼接的特征矩阵,和数据预处理之后的特征矩阵所对应的csv文件名


def run_model(csv_path):
    csv_name = log2fea(csv_path)
    classify_ans = NN_classify(csv_name)
    abnormal_ans = OneClassSVM(csv_name)
    return classify_ans, abnormal_ans
