from bert_serving.client import BertClient
import pandas as pd
import numpy as np


def get_bert_features(file_name: str, file_dir: str):
    bc = BertClient(ip='10.60.38.173',check_length=False)# ip中填入服务器地址
    file_dir = file_dir+'/'
    df_log = pd.read_csv(file_dir+file_name) # 读入日志

    vec_array = []
    i = 0

    for line in df_log['log']:
        str1 = str(line)
        vec = bc.encode([str1])
        vec_array.append(vec)
        print('[flask says] 当前正在计算第'+str(i+1)+'/'+str(len(df_log))+'条日志的bert向量') #观察进度
        i += 1

    np.save(file_dir+'bert_'+file_name.strip('.csv')+'.npy',vec_array)
    print('[flask says] bert特征已存储到 '+file_dir+'bert_'+file_name.strip('.csv')+'.npy')

    return 'bert features done'