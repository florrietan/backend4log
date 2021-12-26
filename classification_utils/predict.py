import pandas as pd
import numpy as np

def predict_device(file_name: str, file_dir: str):

    file_dir = file_dir + '/'
    #读取bert特征
    bert_features = np.load(file_dir+'bert_'+file_name.strip('.csv')+'.npy')
    '''
    # 读取手工特征
    hand_features = pd.read_csv(file_dir+'hand_'+file_name)
    '''

    # 预测结束后，可将预测结果也保存下来，到app.py的classification_result()中读取、返回给前端
    return