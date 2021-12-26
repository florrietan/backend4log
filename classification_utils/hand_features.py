import pandas as pd
import numpy as np

def get_hand_features(file_name: str, file_dir: str):
    file_dir = file_dir+'/'
    df_log = pd.read_csv(file_dir+file_name) # 读入日志

    '''
    # 假设最后要保存的是对象df，下面是存储格式
    df_log.to_csv(file_dir+'hand_'+file_name)
    print('[flask says] 手工特征已存储到 '+file_dir+'hand_'+file_name)
    '''


    return 'hand features done'