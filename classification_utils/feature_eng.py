from classification_utils.bert_features import get_bert_features
from classification_utils.hand_features import get_hand_features
from classification_utils.predict import predict_device

def get_features(file_name: str, file_dir: str):
    print('[flask says] 开始对文件'+file_name+'做特征工程')
    get_bert_features(file_name, file_dir)
    get_hand_features(file_name, file_dir)

    # 然后就可以送入模型做预测了
    predict_device(file_name, file_dir)

    return 'all features done'