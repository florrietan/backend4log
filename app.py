from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import untangle
import pandas as pd
import os


from classification_utils.feature_eng import get_features

upload_type = ['.csv', '.xlsx','.txt']

# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

@app.route('/',methods=['GET'])
def index(): # 测试用的极简前端
    page=open('index.html',encoding='utf-8');
    res=page.read()
    return res;

# 前端对应的url用 http://localhost:5000/upload
@app.route('/upload', methods=['POST'])
def upload_file(): # 接收前端上传的待分类的日志文件，应该只包含一列原始的日志数据
    uploaded_file = request.files['file']
    filename = uploaded_file.filename
    if uploaded_file.filename != '':
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in upload_type:
                return 'file type not allowed, we only allow csv xlsx and txt'
                #abort(400)
            file_dir = os.path.join(os.getcwd(), 'upload_files')
            file_path = os.path.join(file_dir, uploaded_file.filename)
            print('[flask says] '+filename+'已收到')

            uploaded_file.save(file_path) # raw数据存入upload_files文件夹中
            list = get_features(filename,file_dir) # 对其做特征工程
            s = ""
            for each in list:
                s += str(each)
    return 'upload succeeded! '+ s

# 前端对应的url用 http://localhost:5000/class_result
@app.route('/class_result', methods=['GET'])
def classification_result(): # 返回分类结果，就用json格式吧，方便前端处理
    result = []
    '''
    result_df = pd.read_csv(...)
    for line in result_df:
        dict = {'log':line['...'],'device':line['...']}
        result.append(dict)
    '''
    return jsonify(result)






if __name__ == '__main__':
    app.run()