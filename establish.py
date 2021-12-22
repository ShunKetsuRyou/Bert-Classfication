# -*- coding: UTF-8 -*-
import numpy as np
import predict

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
txt = str()
def dataProcess(txt):
    specialChars = "~`@!#$%^&*()_-─》+=|\}】【{><.?/:;；、'" 
    for specialChar in specialChars:
      txt = txt.replace(specialChar, '')
    txt = txt.replace('1', '一')
    txt = txt.replace('2', '二')
    txt = txt.replace('3', '三')
    txt = txt.replace('4', '四')
    txt = txt.replace('5', '五')
    txt = txt.replace('6', '六')
    txt = txt.replace('7', '七')
    txt = txt.replace('8', '八')
    txt = txt.replace('9', '九')
    txt = txt.replace('0', '零')
    txt = txt.replace('A', '欸')
    txt = txt.replace('B', '逼')
    txt = txt.replace('C', '溪')
    txt = txt.replace('D', '低')
    txt = txt.replace('E', '醫')
    txt = txt.replace('F', '福')
    txt = txt.replace('G', '居')
    txt = txt.replace('H', '取')
    txt = txt.replace('I', '哀')
    txt = txt.replace('J', '接')
    txt = txt.replace('K', '科')
    txt = txt.replace('L', '樓')
    txt = txt.replace('M', '母')
    txt = txt.replace('N', '恩')
    txt = txt.replace('O', '喔')
    txt = txt.replace('P', '批')
    txt = txt.replace('Q', '哭')
    txt = txt.replace('R', '阿')
    txt = txt.replace('S', '使')
    txt = txt.replace('T', '踢')
    txt = txt.replace('U', '優')
    txt = txt.replace('V', '複')
    txt = txt.replace('W', '搭')
    txt = txt.replace('X', '差')
    txt = txt.replace('Y', '歪')
    txt = txt.replace('Z', '末')
    txt = txt.replace('a', '欸')
    txt = txt.replace('b', '逼')
    txt = txt.replace('c', '溪')
    txt = txt.replace('d', '低')
    txt = txt.replace('e', '醫')
    txt = txt.replace('f', '福')
    txt = txt.replace('g', '居')
    txt = txt.replace('h', '取')
    txt = txt.replace('i', '哀')
    txt = txt.replace('j', '接')
    txt = txt.replace('k', '科')
    txt = txt.replace('l', '樓')
    txt = txt.replace('m', '母')
    txt = txt.replace('n', '恩')
    txt = txt.replace('o', '喔')
    txt = txt.replace('p', '批')
    txt = txt.replace('q', '哭')
    txt = txt.replace('r', '阿')
    txt = txt.replace('s', '使')
    txt = txt.replace('t', '踢')
    txt = txt.replace('u', '優')
    txt = txt.replace('v', '複')
    txt = txt.replace('w', '搭')
    txt = txt.replace('x', '差')
    txt = txt.replace('y', '歪')
    txt = txt.replace('z', '末')
    txt = txt.replace('"', '')
    txt = txt.replace(',', '')
    txt = txt.replace('。', '')
    txt = txt.replace('，', '')
    txt = txt.replace(' ','')
    return txt
    
@app.route('/')
def index():
    return 'please add /predict'

@app.route('/predict', methods=['POST'])

def postInput():
    # 取得前端傳過來的數值
    insertValues = request.get_json()
    x1=insertValues['typeTitle']
    x2=insertValues['note']
    input1 = dataprocess(str(x1))
    input2 = dataprocess(str(x2))
    
    text = str(input1+' '+input2)      
    print(text)
    try:
        result = predict.prediction_model(text)
    except RuntimeError:
        result = "there are some invaild strings"
    except IndexError:
        result = "there is an index error"    
    return jsonify({'returnMessage': str(result)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
