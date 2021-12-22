import os

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer

# 分類類別
# no 不能遠端處理-->送臨櫃
# yes 可遠端處理
key = {0: 'no',
       1: 'yes'
       }


class Config:
    """Bert配置參數"""

    def __init__(self):
        cru = os.path.dirname(__file__)
        self.class_list = [str(i) for i in range(len(key))] 
        self.save_path = os.path.join(cru, 'Dataser/model/bert.ckpt')
        self.device = torch.device('cpu')
        self.require_improvement = 1000  
        self.num_classes = len(self.class_list)  # 分類類別數量
        self.num_epochs = 3  # epoch数
        self.batch_size = 64  # mini-batch大小
        self.pad_size = 22  # 每句話判讀的長度(短填長切)
        self.learning_rate = 5e-5  # 學習率
        self.bert_path = os.path.join(cru, 'bert_pretrain')
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768

    def build_dataset(self, text):
        lin = text.strip()
        pad_size = len(lin)
        token = self.tokenizer.tokenize(lin)
        token = ['[CLS]'] + token
        token_ids = self.tokenizer.convert_tokens_to_ids(token)
        mask = [1] * pad_size
        token_ids = token_ids[:pad_size]
        return torch.tensor([token_ids], dtype=torch.long), torch.tensor([mask])


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[1]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out


config = Config()
model = Model(config).to(config.device)
model.load_state_dict(torch.load(config.save_path, map_location='cpu'))

#供引用的預測功能
def prediction_model(text):
    data = config.build_dataset(text)
    with torch.no_grad():
        outputs = model(data)
        num = torch.argmax(outputs)
    return key[int(num)]

wordinput = input('please type sentense')
if __name__ == '__main__':
    print(prediction_model(wordinput))
