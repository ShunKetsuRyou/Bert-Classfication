# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'                                # 訓練集
        self.dev_path = dataset + '/data/dev.txt'                                    # 驗證集
        self.test_path = dataset + '/data/test.txt'                                  # 測試集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 分類類別
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 輸出訓練結果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 使用設備

        self.require_improvement = 1000                                 # 若1000batch後效果沒有提升則停止訓練
        self.num_classes = len(self.class_list)                         # 類別數
        self.num_epochs = 100                                             # epoch数
        self.batch_size = 64                                           # mini-batch大小(單次數據量)
        self.pad_size = 22                                              # 每句话統一的長度(短填长切)
        self.learning_rate = 3e-5                                       # 學習率
        self.bert_path = './bert_pretrain'                              # Pre-train Model 這裡用的是Google Bert 中文預訓練集
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768                                          # encoder layers and the pooler layer的維度

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 輸入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out
