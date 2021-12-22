# 單行文本分類(OA維修-可否遠端處理)
## 介紹
此為以Bert為演算法基底眼神的單行文本分類模型，適用於中文短文本分類  
**用途以資料類型分類為佳**。  
原理文引用google Bert(2018) pre-trained 中文文本辨識model，  
此範本為利用labeled的OA維修資料，將bert model微調，  
使模型可依據使用者備註內容以及所選分類標籤初步判斷是否適合遠端處理  
## 操作
### *事前準備*
**執行此模型需要下載google的中文文本預訓練模型**  
下載網址如下:  
`https://www.codenong.com/cs106711565/`  
分別下載三個檔案: 
* BERT的預訓練模型 (pytorch)
* Vocab檔
* BERT 預訓練的config JSON檔    

分別命名為：  
* bert_config
* pytorch_model
* vocab

存放在 /bert_pretrain資料夾  
### *訓練*
如果想要以新的分類資料建立新的分類模型，可以準備好分類類別、訓練集、驗證集以及測試集  
依照格式改為/Dataset/data中檔案的模式，然後放入/Dataset/data之後就能以下述指令trainnig  
    ```
    python run.py --model bert
    ```  
如果想要執行原本的OA維修判斷模型，可以直接執行上述指令  
或是下載先前train好的bert.ckpt檔放入/Dataset/save_dict  

### *預測*
單筆預測直接執行
    ```
    python predict.py  
    ```    
然後輸入要判斷的文字  
會印出預測結果
或是  
`python establish.py`可以直接建立一個預測API[Post]  
input是以下json:  
`
{
    "typeTitle":"維修選項",
    "note":"維修備註"
}
`  
回傳是:  
`
{
    "return": "答案"
}
`
### *參數調整*
訓練參數可以在/models/bert.py中調整  
可以調**batch_size**、**學習率**、**epoch數**、**每句話判斷長度**  
**每句話判斷長度**建議以**文本長度中位數**開始測試  
**學習率**可以以下列參數開始測試:  
1e-4、3e-4、3e-5、5e-5  
**batch_size**以*顯卡記憶體*來說  
8G的顯存可能不要超過64  
有12G的話可以試試128  
epoch數則是視訓練結果每次調整  
  
    
API port可以在*establish.py*中設定
