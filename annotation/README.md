## ˵��
�˴���Ϊ���б�ע������ʵ��ʶ�𣩵Ĳο����룬ʵ�ֵ�ģ��Ϊ˫��LSTM��ʹ�õ�����Ϊʵ����ҵ�����������ݡ�û��ʹ��ѵ���õĴ������Լ�CRFģ�ͣ���ͬѧ������ʵ�֡�

## ����ṹ:
    .
    ������ data                         
    ��   ������ train_corpus.txt              
    ��   ������ train_label.txt          
    ��   ������ test_corpus.txt         
    ��   ������ test_label.txt  
    ������ train.py      # ģ��ѵ���������Լ�����
    ������ model.py      # BiLSTMģ��ʵ��
    ������ utils.py      # �����࣬�����ʱ������Լ�ѵ���������ļ�

## ʹ��:
ѵ��ģ��
```
python train.py
```
����ģ�������train.py�е�val����