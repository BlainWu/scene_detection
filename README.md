Real-time Scene Detection
==================
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)  

## Ŀ¼��
����dataset  
��  ����train  
��  ��  ����1_Portrait  
��  ��  ����2_Group_portrait  
��  ��  ����3_Kids  
��  ��  ����...  
��  ����val  
��  ��  ����1_Portrait  
��  ��  ����2_Group_portrait  
��  ��  ����3_Kids   
��  ��  ����...  
����DataReader.py  
����train.py  

## ͼ����ǿ�ʹ���
�ļ�: **DataReader.py**  
���ܣ�����ѵ��������֤������������гߴ�仯����������������ѵ��������������ǿ�����ֲܷ����¡�  
![image](https://github.com/BlainWu/scene_detection/blob/master/dataset/readme_src/%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86%E5%8A%9F%E8%83%BD.png)  

���⣺ ÿ��ͼƬ��ǿʱ���������seedֱ�ӣ�counter,counter���Ƿ���ʣ��費��Ҫ����ÿ������֮��ʹ�ã�  
> new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]  

## ѵ��
�ļ�:train.py  

**�������ݼ�**��
```python
train_ds = get_train_data()
val_ds = get_val_data()
```
**���ݸ�ʽ**��  
��ͼƬ����ǩ��ţ���tuple��ͼƬ�ߴ���DataReader.py�ĳ����пɵ���
```
<PrefetchDataset shapes: ((None, 576, 576, 3), (None,)), types: (tf.float32, tf.int64)>
```

## ����ģ��