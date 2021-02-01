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

## ���ݼ�˵��  
| ���  | �����Ѷ� | ����             |  ����              | ��Ե����              |
| :---:| :---:   | -----           | ------------------ | ------------------    |
| 1    |  ??   | Portrait          | ���˳�����д��             |��ͯд�棨P153,124,32�� | 
| 2    |  ??   | Group_portrait   | ������Ƭ                   |����������+Ӥ����P201,310��,������ͯ�ĺ��գ�P121�� | 
| 3    |  ??   | Kids               | Ӥ�׶���Ƭ               |һ�������˲�������Ӥ����P24,104��,��ͯ��Ƭ(P23,141)��P72�����彨���ų� | 
| 4    |  ?   | Dog               | ������Ƭ                  | | 
| 5    |  ?   | Cat               | è����Ƭ                 |P114è�ڲݴ��У�ռͼƬλ��̫С��������п��ܻ���û | 
| 7    |  ?   | Food              | ���㾫����ʳ��             | | 
| 11   |  ??   | Snow             | ��ѩ����Ƭ               | ѩɽ��P286,295��,���ռ�������ѩ��ռС���֣�P57,58�� |  
| 13   |  ?  | Underwater      | ˮ���������Ƭ                | ����ˮ�������������Ƭ��P293�� |
| 23   |  ?  | Fireworks      | �̻���Ƭ                   | �������״�Ĺ�����P278�� |
| 24   |  ?  | Candle_Lights      | �����Ƭ                   |  |
| 25   |  ?  | Neon_Lights      | ��ɫ�������������ͼƬ | ��ͨ���׵ƵĹ���ƣ�P50�� |
| 27   |  ??  | Backlight      | �����������Ƭ���κ��������Ǻ�ɫ��Ӱ��ͼƬ | ��ѩ��ͼ��P11,92��,�ճ����䣨P22����ˮ�����㣨P68�� |
| 28   |  ?  | Text_Documents      | ��д�ı���ӡˢ�ı����������ͼ | *����������ά���ͼƬ | 
| 29   |  ?   | QR_image          | �κδ���ά���ͼƬ���ܶ���Ƭ�ж�ά���С������������У� |��������ά�����Ļ��ֽ���ļ� | 
| 30    |  ?  | Computer_Screens  | ������Ļ���������ԡ��ֻ����յ��ն� | *����������ά���ͼƬ | 




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