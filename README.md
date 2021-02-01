Real-time Scene Detection
==================
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)  

## 目录树
├─dataset  
│  ├─train  
│  │  ├─1_Portrait  
│  │  ├─2_Group_portrait  
│  │  ├─3_Kids  
│  │  ├─...  
│  └─val  
│  │  ├─1_Portrait  
│  │  ├─2_Group_portrait  
│  │  ├─3_Kids   
│  │  ├─...  
├─DataReader.py  
├─train.py  

## 图像增强和处理
文件: **DataReader.py**  
功能：加载训练集和验证集，并对其进行尺寸变化，打包成批。尤其对训练集进行数据增强，功能分布如下。  
![image](https://github.com/BlainWu/scene_detection/blob/master/dataset/readme_src/%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86%E5%8A%9F%E8%83%BD.png)  

问题： 每张图片增强时的随机种子seed直接（counter,counter）是否合适？需不需要再在每个方法之间使用：  
> new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]  

## 训练
文件:train.py  

**加载数据集**：
```python
train_ds = get_train_data()
val_ds = get_val_data()
```
**数据格式**：  
（图片，标签序号）的tuple，图片尺寸在DataReader.py的超参中可调。
```
<PrefetchDataset shapes: ((None, 576, 576, 3), (None,)), types: (tf.float32, tf.int64)>
```

## 保存模型