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

## 数据集说明  
| 序号  | 区分难度 | 名称             |  主体              | 边缘分类              |
| :---:| :---:   | -----           | ------------------ | ------------------    |
| 1    |  ??   | Portrait          | 单人成年人写真             |儿童写真（P153,124,32） | 
| 2    |  ??   | Group_portrait   | 多人照片                   |两个成年人+婴儿（P201,310）,三个儿童的合照（P121） | 
| 3    |  ??   | Kids               | 婴幼儿照片               |一个成年人侧脸抱着婴儿（P24,104）,儿童照片(P23,141)，P72看不清建议排除 | 
| 4    |  ?   | Dog               | 狗的照片                  | | 
| 5    |  ?   | Cat               | 猫的照片                 |P114猫在草丛中，占图片位置太小，随机剪切可能会切没 | 
| 7    |  ?   | Food              | 拍摄精美的食物             | | 
| 11   |  ??   | Snow             | 带雪的照片               | 雪山（P286,295）,天空占绝大多数雪地占小部分（P57,58） |  
| 13   |  ?  | Underwater      | 水下拍摄的照片                | 包括水下拍摄的人物照片（P293） |
| 23   |  ?  | Fireworks      | 烟花照片                   | 抽象放射状的光条（P278） |
| 24   |  ?  | Candle_Lights      | 烛光照片                   |  |
| 25   |  ?  | Neon_Lights      | 彩色灯条，抽象光条图片 | 普通带白灯的广告牌（P50） |
| 27   |  ??  | Backlight      | 背光拍摄的照片，任何有物体是黑色剪影的图片 | 带雪的图（P11,92）,日出日落（P22），水下拍摄（P68） |
| 28   |  ?  | Text_Documents      | 手写文本、印刷文本、电子书截图 | *不包括带二维码的图片 | 
| 29   |  ?   | QR_image          | 任何带二维码的图片（很多照片中二维码很小，不能随机剪切） |包括带二维码的屏幕、纸质文件 | 
| 30    |  ?  | Computer_Screens  | 电子屏幕，包括电脑、手机、空调终端 | *不包括带二维码的图片 | 




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