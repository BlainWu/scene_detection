Real-time Scene Detection
==================
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)  

## To-do-list
- [ ] 使用ReLabel策略
- [ ] 数据增强进一步扩充和整合

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

## ReLabel  
**背景**：对于ImageNet这些图像分类数据集，存在一张图片包含多类的问题。在随机裁剪的训练策略下，这种问题会导致对分类器的误导。  
**资料**：[ [arxiv论文](https://arxiv.org/pdf/2101.05022.pdf) ]------[ [github](https://github.com/naver-ai/relabel_imagenet) ]  

 

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


## 数据集说明  
| 序号  | 区分难度 | 名称             |  主体              | 边缘分类              |
| :---:| :---:   | -----           | ------------------ | ------------------    |
| 1    |  中等   | Portrait          | 单人成年人写真             |儿童写真（P153,124,32） | 
| 2    |  中等   | Group_portrait   | 多人照片                   |两个成年人+婴儿（P201,310）,三个儿童的合照（P121） | 
| 3    |  中等   | Kids               | 婴幼儿照片               |一个成年人侧脸抱着婴儿（P24,104）,儿童照片(P23,141)，P72看不清建议排除 | 
| 4    |  简单   | Dog               | 狗的照片                  | | 
| 5    |  简单   | Cat               | 猫的照片                 |P114猫在草丛中，占图片位置太小，随机剪切可能会切没 | 
| 6    |  中等   | Macro              | 微距图片                 | 有花的特写 |
| 7    |  简单   | Food              | 拍摄精美的食物             | | 
| 8    |  中等   | Beach          | 沙滩                 | 和蓝天有重合 |
| 9    |  中等   | Mountain          | 山                 | 和雪山、蓝天两类有重合 |
| 10    |  中等   | WaterFall          | 瀑布                 | 和秋叶类有重合 |
| 11   |  中等   | Snow             | 带雪的照片               | 雪山（P286,295）,天空占绝大多数雪地占小部分（P57,58） |
| 12   |  中等   | Landscape        | 风景照片，多为小山坡构图颜色美丽的图片   | P42蓝天 | 
| 13   |  简单  | Underwater      | 水下拍摄的照片                | 包括水下拍摄的人物照片（P293） |
| 14   |  中等  | Architecture      | 建筑照片              | 地面有雪的建筑（P22,129）,天半黑的建筑（P27） |
| 15   |  中等  | Sunset_Sunrise    | 日出日落照片              | 背光（P158,165,170,287）建筑P132 |
| 16   |  困难  | Blue Sky      | 蓝天照片，包括局部建筑物+蓝天 | 画面底部有山（P10）,背光（P12），像风景图（P23，24），主体部分是建筑局部（P25）天有云或者不蓝（P36，46，55）,图片中心有猫（P161）是人（P185）是花（P115）,地面有雪（P45） |
| 17   |  困难  | Cloudy_Sky    | 阴天照片                    | 背光（P1）山是主体（P7，14）,沙滩（P292） |
| 18   |  简单  | Greenery    |  绿色植物照片  | 主体是绿色，带有蓝天的（P34）,阴天草淡黄（P265） |  
| 19   |  困难  | Autumn_leaves  |  秋叶  | 只有地上有落叶，主体是树（P156）,溪流是主体（P185）,瀑布是主体（P187）,有山（P201,226） | 
| 21   |  中等  | Nigh_shot    | 夜间拍摄的照片，包括建筑、沙滩      |  |  
| 22   |  简单  | Stage_concert    | 舞台或者歌手表演照片                |  |
| 23   |  简单  | Fireworks      | 烟花照片                   | 抽象放射状的光条（P278） |
| 24   |  简单  | Candle_Lights      | 烛光照片                   |  |
| 25   |  简单  | Neon_Lights      | 彩色灯条，抽象光条图片 | 普通带白灯的广告牌（P50） |
| 26   |  中等  | Indoor      | 室内照片 | 没有人对着镜头的多人室内照片（P169,171,173）背身吃食的多只猫（P172），有电脑电视屏幕但是占比很小（P174,195） |
| 27   |  中等  | Backlight      | 背光拍摄的照片，任何有物体是黑色剪影的图片 | 带雪的图（P11,92）,日出日落（P22），水下拍摄（P68） |
| 28   |  简单  | Text_Documents      | 手写文本、印刷文本、电子书截图 | *不包括带二维码的图片 | 
| 29   |  简单   | QR_image          | 任何带二维码的图片（很多照片中二维码很小，不能随机剪切） |包括带二维码的屏幕、纸质文件 | 
| 30    |  简单  | Computer_Screens  | 电子屏幕，包括电脑、手机、空调终端 | *不包括带二维码的图片 | 




## 保存模型