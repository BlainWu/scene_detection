# Real-time Scene Detection

# 图像增强和处理
文件: **DataReader.py**  
功能：加载训练集和验证集，并对其进行尺寸变化，打包成批。尤其对训练集进行数据增强，功能分布如下。  
![image](https://github.com/BlainWu/scene_detection/blob/master/dataset/readme_src/%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86%E5%8A%9F%E8%83%BD.png)  

问题： 每张图片增强时的随机种子seed直接（counter,counter）是否合适？需不需要再在每个方法之间使用：  
> new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]  

# 训练

# 保存模型