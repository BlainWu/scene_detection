import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from config import current_model_dir
import numpy as np
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = ' '

def show():
    pb_path = current_model_dir
    print(pb_path)
    #model = tf.keras.models.load_model(pb_path)
    model = tf.keras.applications.MobileNetV3Large()
    max,min = -10000,10000
    for layer in model.variables:
        if 'BatchNorm' in layer.name:
            continue
        layers = layer.numpy()
        # if np.max(layers) == 233128.28125:
        #     print(layer)
        # if np.min(layers) == -478.7233581542969:
        #     print(layer)
        min = min if min < np.min(layers) else np.min(layers)
        max = max if max > np.max(layers) else np.max(layers)
    print(f'模型参数最大值为：{max},最小值为{min}')

    for layer in model.variables:
        if 'BatchNorm' in layer.name:
            pass
        layers = layer.numpy()
        # if np.max(layers) == 52.83695602416992:
        #     print(layer)
        # if np.min(layers) == -43.029014587402344:
        #     print(layer)
        min = min if min < np.min(layers) else np.min(layers)
        max = max if max > np.max(layers) else np.max(layers)
    print(f'模型参数最大值为：{max},最小值为{min}')


if __name__ == '__main__':
    show()