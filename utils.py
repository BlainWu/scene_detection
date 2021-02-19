import os
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import datetime


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def timestamp_to_time(timestamp):
    timeStruct = time.localtime(timestamp)
    return time.strftime('%Y-%m-%d_%H:%M:%S', timeStruct)


def get_FileCreateTime(filePath):
    t = os.path.getctime(filePath)
    return timestamp_to_time(t)


def resume_from_checkpoint(net):
    net.init_model()
    latest = tf.train.latest_checkpoint("checkpoints/" + net.name + "/")
    print(latest)
    net.model.load_weights(latest)
    return net.model


def save_model(model_name, model):
    saved_model_dir = 'saved_model/' + model_name

    if not os.path.exists(saved_model_dir):
        os.mkdir(saved_model_dir)

    tf.saved_model.save(model, saved_model_dir)


def draw_train_history(history, name=""):
    if name != "":
        name += ":"

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title(name + 'Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title(name + 'Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import numpy as np


class WarmupExponentialDecay(Callback):
    def __init__(self, lr_base=0.0002, lr_min=0.0, decay=0, warmup_epochs=0):
        self.num_passed_batchs = 0  # 一个计数器
        self.warmup_epochs = warmup_epochs
        self.lr = lr_base  # learning_rate_base
        self.lr_min = lr_min  # 最小的起始学习率,此代码尚未实现
        self.decay = decay  # 指数衰减率
        self.steps_per_epoch = 0  # 也是一个计数器

    def on_batch_begin(self, batch, logs=None):
        # params是模型自动传递给Callback的一些参数
        if self.steps_per_epoch == 0:
            # 防止跑验证集的时候呗更改了
            if self.params['steps'] == None:
                self.steps_per_epoch = np.ceil(1. * self.params['samples'] / self.params['batch_size'])
            else:
                self.steps_per_epoch = self.params['steps']
        if self.num_passed_batchs < self.steps_per_epoch * self.warmup_epochs:
            K.set_value(self.model.optimizer.lr,
                        self.lr * (self.num_passed_batchs + 1) / self.steps_per_epoch / self.warmup_epochs)
        else:
            K.set_value(self.model.optimizer.lr,
                        self.lr * ((1 - self.decay) ** (
                                    self.num_passed_batchs - self.steps_per_epoch * self.warmup_epochs)))
        self.num_passed_batchs += 1

    def on_epoch_begin(self, epoch, logs=None):
        # 用来输出学习率的,可以删除
        print("learning_rate:", K.get_value(self.model.optimizer.lr))
