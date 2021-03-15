from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time

import tensorflow as tf
from config import *
from data_reader import get_train_data, get_val_data
from utils import draw_train_history, mkdir
from shutil import copyfile
import os

from tools import get_flops
os.environ['CUDA_VISIBLE_DEVICES'] = GPU

log_index = len(os.listdir('./save'))
train_name = \
    str(log_index)
save_path = "save/" + train_name
train_log_path = os.path.join(save_path, "log_train")
fine_log_path = os.path.join(save_path, "log_fine")
print("-----------------------------")
print("实验编号：",train_name)
print("-----------------------------")
mkdir(save_path)
mkdir(train_log_path)
mkdir(fine_log_path)
copyfile("config.py", os.path.join(save_path, "config.py"))

'''优化器和loss设置'''
losses = tf.losses.CategoricalCrossentropy(label_smoothing=0.05)
optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():

    base_model = tf.keras.applications.MobileNetV3Large(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   alpha=1.0,
                                                   weights='imagenet',
                                                    #dropout_rate=0.2,
                                                    #pooling='max',
                                                    )
    '''
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=IMG_SHAPE,
                                                              include_top=False,
                                                              alpha=1.0,
                                                              weights='imagenet'
                                                              )'''

    base_model.trainable = True
    model = tf.keras.Sequential([
        base_model,
        #tf.keras.layers.Conv2D(192, 3, activation='relu'),
        tf.keras.layers.DepthwiseConv2D(5, activation='relu'),
        #tf.keras.layers.Conv2D(192, 1),
        tf.keras.layers.Conv2D(192, 1, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(30, activation='softmax',kernel_regularizer=tf.keras.regularizers.l2(0.001))
        #tf.keras.layers.Dense(30, activation='softmax')
    ])
    model.summary()
    print("该模型的FLOPs:%.5fM"%(get_flops(model)/10**6) ,flush=True)



    model.compile(optimizer=optimizer,
                  # loss='categorical_crossentropy',
                  loss=losses,
                  metrics=['accuracy'])

'''Get Data'''
train_generator = get_train_data()
val_generator = get_val_data()

'''Part1 Train classification net'''
history = model.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=EPOCHS_TRAIN,
                    validation_data=val_generator,
                    validation_freq=5,
                    callbacks=[
                        tf.keras.callbacks.ModelCheckpoint(
                            filepath=train_log_path,
                            monitor= 'val_accuracy',
                            save_best_only=1,
                            verbose=1)]
                    )
draw_train_history(history)

'''Part2 fine tuning'''

'''Frozen forward layer'''
base_model.trainable = True

for layer in base_model.layers[:FINE_TUNE_START]:
    layer.trainable = True

model.compile(
    loss=losses,
    optimizer=optimizer,
    metrics=['accuracy'])
model.summary()
'''Training'''
history_fine = model.fit(train_generator,
                         steps_per_epoch=len(train_generator),
                         epochs=EPOCHS_FINE,
                         validation_data=val_generator,
                         validation_freq=3,
                         callbacks=[
                             tf.keras.callbacks.ModelCheckpoint(
                                 filepath=train_log_path,
                                 monitor= 'val_accuracy',
                                 save_best_only=1,
                                 verbose=1)])

draw_train_history(history_fine)