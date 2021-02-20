from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import os
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from config import *
from data_reader import get_train_data, get_val_data
from utils import draw_train_history,mkdir
from shutil import copyfile


'''General Setting'''
os.environ['CUDA_VISIBLE_DEVICES'] = GPU

train_name = \
    "t" + str(time.strftime("%Y_%m_%d_%H_%M", time.localtime())) + \
    "_mb2_" + str(IMAGE_SIZE) + \
    "_t" + str(EPOCHS_TRAIN) + \
    "_f" + str(EPOCHS_FINE) + \
    "_n" + NORMALIZATION + \
    "_b" + str(BATCH_SIZE) + \
    "_fs" + str(FINE_TUNE_START) + \
    "_addL2Norm"
save_path = "save/" + train_name
train_log_path = os.path.join(save_path, "log_train")
fine_log_path = os.path.join(save_path, "log_fine")
print(train_name)
mkdir(save_path)
mkdir(train_log_path)
mkdir(fine_log_path)
copyfile("config.py", os.path.join(save_path, "config.py"))

'''Build Model'''

strategy = tf.distribute.MirroredStrategy()
losses = tf.losses.CategoricalCrossentropy(label_smoothing=0.05)
optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01)

with strategy.scope():
    '''
    base_model = tf.keras.applications.MobileNetV3Large(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   alpha=1.0,
                                                   weights='imagenet',
                                                    )
    '''
    base_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(192, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(30, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001))
    ])
    base_model.build(input_shape=(384,384))

    #base_model.trainable = True
    head_model = tf.keras.Sequential([
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(192, 3, activation='relu'),
        #tf.keras.layers.Dropout(0.5),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(30, activation='softmax',kernel_regularizer=tf.keras.regularizers.l2(0.001))
        #tf.keras.layers.Dense(30, activation='softmax')
    ])
    '''量化！'''

    quantize_model = tfmot.quantization.keras.quantize_model
    q_base_model = quantize_model(base_model)
    q_head_model = quantize_model(head_model)
    inputs = tf.keras.Input(shape=(384,384),batch_size=32)
    mid_output = q_base_model(inputs)
    outputs = q_head_model(mid_output)

    full_model = tf.keras.Model(inputs,outputs)
    full_model.compile(optimizer=optimizer,
                  loss=losses,
                  metrics=['accuracy'])

    full_model.summary()

'''Get Data
train_generator = get_train_data()
val_generator = get_val_data()
'''
'''Part1 Train classification net
history = full_model.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=EPOCHS_FINE,
                    validation_data=val_generator,
                    validation_freq=5,
                    callbacks=[
                        tf.keras.callbacks.ModelCheckpoint(
                            filepath=train_log_path,
                            monitor= 'val_accuracy',
                            save_best_only=1,
                            verbose=1)]
                    )

draw_train_history(history)'''