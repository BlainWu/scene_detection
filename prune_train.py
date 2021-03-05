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
import tensorflow_model_optimization as tfmot
import numpy as np

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
pruning_logdir = os.path.join(save_path,'pruning_log')

print(train_name)

mkdir(save_path)
mkdir(pruning_logdir)
mkdir(train_log_path)
mkdir(fine_log_path)
copyfile("config.py", os.path.join(save_path, "config.py"))

'''Build Model'''

'''LOSS AND OPTIMIZER!!!'''
losses = tf.losses.CategoricalCrossentropy(label_smoothing=0.05)
optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01)

from data_reader import shuffle_size
end_step = np.ceil(shuffle_size / BATCH_SIZE).astype(np.int32) * EPOCHS_FINE
#strategy = tf.distribute.MirroredStrategy()
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}

def setup_model_prune():

    backbone = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=IMG_SHAPE,
                                                              include_top=False,
                                                              alpha=1.4,
                                                              weights= 'imagenet' ,
                                                              #weights='imagenet'
                                                              )

    backbone.trainable = True
    backbone_pruning = backbone

    #backbone_pruning = prune_low_magnitude(backbone, **pruning_params)
    backbone_pruning.summary()

    # original_model.summary()

    # Now comes the proposed workaround
    model_top = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(7, 7, 1792)),  # 1280 ,1792
        tf.keras.layers.Conv2D(192, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        # tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Flatten()),
        tf.keras.layers.Dense(30, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001)) #
    ])


    model_top_pruning = prune_low_magnitude(model_top, **pruning_params)
    model_top_pruning.summary()


    '''
    model_top = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(7, 7, 1792)),  # 1280 ,1792
        prune_low_magnitude(tf.keras.layers.Conv2D(192, 3, activation='relu'), **pruning_params),
        tf.keras.layers.GlobalAveragePooling2D(),
        # tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Flatten()),
        prune_low_magnitude(tf.keras.layers.Dense(30, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001)), **pruning_params) #
    ])

    model_top_pruning = model_top
    model_top_pruning.summary()
    '''

    inputs = tf.keras.layers.Input(shape=IMG_SHAPE)
    x = backbone_pruning(inputs)
    outputs = model_top_pruning(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.summary()
    return model

#with strategy.scope():
#    model = setup_model_prune()
model = setup_model_prune()
# Get Data
train_generator = get_train_data()
val_generator = get_val_data()

model.compile(
    loss=losses,
    optimizer=optimizer,
    metrics=['accuracy'])

history = model.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=EPOCHS_FINE,
                    validation_data=val_generator,
                    validation_freq=3,
                    callbacks=[
                            tf.keras.callbacks.ModelCheckpoint(
                            filepath=train_log_path,
                            monitor='val_accuracy',
                            save_best_only=1,
                            verbose=1),
                            tfmot.sparsity.keras.UpdatePruningStep(),
                            tfmot.sparsity.keras.PruningSummaries(log_dir=pruning_logdir),
                            ]
                    )

for layer in model.layers[:FINE_TUNE_START]:
    layer.trainable = True

converter = tf.lite.TFLiteConverter.from_keras_model(model, signature_keys=['serving_default'])
quantized_tflite_model = converter.convert()
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
from tools import model_file
quantized_tflite_model = converter.convert()
with open(model_file, 'wb') as f:
    f.write(quantized_tflite_model)