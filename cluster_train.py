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

'''LOSS AND OPTIMIZER!!!'''
losses = tf.losses.CategoricalCrossentropy(label_smoothing=0.05)
optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01)

strategy = tf.distribute.MirroredStrategy()
cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
clustering_params = {
  'number_of_clusters': 16,
  'cluster_centroids_init': CentroidInitialization.LINEAR
}
with strategy.scope():
    backbone = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   alpha=1.0,
                                                   weights='imagenet'
                                                   )

    backbone.trainable = True
    c_backbone = cluster_weights(backbone, **clustering_params)
    c_backbone.summary()

    #original_model.summary()

    #Now comes the proposed workaround
    model_top = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(7,7,1280)), #1280 ,1792
        tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Flatten()),
        tfmot.quantization.keras.quantize_annotate_layer(
            tf.keras.layers.Dense(30, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001))) #
        ])


    q_model_top = tfmot.quantization.keras.quantize_apply(model_top)
    q_model_top.summary()

    inputs = tf.keras.layers.Input(shape=IMG_SHAPE)
    x = q_backbone(inputs)
    outputs = q_model_top(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.summary()

#Get Data
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
                            monitor= 'val_accuracy',
                            save_best_only=1,
                            verbose=1)]
                    )



for layer in model.layers[:FINE_TUNE_START]:
    layer.trainable = True

