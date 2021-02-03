import os
import tensorflow as tf
from DataReader import get_train_data,get_val_data

"""choose your GPU"""
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

"""load data"""
train_ds = get_train_data()
val_ds = get_val_data()

"""strategy"""
strategy = tf.distribute.MirroredStrategy()

"""initialize the model"""
checkpoint_dir = './training_checkpoints'
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

with strategy.scope():
    metrics = [tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1,name = 'TOP1'),
               tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='TOP3'),
    ]
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model = tf.keras.applications.MobileNetV2(
        input_shape=(384, 576, 3), alpha=1.4,weights=None, classes=30,
        classifier_activation='softmax')
    model.compile(optimizer=optimizer, loss=loss_object, metrics=metrics)

"""train"""
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
model.fit(train_ds, epochs=300, shuffle=False,validation_data=val_ds,
          validation_freq=2,validation_steps=3,
          callbacks=[tf.keras.callbacks.TensorBoard(log_dir='./logs'),
                     tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                                        save_weights_only=True)],
          verbose=1)

