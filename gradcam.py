
import random
import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input,
    decode_predictions,
)
import cv2

# Load full model
image = np.array(load_img("./dataset/origin.jpg", target_size=(224, 224, 3)))
#plt.imshow(image)
model = ResNet50()
#model = tf.keras.models.load_model('./models/96.83')
model.summary()

# Get the last layer, and make a model
last_conv_layer = model.get_layer("conv5_block3_out")#conv2d conv5_block3_out
last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
x = classifier_input
for layer_name in ["avg_pool", "predictions"]:
    x = model.get_layer(layer_name)(x)
classifier_model = tf.keras.Model(classifier_input, x)

with tf.GradientTape() as tape:
    inputs = image[np.newaxis, ...]
    last_conv_layer_output = last_conv_layer_model(inputs)
    tape.watch(last_conv_layer_output)
    preds = classifier_model(last_conv_layer_output)
    top_pred_index = tf.argmax(preds[0])
    top_class_channel = preds[:, top_pred_index]

grads = tape.gradient(top_class_channel, last_conv_layer_output)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

last_conv_layer_output = last_conv_layer_output.numpy()[0]
pooled_grads = pooled_grads.numpy()
for i in range(pooled_grads.shape[-1]):
    last_conv_layer_output[:, :, i] *= pooled_grads[i]

# Average over all the filters to get a single 2D array
gradcam = np.mean(last_conv_layer_output, axis=-1)
# Clip the values (equivalent to applying ReLU)
# and then normalise the values
gradcam = np.clip(gradcam, 0, np.max(gradcam)) / np.max(gradcam)
gradcam = cv2.resize(gradcam, (224, 224))

import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(2, 2)
plt.figure(dpi=100, figsize=(10, 10))
font_size = 26
'''----------------Original_image-----------------'''
#out_put = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.subplot(gs[0, 0])
plt.imshow(image)
plt.title('Input Image', fontsize=font_size)
plt.xticks([])
plt.yticks([])

#plt.imshow(image)
plt.subplot(gs[0,1])
plt.imshow(gradcam)
plt.xticks([])
plt.yticks([])
plt.title("Gradient-weighted CAM", fontsize=font_size)

plt.subplot(2,1,2)
plt.imshow(image)
plt.imshow(gradcam,alpha=0.5)
plt.xticks([])
plt.yticks([])
plt.title("Mixed Image", fontsize=font_size)

plt.savefig('./data/CAM.jpg')
plt.show()
