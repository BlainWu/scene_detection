import tensorflow_model_optimization as tfmot
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "4"
base_model = tf.keras.models.load_model('./96.83')
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model)

model_for_pruning.summary()