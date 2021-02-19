import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

model_dir = '/home/wupeilin/project/scene_detection/MobileV3-94.83'
output = os.path.join(model_dir,'model.tflite')

'''Part3 Convert to TFLite'''

converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)

tflite_model = converter.convert()

with open(output, 'wb') as f:
    f.write(tflite_model)
