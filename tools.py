from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os
from tensorflow.lite.python import interpreter as interpreter_wrapper
from data_reader import load_image, img_preprocess
from utils import get_FileCreateTime
from config import IMAGE_SIZE,NORMALIZATION
from tqdm import tqdm

val_path = '/data2/competition/classification/val_true/'
test_path = '/data2/competition/classification/test/'
model_dir = '/home/wupeilin/project/scene_detection/96.83'
model_file = os.path.join(model_dir,'tflite_model.tflite')
os.environ['CUDA_VISIBLE_DEVICES'] = "4"


''' Convert Pb to TFLite'''
def pb2tflite_common():
    converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)

    tflite_model = converter.convert()

    with open(model_file, 'wb') as f:
        f.write(tflite_model)
def pb2tflite_aware():
    converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    quantized_tflite_model = converter.convert()
    with open(model_file, 'wb') as f:
        f.write(quantized_tflite_model)

'''生成可提交的txt文件* 手动删除最后一行空行'''
def txt_result():
    print("Model Name:", model_file, "Create Time:", get_FileCreateTime(model_file))

    interpreter = interpreter_wrapper.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    floating_model = False

    if input_details[0]['dtype'] == type(np.float32(1.0)):
        floating_model = True

    # Get the size of the input / output tensors

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Process test_model images and display the results
    txt_path = os.path.join(os.path.dirname(model_file),'results.txt')
    with open(txt_path,"w+") as file:
        img_list = os.listdir(test_path)
        img_list.sort(key=lambda x: int(x.split('.')[0]))
        for index,img in enumerate(tqdm(img_list)):
            image = load_image(os.path.join(test_path, img))
            image = img_preprocess(image, IMAGE_SIZE, NORMALIZATION, False)
            image = np.reshape(image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))
            input_data = image

            if floating_model:
                input_data = np.float32(input_data)
            else:
                input_data = np.uint8(input_data)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            results = np.squeeze(output_data)

            prediction = np.argmax(results)
            prediction_top_3 = results.argsort()[-3:][::-1]
            output = prediction_top_3[0] + 1
            if index != (len(img_list)-1) :
                file.writelines(str(output) + '\n')
            else:
                file.writelines(str(output))

def find_wrong_pics():
    interpreter = interpreter_wrapper.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    floating_model = False

    if input_details[0]['dtype'] == type(np.float32(1.0)):
        floating_model = True

    # Get the size of the input / output tensors

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Process test_model images and display the results

    cls_list = os.listdir(val_path)
    cls_list.sort(key=lambda x: int(x.split('_')[0]))

    wrong_count = 0
    all_count = 0

    for index, cls in enumerate(cls_list):
        cls_path = os.path.join(val_path, cls)
        img_list = os.listdir(cls_path)
        for img in img_list:
            #start_time = time.time()

            image = load_image(os.path.join(cls_path, img))
            image = img_preprocess(image, IMAGE_SIZE, NORMALIZATION, False)
            image = np.reshape(image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))
            input_data = image

            if floating_model:
                input_data = np.float32(input_data)
            else:
                input_data = np.uint8(input_data)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            results = np.squeeze(output_data)

            prediction = np.argmax(results)
            prediction_top_3 = results.argsort()[-3:][::-1]
            output = prediction_top_3[0]
            if int(output) != index:
                wrong_count += 1
                print(f"图片{img}预测结果为{cls_list[int(output)]},数据集标签为{cls_list[int(index)]}")
            all_count += 1
            #print(f'预测时间为{time.time() - start_time}')

    print(f"一共检测{str(all_count)}张图片，错误了{str(wrong_count)}张。\n正确率为{str(1 - wrong_count / all_count)}")

if __name__ == '__main__':
    #pb2tflite_common()
    print("已转成tflite模型")
    #txt_result()
    find_wrong_pics()
    pass