from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
from tensorflow.lite.python import interpreter as interpreter_wrapper
from data_reader import load_image
from utils import get_FileCreateTime
from config import IMAGE_SIZE,NORMALIZATION
from tqdm import tqdm
from config import val_data_path,current_model_dir,GPU,test_data_path

val_path = val_data_path
test_path = test_data_path
model_dir = current_model_dir
model_file = os.path.join(model_dir,'tflite_model.tflite')

os.environ['CUDA_VISIBLE_DEVICES'] = GPU


def representative_dataset_object():
    image_size = 300
    data_path = "/home/wupeilin/project/scene_detection/represent_object/"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到文件夹：{data_path}")

    for dirpath, dirnames, filenames in os.walk(data_path):
        filenames.sort(key=lambda x:int(x.split('.')[0]))
        for imgname in filenames:
            image = tf.io.read_file(os.path.join(data_path, imgname))
            image = tf.compat.v1.image.decode_jpeg(image,channels=3)
            image = tf.image.resize(image, [image_size, image_size])
            image = np.array(image)
            print(os.path.join(data_path, imgname))

            image = np.reshape(image, (1, image_size, image_size, 3))
        yield [image.astype(np.float32)]

def pb2tflite_object(type = 'None'):
    '''
    ## TFLite Conversion
    # Before conversion, fix the model input size
    model = tf.saved_model.load(model_dir)
    model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs[0].set_shape([1, 640, 640, 3])
    tf.saved_model.save(model, "./saved_model_updated",
                        signatures=model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY])
    # Convert
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir='./saved_model_updated',
                                                         signature_keys=['serving_default'])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    '''
    converter = tf.lite.TFLiteConverter.from_saved_model(model_dir,signature_keys=['serving_default'])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if type == 'None':
        pass
    elif type == 'float16':
        converter.target_spec.supported_types = [tf.float16]
        model_file = os.path.join(model_dir,'tflite_model_float16.tflite')
    elif type == 'int8':
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.representative_dataset = representative_dataset_object
        model_file = os.path.join(model_dir, 'tflite_model_int8.tflite')
    converter.experimental_new_converter = True
    tflite_model = converter.convert()
    with open(model_file, 'wb') as f:
        f.write(tflite_model)

''' Convert Pb to TFLite'''
def pb2tflite_common():
    converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
    tflite_model = converter.convert()
    with open(model_file, 'wb') as f:
        f.write(tflite_model)

def pb2tflite_aware():
    converter = tf.lite.TFLiteConverter.from_saved_model(model_dir,signature_keys=['serving_default'])
    quantized_tflite_model = converter.convert()
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True

    with open(model_file, 'wb') as f:
        f.write(quantized_tflite_model)

'''生成可提交的txt文件'''
def generate_result():
    print("Model Name:", model_file, "Create Time:", get_FileCreateTime(model_file))

    interpreter = interpreter_wrapper.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    floating_model = False

    if input_details[0]['dtype'] == type(np.float32(1.0)):
        floating_model = True

    # Get the size of the input / output tensors

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

def evaluate_acc(model_file,output_wrong = False,image_size = IMAGE_SIZE):
    print('------------------------------------')
    print(f"加载模型为{model_file}")
    interpreter = interpreter_wrapper.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    floating_model = False

    if input_details[0]['dtype'] == type(np.float32(1.0)):
        floating_model = True
    # Process test_model images and display the results
    cls_list = os.listdir(val_path)
    cls_list.sort(key=lambda x: int(x.split('_')[0]))

    top1_wrong_count = 0
    top3_wrong_count = 0
    all_count = 0

    for index, cls in enumerate(tqdm(cls_list) ):
        cls_path = os.path.join(val_path, cls)
        img_list = os.listdir(cls_path)
        for img in img_list:

            image = load_image(os.path.join(cls_path, img))
            image = img_preprocess(image, image_size, NORMALIZATION, False)
            image = np.reshape(image, (1, image_size, image_size, 3))
            input_data = image

            if floating_model:
                input_data = np.float32(input_data)
            else:
                input_data = np.uint8(input_data)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            results = np.squeeze(output_data)

            prediction_top_3 = results.argsort()[-3:][::-1]
            prediction_top_3_list = [prediction_top_3[1],prediction_top_3[2]]
            output = prediction_top_3[0]
            if int(output) != index:
                top1_wrong_count += 1
                if output_wrong:
                    print(f"图片{img}预测结果为{cls_list[int(output)]},数据集标签为{cls_list[int(index)]}")
                if index not in prediction_top_3_list:
                    top3_wrong_count += 1
            all_count += 1

    print(f"Top1错误了{str(top1_wrong_count)}张。\nTo1 正确率为{str(1 - top1_wrong_count / all_count)}")
    print(f"Top3错误了{str(top3_wrong_count)}张。\nTo3 正确率为{str(1 - top3_wrong_count / all_count)}")
    print('------------------------------------')

'''===================================Writen by Ge========================================================='''

from data_reader import img_preprocess

def representative_dataset():
    global image_size
    data_path = "/home/share/competition/classification/represent_data_half/"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到文件夹：{data_path}")

    for dirpath, dirnames, filenames in os.walk(data_path):
        filenames.sort(key=lambda x:int(x.split('.')[0]))
        for imgname in tqdm(filenames):
            image = tf.io.read_file(os.path.join(data_path, imgname))
            image = tf.compat.v1.image.decode_jpeg(image,channels=3)
            image = img_preprocess(image, image_size, "per", False)
            image = np.array(image)

            image = np.reshape(image, (1, image_size, image_size, 3))
        yield [image.astype(np.float32)]


def convert_from_save_model(model_save_path, _image_size=224, type="normal"):
    global image_size
    image_size = _image_size
    converter = tf.lite.TFLiteConverter.from_saved_model(model_save_path)

    assert type in ['normal','float16','int8',"only-opt"]
    if type == "normal":
        tflite_save_path = os.path.join(model_save_path, 'tflite_model.tflite')
    elif type == "only-opt":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_save_path = os.path.join(model_save_path, 'tflite_model_opt.tflite')
    elif type == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_save_path = os.path.join(model_save_path, 'tflite_model_float16.tflite')
    elif type == "int8":
        #converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.representative_dataset = representative_dataset
        # Ensure that if any ops can't be quantized, the converter throws an error
        # Set the input and output tensors to uint8 (APIs added in r2.3)
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        tflite_save_path = os.path.join(model_save_path, 'tflite_model_int8.tflite')

    tflite_model = converter.convert()
    if tflite_save_path != None:
        with open(tflite_save_path, 'wb') as f:
            f.write(tflite_model)
            print(f"已生成模型：{tflite_save_path}")

def convert_test_all_in_one(_image_size = 224,convert = True,eval = True):
    if convert:
        types = ['normal','float16','int8','only-opt']
        for type in types:
            convert_from_save_model(model_dir,_image_size,type = type)

    if eval:
        models = ['tflite_model','tflite_model_float16','tflite_model_int8','tflite_model_opt']
        for model in models:
            file_path = os.path.join(model_dir,f'{model}.tflite')
            evaluate_acc(file_path,image_size = _image_size)


'''===================================Writen by Ge========================================================='''

from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph
def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops


if __name__ == '__main__':
    #pb2tflite_aware()
    #pb2tflite_common()
    #pb2tflite_object('None')
    #pb2tflite_object('float16')
    #pb2tflite_object('int8')
    #detect_objects(model_file)
    convert_test_all_in_one(_image_size=264,convert = True,eval = True)
    print("已转成tflite模型")
    #generate_result()
    #find_wrong_pics(model_file)
    #convert_from_save_model(model_dir,_image_size=384,type = "full_int")
    #find_wrong_pics(model_file = os.path.join(model_dir,'full_int_model.tflite')) #tflite_model  saved_model_float16
    #find_wrong_pics(model_file = "/home/wupeilin/project/scene_detection/quanti_ware_test/tflite_model.tflite")
    pass