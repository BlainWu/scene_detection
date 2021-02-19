from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
from tensorflow.lite.python import interpreter as interpreter_wrapper

from data_reader import load_image, img_preprocess
from utils import get_FileCreateTime
from constant import IMAGE_SIZE,NORMALIZATION
from tqdm import tqdm

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    val_path = '/data2/competition/classification/test/'
    model_file = "/home/wupeilin/project/scene_detection/MobileV3-94.83/model.tflite"

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


    res = {}
    with open(os.path.join(os.path.dirname(model_file),'result.txt'),"w+") as file:
        for dirpath, dirnames, filenames in tqdm(os.walk(val_path)):
            filenames.sort(key=lambda x: int(x.split('.')[0]))
            for imgname in filenames:
                image = load_image(os.path.join(dirpath, imgname))
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
                output = prediction_top_3[0]+1
                file.writelines(str(output) + '\n')
            # print(os.path.join(dirpath, imgname), "Pred", prediction_top_3[0])
# with open("save/" + train_name + "/tflite_test_log.txt", 'w') as f:
#     f.write("Top 1 Acc "+str(top1_correct / total))
#     f.write('\n')
#     f.write("Top 1 Wrong "+str(sorted(top1_wrong_label.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)))
#     f.write('\n')
#     f.write("Top 3 Acc "+str(top3_correct / total))
#     f.write('\n')
#     f.write("Top 3 Wrong "+str(sorted(top3_wrong_label.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)))
#     f.write('\n')
# print("Top 1 Acc ", top1_correct / total)
# print("Top 1 Wrong ", sorted(top1_wrong_label.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
# print("Top 3 Acc ", top3_correct / total)
# print("Top 3 Wrong ", sorted(top3_wrong_label.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
