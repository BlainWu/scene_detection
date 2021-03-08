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

from tools import get_flops


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

with strategy.scope():

    base_model = tf.keras.applications.MobileNetV3Large(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   alpha=1.0,
                                                   weights='imagenet',
                                                    )
    base_model.trainable = True
    model = tf.keras.Sequential([
        base_model,
        # tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.Conv2D(192, 3, activation='relu'),
        #tf.keras.layers.Dropout(0.5),
        tf.keras.layers.DepthwiseConv2D(5, activation='relu'),
        tf.keras.layers.Conv2D(192, 1, activation="relu"),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(30, activation='softmax',kernel_regularizer=tf.keras.regularizers.l2(0.001))
        #tf.keras.layers.Dense(30, activation='softmax')
    ])
    '''LOSS AND OPTIMIZER!!!'''
    model.summary()
    print("FLOPs:{}M".format(get_flops(model)) ,flush=True)
    losses = tf.losses.CategoricalCrossentropy() #label_smoothing=0.05
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01)

    model.compile(optimizer=optimizer,
                  # loss='categorical_crossentropy',
                  loss=losses,
                  metrics=['accuracy'])

'''Get Data'''
train_generator = get_train_data()
val_generator = get_val_data()

'''Part1 Train classification net'''
history = model.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=EPOCHS_TRAIN,
                    validation_data=val_generator,
                    validation_freq=5,
                    callbacks=[
                        tf.keras.callbacks.ModelCheckpoint(
                            filepath=train_log_path,
                            monitor= 'val_accuracy',
                            save_best_only=1,
                            verbose=1)]
                    )
draw_train_history(history)

'''Part2 fine tuning'''

'''Frozen forward layer'''
base_model.trainable = True

for layer in base_model.layers[:FINE_TUNE_START]:
    layer.trainable = True

model.compile(
    loss=losses,
    optimizer=optimizer,
    metrics=['accuracy'])
model.summary()
'''Training'''
history_fine = model.fit(train_generator,
                         steps_per_epoch=len(train_generator),
                         epochs=EPOCHS_FINE,
                         validation_data=val_generator,
                         validation_freq=3,
                         callbacks=[
                             tf.keras.callbacks.ModelCheckpoint(
                                 filepath=train_log_path,
                                 monitor= 'val_accuracy',
                                 save_best_only=1,
                                 verbose=1)])
model.layers[2].rate = 0
print("set dropout = 0")
results = model.evaluate(val_generator)
print("test loss, test acc:", results)



'''

tf.saved_model.save(model, save_path)
converter = tf.lite.TFLiteConverter.from_saved_model(save_path)
tflite_model = converter.convert()

with open(save_path + '/' + train_name + '.tflite', 'wb') as f:
    f.write(tflite_model)
draw_train_history(history)

save_wrong = True

CLASSES = ['Portrait', 'Group Portrait', 'Kids / Infants', 'Dog', 'Cat', 'Macro / Close-up', 'Food / Gourmet',
           'Beach', 'Mountains', 'Waterfall', 'Snow', 'Landscape', 'Underwater', 'Architecture', 'Sunrise / Sunset',
           'Blue Sky', 'Overcast / Cloudy Sky', 'Greenery / Green Plants / Grass', 'Autumn Plants', 'Flower',
           'Night Shot', 'Stage / Concert', 'Fireworks', 'Candle light', 'Neon Lights / Neon Signs', 'Indoor',
           'Backlight / Contre-jour', 'Text / Document', 'QR Code', 'Monitor Screen']
for i in range(len(CLASSES)):
    CLASSES[i] = "".join("".join(CLASSES[i].split()).split('/'))

model_file = "save/" + train_name + "/" + train_name + ".tflite"
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

# Process test images and display the results

val_path = 'data/val_new/'
total = 0
top1_correct = 0
top3_correct = 0
top1_error = 0
top3_error = 0

correct = {}
total_dict = {}
predict_dict = {}

top1_wrong_label = {}
top3_wrong_label = {}
if save_wrong:
    if not os.path.exists("save/" + train_name + "/wrong/"):
        os.mkdir("save/" + train_name + "/wrong")
for dirpath, dirnames, filenames in os.walk(val_path):
    for dirname in dirnames:
        for path, _, imgnames in os.walk(os.path.join(dirpath, dirname)):
            for imgname in imgnames:
                label = int(dirname.split('_')[0]) - 1
                total += 1
                image = load_image(os.path.join(path, imgname))
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
                if CLASSES[label] not in total_dict:
                    total_dict[CLASSES[label]] = 1
                    correct[CLASSES[label]] = 0

                else:
                    total_dict[CLASSES[label]] += 1

                if CLASSES[prediction_top_3[0]] not in predict_dict:
                    predict_dict[CLASSES[prediction_top_3[0]]] = 1
                else:
                    predict_dict[CLASSES[prediction_top_3[0]]] += 1

                if prediction_top_3[0] == label:
                    top1_correct += 1
                    correct[CLASSES[label]] += 1
                else:
                    if save_wrong:
                        if not os.path.exists("save/" + train_name + "/wrong/" + CLASSES[label]):
                            os.mkdir("save/" + train_name + "/wrong/" + CLASSES[label])
                        copyfile(os.path.join(path, imgname),
                                 os.path.join("save/" + train_name + "/wrong/" + CLASSES[label],
                                              CLASSES[prediction_top_3[0]] + "_" +
                                              CLASSES[prediction_top_3[1]] + "_" +
                                              CLASSES[prediction_top_3[2]] + "_" + ".jpg"))

                    if dirname in top1_wrong_label:
                        top1_wrong_label[dirname] += 1
                    else:
                        top1_wrong_label[dirname] = 1
                    top1_error += 1
                if label in prediction_top_3:
                    top3_correct += 1
                else:
                    top3_error += 1
                    if dirname in top3_wrong_label:
                        top3_wrong_label[dirname] += 1
                    else:
                        top3_wrong_label[dirname] = 1

                print(os.path.join(path, imgname), "True", label, "Pred", prediction_top_3)
accuracy = {}
recall = {}
for k in correct.keys():
    accuracy[k] = correct[k] / total_dict[k]
    recall[k] = correct[k] / (predict_dict[k])
with open("save/" + train_name + "/tflite_test_log.txt", 'w') as f:
    f.write("Top 1 Acc " + str(top1_correct / total))
    f.write('\n')
    f.write("Top 1 Wrong " + str(sorted(top1_wrong_label.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)))
    f.write('\n')
    f.write("Top 3 Acc " + str(top3_correct / total))
    f.write('\n')
    f.write("Top 3 Wrong " + str(sorted(top3_wrong_label.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)))
    f.write('\n')
    f.write("Accuracy in category:" + str(accuracy))
    f.write('\n')
    f.write("Recall in category:" + str(recall))
    f.write('\n')
print("Top 1 Acc ", top1_correct / total)
print("Top 1 Wrong ", sorted(top1_wrong_label.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
print("Top 3 Acc ", top3_correct / total)
print("Top 3 Wrong ", sorted(top3_wrong_label.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
print("Accuracy in category:", accuracy)
print("Recall in category:", recall)
'''