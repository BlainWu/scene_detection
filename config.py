IMAGE_SIZE = 264
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

GPU = "0,1"
BATCH_SIZE = 32
EPOCHS_TRAIN = 10
EPOCHS_FINE = 500
FINE_TUNE_START = 0
NORMALIZATION = 'None'

train_data_path = "/home/share/competition/classification/train_2_8/"
val_data_path = '/home/share/competition/classification/val_true'
test_data_path = '/home/share/competition/classification/test'

current_model_dir = \
    '/home/wupeilin/project/scene_detection/save/75/log_train' #ssd_mobilenet_v2   quanti_ware_test