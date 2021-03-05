IMAGE_SIZE = 224
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

GPU = "2"
BATCH_SIZE = 32
EPOCHS_TRAIN = 10
EPOCHS_FINE = 100
FINE_TUNE_START = 0
NORMALIZATION = "per"

train_data_path = '/home/share/competition/classification/train_2_8'
val_data_path = '/home/share/competition/classification/val_true'


current_model_dir = \
    '/home/wupeilin/project/scene_detection/depth_estimation' #ssd_mobilenet_v2   quanti_ware_test