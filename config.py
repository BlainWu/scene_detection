IMAGE_SIZE = 245
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

GPU = "3"
BATCH_SIZE = 32
EPOCHS_TRAIN = 10
EPOCHS_FINE = 300
FINE_TUNE_START = 0
NORMALIZATION = "per"

train_data_path = "/home/share/competition/classification/train_2_8/"
val_data_path = '/home/share/competition/classification/val_true'
test_data_path = '/home/share/competition/classification/test'

current_model_dir = \
    '/home/wupeilin/project/scene_detection/save/t2021_03_09_03_19_mb2_264_t10_f300_nper_b32_fs0_addL2Norm/log_train' #ssd_mobilenet_v2   quanti_ware_test