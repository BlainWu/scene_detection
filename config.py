IMAGE_SIZE = 224
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

GPU = "0,1"
BATCH_SIZE = 64
EPOCHS_TRAIN = 10
EPOCHS_FINE = 200
FINE_TUNE_START = 0
NORMALIZATION = "per"

train_data_path = '/data2/competition/classification/train_2_8'
val_data_path = '/data2/competition/classification/val_true'