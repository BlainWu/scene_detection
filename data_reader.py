
import pathlib
import tensorflow as tf
import random
from config import *

AUTOTUNE = tf.data.experimental.AUTOTUNE

shuffle_size = 9940

def random_rot(image):
    times = random.randint(0, 3)
    image = tf.image.rot90(image, k=times)
    return image


def resize_and_rescale(image, label="", train=True):
    image = img_preprocess(image, IMAGE_SIZE, NORMALIZATION, train)
    return image, label


def resize_and_rescale_val(image, label="", train=False):
    image = img_preprocess(image, IMAGE_SIZE, NORMALIZATION, train)
    return image, label


def img_preprocess(image, image_size, normalization, train=True):
    image = tf.cast(image, tf.float32)
    # image = tf.image.rgb_to_hsv(image)
    image = tf.image.resize(image, [image_size, image_size])

    if normalization == "255":
        image = (image / 255.0)
    elif normalization == "imagenet":
        mean_RGB = tf.constant([123.68, 116.779, 109.939], dtype=tf.float32)
        std_RGB = tf.constant([58.393, 57.12, 57.375], dtype=tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.subtract(image, mean_RGB)
        image = tf.divide(image, std_RGB)
    elif normalization == "per":
        image = tf.image.per_image_standardization(image)
    elif normalization == 'self':
        mean_RGB = tf.constant([108.037, 104.083, 95.793], dtype=tf.float32)
        std_RGB = tf.constant([58.102, 54.242, 55.215], dtype=tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.subtract(image, mean_RGB)
        image = tf.divide(image, std_RGB)
    elif normalization == 'None':
        image = image

    if train:
        # if random.random() > 0.5:
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.7, upper=1)
        #image = random_rot(image)
    # # # 随机色相
    # # image = tf.image.random_hue(image, max_delta=0.1)
    # # # 随机饱和度
    # # image = tf.image.random_saturation(image, lower=0.5, upper=1)
    # # # 随机图片质量
    # # image = tf.image.random_jpeg_quality(image, min_jpeg_quality=40, max_jpeg_quality=100)

    return image


def augment(image_label, seed):
    image, label = image_label
    image, label = resize_and_rescale(image, label)

    return image, label

def index_to_onehot(index, class_num):
    onehot = [0] * class_num
    onehot[int(index)] = 1
    return onehot


def get_imgs_labels(dir_root, onehot=True):
    data_root = pathlib.Path(dir_root)
    all_img_paths = list(data_root.glob("*/*"))
    all_img_paths = [str(path) for path in all_img_paths]
    label_names = sorted(str(item.name) for item in data_root.glob('*/') if item.is_dir())
    label_names.sort(key=lambda x: int(x.split('_')[0]))
    if onehot:
        label_to_index = dict(
            (name, index_to_onehot(index, len(label_names))) for index, name in enumerate(label_names))
    else:
        label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_img_labels = [label_to_index[pathlib.Path(path).parent.name]
                      for path in all_img_paths]
    try:
        if (len(all_img_paths) != len(all_img_labels)):
            raise ValueError("图片路径与标签不匹配")
    finally:
        print(f'{dir_root} 图片数量为{len(all_img_paths)}，标签数量为{len(all_img_labels)}')
    return all_img_paths, all_img_labels


def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def get_train_data():
    train_img_paths, train_labels = get_imgs_labels(train_data_path)
    path_ds = tf.data.Dataset.from_tensor_slices(train_img_paths)

    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_labels, tf.int64))
    img_ds = path_ds.map(load_image, num_parallel_calls=AUTOTUNE)

    dataset = tf.data.Dataset.zip((img_ds, label_ds))
    # Create counter and zip together with train dataset
    counter = tf.data.experimental.Counter()
    train_ds = tf.data.Dataset.zip((dataset, (counter, counter)))  # (counter,counter) is seed
    train_ds = (
        train_ds
            .shuffle(buffer_size=shuffle_size)
            .map(augment, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE)
    )
    return train_ds


def get_val_data():
    val_img_paths, val_labels = get_imgs_labels(val_data_path)
    path_ds = tf.data.Dataset.from_tensor_slices(val_img_paths)

    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(val_labels, tf.int64))
    img_ds = path_ds.map(load_image, num_parallel_calls=AUTOTUNE)
    val_ds = tf.data.Dataset.zip((img_ds, label_ds))

    val_ds = (
        val_ds
            .map(resize_and_rescale_val, num_parallel_calls=AUTOTUNE)
            .batch(1)
            .prefetch(AUTOTUNE)
    )
    return val_ds
