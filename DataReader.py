import pathlib
import tensorflow as tf
import argparse
"""HyperParameters"""
AUTOTUNE = tf.data.experimental.AUTOTUNE
HEIGHT = 128
WIDTH = 128
BATCH_SIZE = 32
SHUFFLE = 1000

parser = argparse.ArgumentParser(description="Load and augment the dataset.")
parser.add_argument('-tp','--train_path',default='./dataset/train',type=str)
parser.add_argument('-vp','--val_path',default='./dataset/val',type=str)
args = parser.parse_args()
train_data_path = args.train_path
val_data_path = args.val_path

def resize_and_rescale(image, label):
  image = tf.cast(image, tf.float32)
  '''Size can be changed'''
  image = tf.image.resize(image, [HEIGHT, WIDTH])
  image = (image / 255.0)
  return image, label

def augment(image_label, seed):
  image, label = image_label
  image, label = resize_and_rescale(image, label)
  image = tf.image.resize_with_crop_or_pad(image, HEIGHT + 6, WIDTH + 6)
  # Make a new seed
  new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
  # Random crop back to the original size
  image = tf.image.stateless_random_crop(
      image, size=[HEIGHT, WIDTH, 3], seed=seed)
  # Random brightness
  image = tf.image.stateless_random_brightness(
      image, max_delta=0.5, seed=new_seed)
  image = tf.clip_by_value(image, 0, 1)
  return image, label

def get_imgs_labels(dir_root):
    # Get all image paths, unsorted
    data_root = pathlib.Path(dir_root)
    all_img_paths = list(data_root.glob("*/*"))
    all_img_paths = [str(path) for path in all_img_paths]
    # generate a dict,sorted (name,index)
    label_names = sorted(str(item.name) for item in data_root.glob('*/') if item.is_dir())
    label_names.sort(key=lambda x: int(x.split('_')[0]))
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_img_labels = [label_to_index[pathlib.Path(path).parent.name]
                      for path in all_img_paths]
    # check the label and imgs
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
    img_ds = path_ds.map(load_image,num_parallel_calls=AUTOTUNE)

    dataset = tf.data.Dataset.zip((img_ds, label_ds))
    # Create counter and zip together with train dataset
    counter = tf.data.experimental.Counter()
    train_ds = tf.data.Dataset.zip((dataset, (counter, counter)))
    train_ds = (
        train_ds
        .shuffle(SHUFFLE)
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
        .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
        )
    return val_ds

