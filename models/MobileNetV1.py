import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as backend

def MobileNetV1():
    pass

def _conv_block(inputs,filter,kernel,strides,id):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    x = layers.Conv2D(filter,kernel,strides=strides,name=f'Conv2D:{id}')(inputs)
    x = layers.BatchNormalization(axis=channel_axis,name = f'Conv2D_BN:{id}')(x)
    return layers.ReLU(max_value=6.0,name = f'Conv2D_Relu:{id}')(x)

def _depthwise_conv_block(inputs,pointwise_conv_filters,alpha,
                          depth_mutipliter=1,
                          strides=(1,1),
                          block_id=1):
    if strides == (1,1):
        x = inputs
    else:
        x = layers.ZeroPadding2D()


if __name__ == '__main__':
    import numpy as np
    from silence_tensorflow import silence_tensorflow
    tf.get_logger().setLevel('INFO')

    input_shape = (1, 1, 2, 2)
    x = np.arange(np.prod(input_shape)).reshape(input_shape)
    print(x)
    y = tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
    print(y)

