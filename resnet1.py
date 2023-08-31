import tensorflow as tf
from residual_block1 import make_basic_block_layer, make_bottleneck_layer

NUM_CLASSES = 5

def ResNetTypeI(inputs, layer_params, training=None):
    model = tf.keras.Sequential()
    
    # Convolutional Layer 1
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same", input_shape=inputs.shape[1:]))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="same"))

    # Basic Block Layers
    for i in range(4):
        stride = 1 if i == 0 else 2
        model.add(make_basic_block_layer(filter_num=64 * (2**i), blocks=layer_params[i], stride=stride))

    # Global Average Pooling
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # Fully Connected Layer
    model.add(tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.nn.softmax))

    return model



def resnet_18(inputs, training=None):
    return ResNetTypeI(inputs, layer_params=[2, 2, 2, 2], training=training)


def resnet_34(inputs, training=None):
    return ResNetTypeI(inputs, layer_params=[3, 4, 6, 3], training=training)


def ResNetTypeII(inputs, layer_params, training=None):
    conv1 = tf.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same")(inputs)
    bn1 = tf.layers.BatchNormalization()(conv1)
    pool1 = tf.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")(bn1)

    layer1 = make_bottleneck_layer(pool1, filter_num=64, blocks=layer_params[0], training=training)
    layer2 = make_bottleneck_layer(layer1, filter_num=128, blocks=layer_params[1], stride=2, training=training)
    layer3 = make_bottleneck_layer(layer2, filter_num=256, blocks=layer_params[2], stride=2, training=training)
    layer4 = make_bottleneck_layer(layer3, filter_num=512, blocks=layer_params[3], stride=2, training=training)

    avgpool = tf.keras.layers.GlobalAveragePooling2D()(layer4)
    fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)(avgpool)

    return fc


def resnet_50(inputs, training=None):
    return ResNetTypeII(inputs, layer_params=[3, 4, 6, 3], training=training)


def resnet_101(inputs, training=None):
    return ResNetTypeII(inputs, layer_params=[3, 4, 23, 3], training=training)


def resnet_152(inputs, training=None):
    return ResNetTypeII(inputs, layer_params=[3, 8, 36, 3], training=training)